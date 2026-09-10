"""Quality Control (QC) service.

QC is a gatekeeper, not a fixer. It inspects the CURRENT final assembly (the
successful, non-stale final MP4) plus the scene list / video generations and
returns PASS / WARN / FAIL with structured findings. It never modifies the
video, never regenerates images or clips, and never reassembles.

The implementation is split into pluggable "stages" behind a small interface:

  * ``TechnicalQC``  — V1: automated, offline, FFmpeg-only checks
    (file presence/readability, duration, resolution/aspect, codec, frame
    rate, pixel format, audio presence, decode integrity, file size, scene
    integrity vs. the current clips, duration consistency, and lightweight
    frame sampling for black/frozen video).
  * ``SemanticVisualQC`` — FUTURE extension point for a vision-capable LLM
    review (final frames + storyboard + Project Bible). Not enabled in V1, so
    no AI cost is ever incurred.

PASS/WARN/FAIL semantics
------------------------
Each check carries a ``severity``: a FAIL always blocks, a WARN only blocks
when its severity is "medium" or "high". ``info``-level warnings (e.g. missing
audio, which is optional in V1) are reported but do NOT downgrade the overall
verdict, so a silent clip can still PASS while surfacing the informational
warning in the findings list.

Overall verdict:
  * any FAIL        -> FAILED
  * else any blocking WARN -> WARNINGS
  * else            -> PASSED
"""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backend.storage as storage_mod
from backend.assembly import current_generation, get_ffmpeg_info, probe_video
from backend.config import Settings, get_settings
from backend.models import AssemblyRecord, Project, QCCheck, QCReport, QCStatus

ASPECT_9_16 = 9 / 16
ASPECT_TOLERANCE = 0.02
ACCEPTABLE_PIX_FMTS = {"yuv420p", "yuvj420p", "nv12"}
UNUSUAL_PIX_FMTS = {"yuv422p", "yuv444p", "yuvj422p", "yuvj444p"}


class QCError(Exception):
    """Raised when QC cannot run at all (message is user-facing)."""


def _check(
    name: str,
    status: str,
    severity: str = "info",
    message: str = "",
    measured: object = None,
    expected: object = None,
) -> QCCheck:
    return QCCheck(
        check_name=name,
        status=status,
        severity=severity,
        message=message,
        measured_value=measured,
        expected_value=expected,
    )


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class BaseQCStage(ABC):
    """Interface for a QC stage. V1 ships ``TechnicalQC`` only."""

    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def run(
        self, project: Project, assembly: AssemblyRecord, settings: Settings
    ) -> List[QCCheck]:
        ...


class TechnicalQC(BaseQCStage):
    """Automated, offline, FFmpeg-only technical checks for the final video."""

    name = "technical"

    def run(
        self, project: Project, assembly: AssemblyRecord, settings: Settings
    ) -> List[QCCheck]:
        checks: List[QCCheck] = []
        path = Path(assembly.output_path or "")

        # A. File existence
        exists = path.exists() and path.is_file()
        checks.append(
            _check(
                "file_exists",
                "PASS" if exists else "FAIL",
                "high",
                "Final video file exists." if exists else "Final video file is missing.",
                measured=str(path) if str(path) else None,
                expected="present",
            )
        )
        if not exists:
            checks.append(
                _check(
                    "file_size",
                    "FAIL",
                    "high",
                    "Cannot measure file size - final video file is missing.",
                    measured=0,
                    expected="present",
                )
            )
            return checks

        # J. File size (zero / suspiciously small)
        size = path.stat().st_size
        min_size = settings.qc_min_file_size_bytes
        if size <= 0 or size < min_size:
            checks.append(
                _check(
                    "file_size",
                    "FAIL",
                    "high",
                    f"Final video is suspiciously small ({size} bytes).",
                    measured=size,
                    expected=f">= {min_size} bytes",
                )
            )
        else:
            checks.append(
                _check(
                    "file_size",
                    "PASS",
                    "info",
                    f"File size {size} bytes is reasonable.",
                    measured=size,
                    expected=f">= {min_size} bytes",
                )
            )

        # B. Readability (ffprobe)
        try:
            probe = probe_video(str(path))
        except Exception as e:  # ffprobe failed / no video stream
            message = str(e)
            checks.append(
                _check(
                    "file_readable",
                    "FAIL",
                    "high",
                    f"Video could not be opened for inspection. {message}",
                    measured=str(path),
                    expected="readable MP4",
                )
            )
            return checks
        checks.append(
            _check(
                "file_readable",
                "PASS",
                "info",
                "FFmpeg can open and read the final video.",
                measured=str(path),
                expected="readable MP4",
            )
        )

        # I. Corruption (full decode)
        rc, err = self._decode_check(path)
        if rc == 0 and not err:
            checks.append(
                _check(
                    "corruption",
                    "PASS",
                    "info",
                    "FFmpeg decoded the full video without errors.",
                    measured="clean decode",
                    expected="no decode errors",
                )
            )
        else:
            detail = (err or "decode failed").splitlines()[-1][-160:]
            checks.append(
                _check(
                    "corruption",
                    "FAIL",
                    "high",
                    f"FFmpeg reported errors while decoding: {detail}",
                    measured="decode errors",
                    expected="clean decode",
                )
            )

        # E. Codec
        codec = probe.video_codec or ""
        if codec == "h264":
            checks.append(
                _check("codec", "PASS", "info", "Video is H.264.", measured=codec, expected="h264")
            )
        else:
            checks.append(
                _check(
                    "codec",
                    "FAIL",
                    "high",
                    f"Video codec is {codec or 'unknown'}, expected H.264.",
                    measured=codec,
                    expected="h264",
                )
            )

        # G. Pixel format
        self._check_pixel_format(probe.pix_fmt or "", checks)

        # D. Resolution (reasonable 9:16, configurable target)
        self._check_resolution(probe.width, probe.height, settings, checks)

        #   Aspect ratio (vertical 9:16)
        if probe.width <= 0 or probe.height <= 0:
            checks.append(
                _check(
                    "aspect_ratio",
                    "FAIL",
                    "high",
                    "Resolution could not be determined.",
                    measured=f"{probe.width}x{probe.height}",
                    expected="9:16",
                )
            )
        else:
            ratio = probe.width / probe.height
            shaped = probe.height > probe.width and abs(ratio - ASPECT_9_16) <= ASPECT_TOLERANCE
            if shaped:
                checks.append(
                    _check(
                        "aspect_ratio",
                        "PASS",
                        "info",
                        f"Video is vertical 9:16 ({probe.width}x{probe.height}).",
                        measured=f"{probe.width}x{probe.height}",
                        expected="9:16",
                    )
                )
            else:
                checks.append(
                    _check(
                        "aspect_ratio",
                        "FAIL",
                        "high",
                        f"Video is not vertical 9:16 ({probe.width}x{probe.height}).",
                        measured=f"{probe.width}x{probe.height}",
                        expected="9:16",
                    )
                )

        # C. Duration
        dur = probe.duration
        expected_dur = (
            f"between {settings.qc_min_duration}s and {settings.qc_max_duration}s"
        )
        if dur <= 0:
            checks.append(
                _check(
                    "duration",
                    "FAIL",
                    "high",
                    "Video has no readable duration.",
                    measured=dur,
                    expected=expected_dur,
                )
            )
        elif dur < settings.qc_min_duration or dur > settings.qc_max_duration:
            checks.append(
                _check(
                    "duration",
                    "FAIL",
                    "high",
                    f"Video duration {dur:.1f}s is outside the acceptable "
                    f"range ({settings.qc_min_duration:.0f}s-{settings.qc_max_duration:.0f}s).",
                    measured=round(dur, 3),
                    expected=expected_dur,
                )
            )
        else:
            checks.append(
                _check(
                    "duration",
                    "PASS",
                    "info",
                    f"Duration {dur:.1f}s is within the acceptable range.",
                    measured=round(dur, 3),
                    expected=expected_dur,
                )
            )

        # F. Frame rate
        fps = probe.fps
        expected_fps = (
            f"between {settings.qc_min_fps:.1f} and {settings.qc_max_fps:.1f}"
        )
        if fps <= 0 or fps < settings.qc_min_fps or fps > settings.qc_max_fps:
            checks.append(
                _check(
                    "frame_rate",
                    "FAIL",
                    "high",
                    f"Frame rate {fps:.2f} fps is outside the acceptable range.",
                    measured=round(fps, 3),
                    expected=expected_fps,
                )
            )
        else:
            checks.append(
                _check(
                    "frame_rate",
                    "PASS",
                    "info",
                    f"Frame rate {fps:.2f} fps is acceptable.",
                    measured=round(fps, 3),
                    expected=expected_fps,
                )
            )

        # H. Audio (optional in V1 -> informational warning)
        if probe.has_audio:
            checks.append(
                _check(
                    "audio",
                    "PASS",
                    "info",
                    f"Audio track present ({probe.audio_codec or 'audio'}).",
                    measured=probe.audio_codec or "present",
                    expected="optional",
                )
            )
        else:
            checks.append(
                _check(
                    "audio",
                    "WARN",
                    "info",
                    "No audio track. Audio is optional in V1.",
                    measured="none",
                    expected="optional",
                )
            )

        # 4. Scene integrity
        self._check_scene_integrity(project, assembly, checks)

        # 5. Duration consistency vs. current scene clips
        self._check_duration_consistency(project, assembly, probe.duration, settings, checks)

        # 6. Basic visual failure detection (black / frozen frames)
        self._check_visual(path, probe.duration, settings, checks)

        return checks

    # -- individual sub-checks -------------------------------------------

    @staticmethod
    def _check_pixel_format(pix_fmt: str, checks: List[QCCheck]) -> None:
        if pix_fmt in ACCEPTABLE_PIX_FMTS:
            checks.append(
                _check("pixel_format", "PASS", "info", f"Pixel format {pix_fmt} is compatible.", measured=pix_fmt)
            )
        elif pix_fmt in UNUSUAL_PIX_FMTS:
            checks.append(
                _check(
                    "pixel_format",
                    "WARN",
                    "medium",
                    f"Pixel format {pix_fmt} is unusual but usually playable.",
                    measured=pix_fmt,
                    expected="yuv420p",
                )
            )
        else:
            checks.append(
                _check(
                    "pixel_format",
                    "FAIL",
                    "high",
                    f"Pixel format {pix_fmt or 'unknown'} is not a compatible H.264 format.",
                    measured=pix_fmt,
                    expected="yuv420p",
                )
            )

    @staticmethod
    def _check_resolution(width: int, height: int, settings: Settings, checks: List[QCCheck]) -> None:
        if width <= 0 or height <= 0:
            checks.append(
                _check("resolution", "FAIL", "high", "Resolution could not be determined.", measured=f"{width}x{height}")
            )
            return
        problems = []
        if width % 2 or height % 2:
            problems.append(f"non-even dimensions ({width}x{height})")
        if width < settings.qc_min_width or height < settings.qc_min_height:
            problems.append(
                f"smaller than {settings.qc_min_width}x{settings.qc_min_height}"
            )
        if settings.qc_target_width and settings.qc_target_height:
            if width != settings.qc_target_width or height != settings.qc_target_height:
                problems.append(
                    f"does not match target {settings.qc_target_width}x{settings.qc_target_height}"
                )
        if not problems:
            checks.append(
                _check(
                    "resolution",
                    "PASS",
                    "info",
                    f"Resolution {width}x{height} is acceptable.",
                    measured=f"{width}x{height}",
                    expected="reasonable 9:16",
                )
            )
        elif any(p.startswith("smaller") for p in problems):
            checks.append(
                _check(
                    "resolution",
                    "FAIL",
                    "high",
                    f"Resolution {width}x{height} is too small for a 9:16 short.",
                    measured=f"{width}x{height}",
                    expected=f">= {settings.qc_min_width}x{settings.qc_min_height}",
                )
            )
        else:
            checks.append(
                _check(
                    "resolution",
                    "WARN",
                    "medium",
                    "Resolution is 9:16 but " + "; ".join(problems) + ".",
                    measured=f"{width}x{height}",
                    expected="reasonable 9:16",
                )
            )

    @staticmethod
    def _check_scene_integrity(
        project: Project, assembly: AssemblyRecord, checks: List[QCCheck]
    ) -> None:
        scenes = project.scenes
        ids = [s.id for s in scenes]
        recorded = list(assembly.input_scene_ids)
        clip_ids = list(assembly.input_clip_ids)

        # every scene exists / assembly contains every scene
        present = len(recorded) == len(ids) and set(recorded) == set(ids)
        if present:
            checks.append(
                _check(
                    "scenes_present",
                    "PASS",
                    "info",
                    f"Assembly contains all {len(ids)} scenes.",
                    measured=len(recorded),
                    expected=len(ids),
                )
            )
        else:
            missing = [sid for sid in ids if sid not in recorded]
            checks.append(
                _check(
                    "scenes_present",
                    "FAIL",
                    "high",
                    f"Assembly is missing scene(s): {missing[:5]}",
                    measured=len(recorded),
                    expected=len(ids),
                )
            )

        # ordering matches storyboard order
        if present and recorded == ids:
            checks.append(
                _check(
                    "scene_order",
                    "PASS",
                    "info",
                    "Scene order matches the storyboard.",
                    measured="in storyboard order",
                    expected="storyboard order",
                )
            )
        else:
            checks.append(
                _check(
                    "scene_order",
                    "FAIL",
                    "high",
                    "Assembly scene order does not match the storyboard order.",
                    measured=recorded,
                    expected=ids,
                )
            )

        # assembly uses the CURRENT generation id for every scene
        current_ids = []
        stale_scenes = []
        for scene in scenes:
            gen = current_generation(scene)
            if gen is None or gen.status != "SUCCEEDED" or not gen.output_path:
                stale_scenes.append(scene.description[:40] or scene.id[:8])
                current_ids.append(None)
            else:
                current_ids.append(gen.attempt_id)
        if (
            len(clip_ids) == len(current_ids)
            and clip_ids == current_ids
            and not stale_scenes
        ):
            checks.append(
                _check(
                    "clips_current",
                    "PASS",
                    "info",
                    "Every scene clip in the assembly is the current generation.",
                    measured="all current",
                    expected="current generation per scene",
                )
            )
        else:
            reason = (
                f"assembly built from an older generation for scene(s): {stale_scenes[:3]}"
                if stale_scenes
                else "assembly clip ids do not match the current generations"
            )
            checks.append(
                _check(
                    "clips_current",
                    "FAIL",
                    "high",
                    f"Final video is out of date - {reason}. Reassemble and rerun QC.",
                    measured=clip_ids,
                    expected=current_ids,
                )
            )

    @staticmethod
    def _check_duration_consistency(
        project: Project,
        assembly: AssemblyRecord,
        final_duration: float,
        settings: Settings,
        checks: List[QCCheck],
    ) -> None:
        durations = []
        unreadable = []
        for index, scene in enumerate(project.scenes, start=1):
            gen = current_generation(scene)
            if gen is None or gen.status != "SUCCEEDED" or not gen.output_path:
                unreadable.append(f"scene {index}")
                continue
            clip_path = Path(gen.output_path)
            if not clip_path.is_file():
                unreadable.append(f"scene {index}")
                continue
            try:
                durations.append(probe_video(str(clip_path)).duration)
            except Exception:
                unreadable.append(f"scene {index}")

        if unreadable:
            checks.append(
                _check(
                    "duration_consistency",
                    "WARN",
                    "medium",
                    f"Could not measure all clip durations ({', '.join(unreadable)}); "
                    "duration consistency not verified.",
                )
            )
            return
        if not durations:
            checks.append(
                _check("duration_consistency", "PASS", "info", "No scenes to compare.", measured=0, expected=0)
            )
            return

        total = sum(durations)
        delta = abs(total - final_duration)
        if delta <= settings.qc_duration_tolerance:
            checks.append(
                _check(
                    "duration_consistency",
                    "PASS",
                    "info",
                    f"Final duration {final_duration:.1f}s matches the scene clips "
                    f"({total:.1f}s, diff {delta:.2f}s).",
                    measured=round(delta, 3),
                    expected="within tolerance",
                )
            )
        elif delta <= settings.qc_duration_warn_tolerance:
            checks.append(
                _check(
                    "duration_consistency",
                    "WARN",
                    "medium",
                    f"Final duration {final_duration:.1f}s differs from the scene clips "
                    f"({total:.1f}s, diff {delta:.2f}s).",
                    measured=round(delta, 3),
                    expected=f"diff <= {settings.qc_duration_tolerance}s",
                )
            )
        else:
            checks.append(
                _check(
                    "duration_consistency",
                    "FAIL",
                    "high",
                    f"Final duration {final_duration:.1f}s does not match the scene clips "
                    f"({total:.1f}s, diff {delta:.2f}s).",
                    measured=round(delta, 3),
                    expected=f"diff <= {settings.qc_duration_warn_tolerance}s",
                )
            )

    def _check_visual(
        self, path: Path, duration: float, settings: Settings, checks: List[QCCheck]
    ) -> None:
        if duration <= 0:
            return
        ffmpeg = get_ffmpeg_info()
        if not ffmpeg.path:
            return
        stats = self._sample_frame_stats(
            ffmpeg.path, path, duration, settings.qc_visual_frames
        )
        if stats is None:
            checks.append(
                _check(
                    "visual_black",
                    "WARN",
                    "medium",
                    "Could not sample frames for a visual check.",
                )
            )
            return

        black_frames = sum(1 for m in stats["means"] if m < settings.qc_black_luma)
        black_frac = black_frames / len(stats["means"]) if stats["means"] else 0.0
        if black_frac >= 0.5:
            checks.append(
                _check(
                    "visual_black",
                    "FAIL",
                    "high",
                    f"Video looks black or empty ({black_frames}/{len(stats['means'])} "
                    "sampled frames are near-black).",
                    measured=round(black_frac, 2),
                    expected="< 0.5 black frames",
                )
            )
        else:
            checks.append(
                _check(
                    "visual_black",
                    "PASS",
                    "info",
                    "Sampled frames are not black.",
                    measured=round(black_frac, 2),
                    expected="< 0.5 black frames",
                )
            )

        if stats.get("frame_deltas"):
            mean_delta = sum(stats["frame_deltas"]) / len(stats["frame_deltas"])
            if mean_delta < settings.qc_freeze_delta:
                checks.append(
                    _check(
                        "visual_freeze",
                        "WARN",
                        "medium",
                        f"Video appears frozen or static (mean frame delta {mean_delta:.3f}).",
                        measured=round(mean_delta, 4),
                        expected=f">= {settings.qc_freeze_delta}",
                    )
                )
            else:
                checks.append(
                    _check(
                        "visual_freeze",
                        "PASS",
                        "info",
                        "Motion detected between sampled frames.",
                        measured=round(mean_delta, 4),
                        expected=f">= {settings.qc_freeze_delta}",
                    )
                )

    # -- ffmpeg helpers --------------------------------------------------

    @staticmethod
    def _decode_check(path: Path) -> Tuple[int, str]:
        ffmpeg = get_ffmpeg_info()
        cmd = [
            ffmpeg.path,
            "-v", "error",
            "-i", str(path),
            "-f", "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return proc.returncode, (proc.stderr or "").strip()

    @staticmethod
    def _sample_frame_stats(
        ffmpeg_path: str,
        path: Path,
        duration: float,
        max_frames: int,
        width: int = 56,
        height: int = 100,
    ) -> Optional[Dict[str, List[float]]]:
        """Decode a handful of evenly spaced grayscale frames and return stats."""
        n = max(2, min(max_frames, max(2, int(duration))))
        interval = max(duration / n, 0.01)
        cmd = [
            ffmpeg_path,
            "-v", "error",
            "-i", str(path),
            "-vf", f"fps=1/{interval:.4f},scale={width}:{height},format=gray",
            "-frames:v", str(n),
            "-f", "rawvideo",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        data = proc.stdout
        fs = width * height
        frame_count = len(data) // fs
        if frame_count == 0:
            return None
        frames = [data[i * fs:(i + 1) * fs] for i in range(frame_count)]
        means: List[float] = []
        for frame in frames:
            means.append(sum(frame) / fs)
        deltas: List[float] = []
        for a, b in zip(frames, frames[1:]):
            deltas.append(sum(abs(x - y) for x, y in zip(a, b)) / fs)
        return {"means": means, "frame_deltas": deltas}


class SemanticVisualQC(BaseQCStage):
    """FUTURE extension point: final frames + storyboard + Project Bible -> vision LLM.

    Not enabled in V1 — technical checks suffice and enabling it would incur AI
    cost. Subclass/override ``run`` (and set ``enabled=True``) in a later phase
    to add a semantic review; the orchestrator below already discovers stages.
    """

    name = "semantic"
    enabled = False  # never run in V1

    def run(
        self, project: Project, assembly: AssemblyRecord, settings: Settings
    ) -> List[QCCheck]:
        raise NotImplementedError(
            "SemanticVisualQC is a future stage and is not implemented."
        )


class QCService:
    """Orchestrates QC stages and produces a persistent QC report."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        # Order matters: technical first. A future phase can enable
        # SemanticVisualQC by swapping in an implementing stage class.
        self.stages: List[BaseQCStage] = [
            s for s in (TechnicalQC(), SemanticVisualQC()) if s.enabled
        ]

    def current_assembly(self, project: Project) -> Optional[AssemblyRecord]:
        """The assembly QC should check: the live, non-stale successful one."""
        for record in reversed(project.assemblies):
            if record.status in ("SUCCEEDED", "STALE"):
                return record
        return None

    def can_run(self, project: Project, assembly: Optional[AssemblyRecord] = None) -> Tuple[bool, str]:
        """Precondition check: QC must never run against a stale assembly."""
        if not get_ffmpeg_info().available:
            return False, "FFmpeg is not available; QC cannot run."
        assembly = assembly or self.current_assembly(project)
        if assembly is None:
            return False, "No assembly has been created yet."
        if assembly.is_stale or (project.current_assembly_id and project.current_assembly_id != assembly.assembly_id):
            return False, "The final video is out of date - reassemble it before running QC."
        if not assembly.succeeded:
            return False, "QC can only check a successful assembly."
        if not Path(assembly.output_path or "").is_file():
            # Missing file is a legitimate FAIL, so let QC run and report it.
            pass
        return True, ""

    def assemble_report(
        self, project: Project, assembly: AssemblyRecord, report: Optional[QCReport] = None
    ) -> QCReport:
        """Run every enabled stage against the assembly and build the report.

        ``report`` may be a pre-created RUNNING report (e.g. already persisted
        by the master agent for crash recovery); its ids/timestamps are kept.
        """
        if report is None:
            report = QCReport(
                project_id=project.id,
                assembly_id=assembly.assembly_id,
                status=QCStatus.RUNNING,
                started_at=_now(),
            )
        report.project_id = project.id
        report.assembly_id = assembly.assembly_id
        report.status = QCStatus.RUNNING
        if not report.started_at:
            report.started_at = _now()
        report.raw.setdefault("qc_version", "1.0")
        report.raw.setdefault("stages", [s.name for s in self.stages])
        try:
            checks: List[QCCheck] = []
            for stage in self.stages:
                checks.extend(stage.run(project, assembly, self._settings))
            report.checks = checks
            report.findings = [c.message for c in checks if c.status != "PASS"]
            report.status = self.overall(checks)
            report.summary = self._summary(report.status, report.findings)
        except QCError:
            raise
        except Exception as e:  # never lose the report - mark it FAILED
            report.status = QCStatus.FAILED
            report.checks = [
                _check("qc_run", "FAIL", "high", f"QC could not complete: {e}")
            ]
            report.findings = [report.checks[0].message]
            report.summary = "Automated QC failed to complete."
        report.completed_at = _now()
        return report

    @staticmethod
    def overall(checks: List[QCCheck]) -> QCStatus:
        if any(c.status == "FAIL" for c in checks):
            return QCStatus.FAILED
        if any(c.blocking for c in checks):
            return QCStatus.WARNINGS
        return QCStatus.PASSED

    @staticmethod
    def _summary(status: QCStatus, findings: List[str]) -> str:
        if status == QCStatus.PASSED:
            return "All automated checks passed."
        if status == QCStatus.WARNINGS:
            return "Automated checks passed with warnings: " + "; ".join(findings[:3])
        return "Automated QC failed: " + "; ".join(findings[:3])


def latest_qc_report(project: Project) -> Optional[QCReport]:
    """The most recent QC report (any status) for a project."""
    return project.qc_reports[-1] if project.qc_reports else None
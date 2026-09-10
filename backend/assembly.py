"""FFmpeg video assembly service.

Takes every scene's current (successful) clip — in storyboard order — and
concatenates them into ONE vertical 9:16 MP4 at ``projects/<id>/final/final.mp4``.

Responsibilities:
  * validate inputs (locked image, successful current clip, readable file)
  * inspect each clip with ffprobe (resolution / fps / codec / audio)
  * normalize only clips that differ from the common target profile
  * concatenate (direct stream copy when every clip already matches)
  * validate the final output and record it on the project

Assembly is FFmpeg-only: it never regenerates images or video clips. Failure
reports identify which scene is the problem. The service is provider-agnostic
and never talks to external APIs.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.ffmpeg import get_ffmpeg_info
from backend.models import AssemblyRecord, AssemblyStatus, Project, Scene, VideoGeneration
import backend.storage as storage_mod

# Target profile used for normalization. Already-compatible clips are left
# untouched (stream copy) to avoid unnecessary re-encoding.
TARGET_VIDEO_CODEC = "h264"
TARGET_PIX_FMT = "yuv420p"
TARGET_AUDIO_CODEC = "aac"
ASPECT_9_16 = 9 / 16
ASPECT_TOLERANCE = 0.02
DURATION_TOLERANCE = 0.75  # seconds of slack vs. the sum of clip durations


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssemblyError(Exception):
    """Raised when the inputs cannot be assembled as-is (message is user-facing)."""


@dataclass
class ClipInfo:
    """Media summary for one clip, derived from ffprobe."""

    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str = ""
    pix_fmt: str = ""
    duration: float = 0.0
    has_audio: bool = False
    audio_codec: str = ""

    @property
    def portrait9x16(self) -> bool:
        if not (self.width and self.height):
            return False
        if self.height <= self.width:
            return False
        ratio = self.width / self.height
        return abs(ratio - ASPECT_9_16) <= ASPECT_TOLERANCE

    def compatible_with(self, other: "ClipInfo") -> bool:
        """True when two clips can be concatenated with a plain stream copy."""
        return (
            self.width == other.width
            and self.height == other.height
            and round(self.fps, 3) == round(other.fps, 3)
            and self.video_codec == other.video_codec
            and self.pix_fmt == other.pix_fmt
            and self.has_audio == other.has_audio
            and self.audio_codec == other.audio_codec
        )


@dataclass
class ClipInput:
    """A validated clip the assembly will concatenate (per scene, in order)."""

    index: int
    scene: Scene
    generation: VideoGeneration
    path: Path
    info: Optional[ClipInfo] = None


def _ffprobe_path() -> str:
    """Locate ffprobe (prefer it next to ffmpeg; fall back to PATH)."""
    info = get_ffmpeg_info()
    if info.available and info.path:
        sibling = Path(info.path).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    raise AssemblyError("ffprobe is not available; cannot assemble videos.")


def _parse_fraction(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        num, den = value.split("/")
        if float(den) == 0:
            return 0.0
        return float(num) / float(den)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return 0.0


def probe_video(path: str) -> ClipInfo:
    """Probe a media file with ffprobe and summarize its streams."""
    probe = _ffprobe_path()
    cmd = [
        probe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise AssemblyError(f"Could not inspect video file: {e}")
    if proc.returncode != 0:
        raise AssemblyError(
            f"Video file could not be read ({os.path.basename(path)})."
        )

    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError:
        raise AssemblyError(f"Video file could not be read ({os.path.basename(path)}).")

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise AssemblyError(f"Video file contains no video stream ({os.path.basename(path)}).")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    simple_format = data.get("format", {})
    duration = float(video.get("duration") or 0.0)
    if not duration:
        duration = float(simple_format.get("duration") or 0.0)

    return ClipInfo(
        path=path,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        video_codec=str(video.get("codec_name") or "").lower(),
        pix_fmt=str(video.get("pix_fmt") or "").lower(),
        duration=duration,
        has_audio=audio is not None,
        audio_codec=str((audio or {}).get("codec_name") or "").lower(),
    )


def current_generation(scene: Scene) -> Optional[VideoGeneration]:
    """The scene's current generation (by current_video_id, else latest)."""
    if scene.current_video_id:
        for gen in reversed(scene.video_generations):
            if gen.attempt_id == scene.current_video_id:
                return gen
    return scene.video_generations[-1] if scene.video_generations else None


class AssemblyService:
    """Validates, plans, and runs FFmpeg concatenation for a project's scenes."""

    def __init__(self, *, settings: Optional[object] = None):
        # settings is accepted for parity with other services but assembly has
        # no AI configuration to honor.
        self._settings = settings

    # -- validation / planning ------------------------------------------

    def plan(self, project: Project) -> List[ClipInput]:
        """Validate every scene and return ordered, readable clip inputs.

        Raises ``AssemblyError`` listing every problem with a scene reference
        (1-based, matching the storyboard order).
        """
        ffmpeg = get_ffmpeg_info()
        if not ffmpeg.available or not ffmpeg.path:
            raise AssemblyError(
                "FFmpeg is not installed; can't assemble the final video."
            )

        problems: List[str] = []
        inputs: List[ClipInput] = []
        for index, scene in enumerate(project.scenes, start=1):
            label = f"Scene {index}"

            if not scene.selected_image:
                problems.append(f"{label} has no locked (selected) image.")
                continue
            gen = current_generation(scene)
            if gen is None or gen.status != "SUCCEEDED" or not gen.output_path:
                problems.append(f"{label} has no successful video clip yet.")
                continue
            path = Path(gen.output_path)
            if not path.is_file():
                problems.append(f"{label} video file is missing or unreadable.")
                continue
            inputs.append(
                ClipInput(
                    index=index,
                    scene=scene,
                    generation=gen,
                    path=path.resolve(),
                )
            )

        if problems:
            raise AssemblyError(" ".join(problems))
        return inputs

    def readiness(self, project: Project) -> Dict[str, object]:
        """Non-raising validation summary for the UI: ready + human issues."""
        try:
            inputs = self.plan(project)
            issue = ""
            ready = len(inputs) == len(project.scenes) and bool(inputs)
        except AssemblyError as e:
            ready = False
            issue = str(e)
        return {
            "ready": ready,
            "issues": [f.strip() for f in issue.split(".") if f.strip()] if issue else [],
            "scene_count": len(project.scenes),
        }

    def _ordered(self, project: Project) -> List[ClipInput]:
        return sorted(self.plan(project), key=lambda c: c.index)

    def matching_record(self, project: Project, inputs: List[ClipInput]) -> Optional[AssemblyRecord]:
        """A SUCCEEDED assembly whose exact input clips equal the current ones."""
        current_ids = [c.generation.attempt_id for c in inputs]
        for record in reversed(project.assemblies):
            if not record.succeeded:
                continue
            if len(record.input_clip_ids) != len(current_ids):
                continue
            if record.input_clip_ids == current_ids:
                return record
        return None

    def live_record(self, project: Project) -> Optional[AssemblyRecord]:
        """The most recent record describing the on-disk final file.

        SUCCEEDED (a live final) or STALE (a final that was later invalidated by
        a regenerated clip but whose file may still exist for preview).
        """
        for record in reversed(project.assemblies):
            if record.status in (AssemblyStatus.SUCCEEDED, AssemblyStatus.STALE):
                return record
        return None

    def is_stale(self, project: Project) -> bool:
        """True when the project's current clips differ from its final assembly."""
        record = self.live_record(project)
        if record is None:
            return False
        if record.is_stale:
            return True
        try:
            current_ids = [c.generation.attempt_id for c in self._ordered(project)]
        except AssemblyError:
            return True
        return record.input_clip_ids != current_ids

    # -- execution -------------------------------------------------------

    def assemble(
        self, project: Project
    ) -> Tuple[AssemblyRecord, bool, Optional[str]]:
        """Run the assembly. Returns (record, reused, error).

        ``reused=True`` means an existing SUCCEEDED assembly still matches the
        current clips (no work needed). ``error`` is a user-facing message when
        the attempt failed; the returned record carries the persisted details.
        """
        inputs = self._ordered(project)

        existing = self.matching_record(project, inputs)
        if existing and Path(existing.output_path).is_file():
            return existing, True, None

        record = AssemblyRecord(
            project_id=project.id,
            status=AssemblyStatus.PENDING,
            input_scene_ids=[c.scene.id for c in inputs],
            input_clip_ids=[c.generation.attempt_id for c in inputs],
            input_clip_paths=[str(c.path) for c in inputs],
            scene_count=len(inputs),
        )
        try:
            self._build(project, inputs, record)
            record.status = AssemblyStatus.SUCCEEDED
            record.completed_at = _now()
            return record, False, None
        except AssemblyError as e:
            record.status = AssemblyStatus.FAILED
            record.error = str(e)
            record.completed_at = _now()
            return record, False, str(e)

    def _build(self, project: Project, inputs: List[ClipInput], record: AssemblyRecord) -> None:
        """Inspect, (optionally) normalize, concatenate, and validate the final."""
        record.status = AssemblyStatus.PROCESSING
        record.started_at = _now()
        started = time.monotonic()
        ffmpeg = get_ffmpeg_info()
        ffmpeg_bin = ffmpeg.path

        # 1. Inspect every clip.
        for c in inputs:
            c.info = probe_video(str(c.path))
            info = c.info
            if info.duration <= 0:
                raise AssemblyError(
                    f"Scene {c.index} video has no readable duration."
                )
            if not info.portrait9x16:
                raise AssemblyError(
                    f"Scene {c.index} video is not vertical 9:16 "
                    f"({info.width}x{info.height})."
                )

        # 2. Decide the common target profile (from the first clip).
        first = inputs[0].info
        target_width, target_height = first.width, first.height
        target_fps = first.fps or 24.0
        has_audio = any(c.info.has_audio for c in inputs)
        target = {
            "width": target_width,
            "height": target_height,
            "fps": target_fps,
            "video_codec": TARGET_VIDEO_CODEC,
            "pix_fmt": TARGET_PIX_FMT,
            "has_audio": has_audio,
            "audio_codec": TARGET_AUDIO_CODEC if has_audio else "",
        }

        # 3. Normalize only the clips that don't already match the profile.
        normalized = []
        needs_normalization = any(
            not _clip_matches_target(c.info, target) for c in inputs
        )

        with tempfile.TemporaryDirectory(prefix="ai_shorts_assembly_") as tmp:
            tmp_dir = Path(tmp)
            concat_sources: List[str] = []
            if needs_normalization:
                for c in inputs:
                    out = tmp_dir / f"norm_{c.index}.mp4"
                    self._render_normalized(c, out, target)
                    normalized.append(out)
                    concat_sources.append(str(out))
            else:
                concat_sources = [str(c.path) for c in inputs]

            # 4. Concatenate in storyboard order (fast path: stream copy).
            list_file = tmp_dir / "concat.txt"
            list_file.write_text(
                "\n".join(f"file '{_escape_path(s)}'" for s in concat_sources) + "\n"
            )
            final_tmp = tmp_dir / "final_tmp.mp4"
            concat_cmd = [
                ffmpeg_bin,
                "-y",
                "-fflags", "+genpts+igndts",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(final_tmp),
            ]
            proc = self._run_capture(concat_cmd)
            if proc.returncode != 0:
                raise AssemblyError(_ffmpeg_failure("concatenate the clips", proc))

            # 5. Persist into the project's final/ directory (atomic replace).
            out_dir = storage_mod.project_asset_dir(project.id, "final")
            final_path = out_dir / "final.mp4"
            os.replace(str(final_tmp), str(final_path))

            # 6. Validate the output.
            probe = probe_video(str(final_path))
            if probe.duration <= 0:
                raise AssemblyError("The assembled video has no readable duration.")
            expected = sum(c.info.duration for c in inputs)
            if abs(probe.duration - expected) > DURATION_TOLERANCE:
                raise AssemblyError(
                    "Assembled video duration does not match the source clips."
                )
            if probe.width != target_width or probe.height != target_height:
                raise AssemblyError(
                    "Assembled video has an unexpected resolution."
                )

            record.output_path = str(final_path)
            record.output_duration = round(probe.duration, 3)
            record.output_width = probe.width
            record.output_height = probe.height
            record.output_codec = probe.video_codec
            record.output_size_bytes = final_path.stat().st_size
            record.ffmpeg_version = (ffmpeg.version or "").splitlines()[0] if ffmpeg.version else ""
            record.raw["action"] = "reassemble" if self.live_record(project) else "assemble"
            record.raw["exec_seconds"] = round(time.monotonic() - started, 3)
            record.raw["normalized"] = needs_normalization
            record.raw["normalized_clip_count"] = len(normalized)

    # -- ffmpeg helpers --------------------------------------------------

    def _render_normalized(self, clip: ClipInput, out: Path, target: Dict) -> None:
        """Re-encode ONE clip to the common target profile (no stretching)."""
        ffmpeg = get_ffmpeg_info()
        cmd = [ffmpeg.path, "-y", "-i", str(clip.path)]
        if target["has_audio"] and not clip.info.has_audio:
            # Add a silent track so every clip shares an audio stream.
            cmd += [
                "-f", "lavfi",
                "-t", f"{clip.info.duration:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]
        scale = (
            f"scale={target['width']}:{target['height']}"
            ":force_original_aspect_ratio=decrease,"
            f"pad={target['width']}:{target['height']}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1"
        )
        cmd += [
            "-vf", scale,
            "-r", f"{target['fps']:.3f}",
            "-pix_fmt", TARGET_PIX_FMT,
            "-c:v", "libx264",
            "-profile:v", "high",
            "-preset", "veryfast",
        ]
        if target["has_audio"]:
            cmd += ["-c:a", TARGET_AUDIO_CODEC, "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", "-f", "mp4", str(out)]
        proc = self._run_capture(cmd)
        if proc.returncode != 0:
            raise AssemblyError(
                f"Scene {clip.index} video could not be normalized for assembly."
            )

    @staticmethod
    def _run_capture(cmd: List[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise AssemblyError(f"FFmpeg could not run: {e}")

    @staticmethod
    def _ffmpeg_tail(proc: subprocess.CompletedProcess, limit: int = 400) -> str:
        text = (proc.stderr or proc.stdout or "").strip()
        return text[-limit:]


def _clip_matches_target(info: ClipInfo, target: Dict) -> bool:
    return (
        info.video_codec == target["video_codec"]
        and info.pix_fmt == target["pix_fmt"]
        and info.width == target["width"]
        and info.height == target["height"]
        and round(info.fps, 3) == round(target["fps"], 3)
        and info.has_audio == target["has_audio"]
        and (info.audio_codec == target["audio_codec"] if target["has_audio"] else True)
    )


def _escape_path(path: str) -> str:
    # concat demuxer quoting: single-quote escape on Windows-style and odd cases.
    return path.replace("\\", "/").replace("'", "'\\''")


def _ffmpeg_failure(action: str, proc: subprocess.CompletedProcess) -> str:
    tail = AssemblyService._ffmpeg_tail(proc)
    detail = tail.splitlines()[-1] if tail else "no detail available"
    # Keep the user-facing message free of raw FFmpeg dumps.
    return f"FFmpeg failed to {action}. {detail[-120:]}"
"""Tests for the Quality Control layer.

QC inspects the CURRENT final assembly and returns PASS/WARN/FAIL with
structured findings. It never regenerates anything and never reassembles.
Everything runs offline against real FFmpeg-generated mock files (no external
AI / video APIs). Covers the full list of required checks, completeness of the
check set, staleness/invalidation/rerun behavior, persistence, restart
recovery, state transitions, and the API surface.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_qc_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from agents.image import ImageAgent
from agents.image_schemas import ImagePromptOutput
from agents.master import MasterAgent, InvalidTransitionError
from agents.video import VideoAgent
from agents.video_schemas import VideoPromptOutput
from backend.assembly import AssemblyError, AssemblyService, probe_video
from backend.config import Settings
from backend.ffmpeg import get_ffmpeg_info
from backend.models import (
    AssemblyRecord,
    AssemblyStatus,
    ProjectStatus,
    QCCheck,
    QCReport,
    QCStatus,
    VideoGeneration,
)
from backend.providers.registry import get_image_provider
from backend.providers.video import MockVideoProvider
from backend.qc import QCError, QCService, TechnicalQC
from backend.storage import load_project, save_project


@pytest.fixture(autouse=True)
def clean_storage():
    storage_mod.STORAGE_DIR = TEST_STORAGE
    storage_mod.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for f in storage_mod.STORAGE_DIR.glob("*.json"):
        f.unlink()
    yield
    for f in storage_mod.STORAGE_DIR.glob("*.json"):
        f.unlink()


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setenv("VIDEO_POLL_INTERVAL", "0.001")
    monkeypatch.setenv("VIDEO_POLL_TIMEOUT", "10")


VIDEO_PROMPT = dict(
    global_continuity="same single plant, same dark soil, locked-down macro camera",
    scene_action="the shoot pushes upward as the bud swells open",
    motion="very slow, continuous upward growth only",
    environment="identical dark soil, nothing else changes",
    camera="completely locked-down, zero camera movement",
    lighting="same warm golden-hour side lighting throughout",
    timelapse="4-second growth timelapse, chronological",
    negative_constraints="no people, no text, no flicker",
    prompt=(
        "Macro 4-second 9:16 timelapse of a single plant shoot pushing "
        "upward as its bud swells open on the same dark moist soil, camera "
        "locked down, warm golden-hour lighting, photorealistic macro look"
    ),
)


class FakeImageLLM:
    def __call__(self, **kwargs):
        return ImagePromptOutput(
            global_continuity="plant, soil, camera, lighting",
            scene_action="seed cracking",
            prompt_a="Macro close-up of a seed cracking open on dark moist soil",
            prompt_b="Bird's-eye view of the seed on the same soil",
            composition_notes="alternate framing",
        ).model_dump()


class FakeLLM:
    def __call__(self, **kwargs):
        return VideoPromptOutput(**VIDEO_PROMPT).model_dump()


def make_settings(tmp_path):
    return Settings(env={"STORAGE_PATH": str(tmp_path), "VIDEO_POLL_INTERVAL": "0.001"})


def qc_settings(tmp_path, extra=None):
    env = {"STORAGE_PATH": str(tmp_path), "VIDEO_POLL_INTERVAL": "0.001"}
    env.update(extra or {})
    return Settings(env=env)


def svc(settings=None):
    return QCService(settings=settings)


def make_video_project(tmp_path, n=3, videos_generated=True):
    """Same offline pipeline helper as test_assembly: mock story+images+videos."""
    agent = MasterAgent.create("Seed growing from seed to flowering plant")
    agent.start_planning()
    agent.submit_storyboard(
        storyboard={
            "title": "Seed to Bloom",
            "summary": "A seed grows.",
            "scenes": [
                {
                    "scene_number": i,
                    "title": f"Scene {i}",
                    "visual_description": f"visual description {i}",
                    "action": f"action {i}",
                    "estimated_duration": 4,
                    "continuity_notes": f"continuity notes {i}",
                }
                for i in range(1, n + 1)
            ],
        },
        scenes=[
            {"description": f"Scene {i} visual", "duration_seconds": 4}
            for i in range(1, n + 1)
        ],
        project_bible={
            "subject": "seed",
            "environment": "dark soil",
            "time_lighting": "evening golden hour",
            "camera": "locked-down macro",
            "visual_style": "Photorealistic",
            "aspect_ratio": "9:16",
            "target_duration": 30.0,
        },
    )
    agent.approve_story()
    agent.images_generated()
    img = ImageAgent(
        llm=FakeImageLLM(),
        image_provider=get_image_provider(make_settings(tmp_path)),
    )
    agent = MasterAgent.load(agent.project.id)
    img.run(agent.project)
    agent.select_images({s.id: "A" for s in agent.project.scenes})
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=MockVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)
    agent = MasterAgent.load(agent.project.id)
    if videos_generated:
        agent.videos_generated()
    return agent


def assemble_project(agent):
    r, reused, error = AssemblyService().assemble(agent.project)
    agent.begin_assembly()
    if error:
        agent.assembly_failed(error, record=r)
        return r, reused, error
    agent.assembly_complete(record=r)
    return r, reused, error


def render_at(path, *, width=576, height=1024, duration=4, fps=24,
              codec="libx264", pix_fmt="yuv420p", audio=False, source="testsrc"):
    """Render a small real MP4 with FFmpeg."""
    if source.startswith("testsrc"):
        graph = f"{source}=size={width}x{height}:rate={fps}:duration={duration}"
    else:
        # e.g. color=c=0x808080 (the color source takes its own c= option)
        graph = f"{source}:size={width}x{height}:rate={fps}:duration={duration}"
    cmd = [
        get_ffmpeg_info().path,
        "-y",
        "-f", "lavfi",
        "-i", graph,
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3.5"]
        cmd += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    cmd += ["-pix_fmt", pix_fmt, "-c:v", codec, "-movflags", "+faststart", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:400])
    return Path(path)


def scene_index(agent, index):
    return agent.project.scenes[index]


def set_scene_clip(agent, index, path, duration_seconds=4):
    """Point a scene's current clip at a specific MP4 (as a new attempt)."""
    scene = scene_index(agent, index)
    gen = VideoGeneration(
        project_id=agent.project.id,
        scene_id=scene.id,
        provider="mock",
        model="mock-video-model",
        status="SUCCEEDED",
        prompt=scene.video_prompt or VIDEO_PROMPT["prompt"],
        source_image_id=scene.selected_image_id or "A",
        source_image=scene.selected_image,
        duration_seconds=duration_seconds,
        output_path=str(path),
    )
    scene.video_generations.append(gen)
    scene.current_video_id = gen.attempt_id
    scene.video_clip = gen.output_path
    save_project(agent.project)


def fabricate_project(tmp_path, final_path, *, n=1, clip_path=None, record=None):
    """Build a READY_FOR_REVIEW project whose current assembly points at files.

    ``final_path`` (and each scene's ``clip_path``) must already exist as real
    MP4s. The record's input ids are kept consistent with the scenes so the
    assembly is NOT considered stale (mimicking a freshly assembled final).
    """
    clip_path = clip_path or final_path
    agent = MasterAgent.create("QC fabricate")
    agent.start_planning()
    agent.submit_storyboard(
        storyboard={
            "title": "QC",
            "summary": "fabricated",
            "scenes": [
                {
                    "scene_number": i,
                    "title": f"Scene {i}",
                    "visual_description": f"visual description {i}",
                    "action": f"action {i}",
                    "estimated_duration": 4,
                    "continuity_notes": "",
                }
                for i in range(1, n + 1)
            ],
        },
        scenes=[
            {"description": f"Scene {i} visual", "duration_seconds": 4}
            for i in range(1, n + 1)
        ],
        project_bible={"subject": "fabricated"},
    )
    agent.approve_story()
    agent.images_generated()
    gens = []
    for scene in agent.project.scenes:
        scene.selected_image = "sel.png"
        scene.selected_image_id = "A"
        gen = VideoGeneration(
            project_id=agent.project.id,
            scene_id=scene.id,
            provider="mock",
            model="mock-video-model",
            status="SUCCEEDED",
            prompt=VIDEO_PROMPT["prompt"],
            source_image_id="A",
            source_image="sel.png",
            duration_seconds=4,
            output_path=str(clip_path),
        )
        scene.video_generations.append(gen)
        scene.current_video_id = gen.attempt_id
        scene.video_clip = gen.output_path
        gens.append(gen)
    rec = record or AssemblyRecord(
        project_id=agent.project.id,
        status=AssemblyStatus.SUCCEEDED,
        input_scene_ids=[s.id for s in agent.project.scenes],
        input_clip_ids=[g.attempt_id for g in gens],
        input_clip_paths=[str(clip_path) for _ in gens],
        scene_count=n,
        output_path=str(final_path),
    )
    agent.project.assemblies = [rec]
    agent.project.current_assembly_id = rec.assembly_id
    agent.project.final_video = str(final_path)
    agent.project.status = ProjectStatus.READY_FOR_REVIEW
    save_project(agent.project)
    return agent


def run_qc(agent, settings=None, svc_override=None):
    """Emulate the API /qc/run flow at the master level (returns the report)."""
    qcs = svc_override or svc(settings)
    assembly = agent.successful_assembly()
    assert assembly is not None, "no current assembly"
    agent.start_quality_check()
    report = QCReport(
        project_id=agent.project.id,
        assembly_id=assembly.assembly_id,
        status=QCStatus.RUNNING,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    agent.qc_running(report)
    report = qcs.assemble_report(agent.project, assembly, report)
    agent.qc_complete(report)
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    return report


def find_check(report, name):
    return next((c for c in report.checks if c.check_name == name), None)


# =========================================================================
# 1-8. Technical checks (PASS/FAIL/WARN) + completeness
# =========================================================================


def test_valid_final_video_passes_qc(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    report = run_qc(agent)
    assert report.status == QCStatus.PASSED
    assert report.checks, "expected checks to be populated"
    assert report.summary == "All automated checks passed."


def test_check_set_is_complete(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    report = run_qc(agent)
    names = {c.check_name for c in report.checks}
    expected = {
        "file_exists", "file_readable", "duration", "resolution",
        "aspect_ratio", "codec", "frame_rate", "pixel_format", "audio",
        "corruption", "file_size", "scenes_present", "scene_order",
        "clips_current", "duration_consistency", "visual_black",
        "visual_freeze",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


def test_missing_final_video_fails(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    record, _, _ = assemble_project(agent)
    Path(record.output_path).unlink()
    report = run_qc(agent)
    assert report.status == QCStatus.FAILED
    assert find_check(report, "file_exists").status == "FAIL"


def test_corrupt_final_video_fails(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    record, _, _ = assemble_project(agent)
    path = Path(record.output_path)
    sz = path.stat().st_size
    with path.open("r+b") as f:
        f.seek(sz // 2)
        f.write(b"\x00" * 4096)  # header stays intact, mdat is garbage
    report = run_qc(agent)
    assert report.status == QCStatus.FAILED
    assert find_check(report, "file_readable").status == "PASS"
    assert find_check(report, "corruption").status == "FAIL"


def test_wrong_orientation_fails(tmp_path):
    final = render_at(tmp_path / "landscape.mp4", width=1024, height=576)
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent)
    assert report.status == QCStatus.FAILED
    check = find_check(report, "aspect_ratio")
    assert check.status == "FAIL"
    assert check.measured_value == "1024x576"


def test_wrong_codec_fails(tmp_path):
    final = render_at(tmp_path / "mpeg4.mp4", codec="mpeg4", pix_fmt="yuv420p")
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent)
    assert report.status == QCStatus.FAILED
    check = find_check(report, "codec")
    assert check.status == "FAIL"
    assert check.measured_value == "mpeg4"
    assert check.expected_value == "h264"


def test_invalid_duration_fails(tmp_path):
    final = render_at(tmp_path / "short.mp4", duration=3)
    settings = qc_settings(tmp_path, {"QC_MIN_DURATION": "5"})
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.FAILED
    check = find_check(report, "duration")
    assert check.status == "FAIL"
    assert find_check(report, "file_size").status == "PASS"  # size unrelated


def test_suspicious_file_size_fails(tmp_path):
    final = render_at(tmp_path / "tiny.mp4", duration=2)
    settings = qc_settings(tmp_path, {"QC_MIN_FILE_SIZE": "999999999"})
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.FAILED
    check = find_check(report, "file_size")
    assert check.status == "FAIL"
    assert find_check(report, "duration").status == "PASS"  # duration unrelated


def test_missing_audio_is_informational_warn(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    report = run_qc(agent)
    check = find_check(report, "audio")
    assert check.status == "WARN"
    assert check.severity == "info"
    # informational -> does not downgrade the overall verdict
    assert report.status == QCStatus.PASSED


def test_audio_present_passes(tmp_path):
    final = render_at(tmp_path / "with_audio.mp4", audio=True)
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent)
    check = find_check(report, "audio")
    assert check.status == "PASS"
    assert find_check(report, "corruption").status == "PASS"


# =========================================================================
# 9-13. Scene integrity + duration consistency
# =========================================================================


def test_scene_missing_fails_service(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    record.input_scene_ids = record.input_scene_ids[:1]
    record.input_clip_ids = record.input_clip_ids[:1]
    service = QCService()
    report = service.assemble_report(agent.project, record)
    assert report.status == QCStatus.FAILED
    assert find_check(report, "scenes_present").status == "FAIL"


def test_scene_order_mismatch_fails_service(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    record.input_scene_ids = list(reversed(record.input_scene_ids))
    record.input_clip_ids = list(reversed(record.input_clip_ids))
    service = QCService()
    report = service.assemble_report(agent.project, record)
    assert report.status == QCStatus.FAILED
    assert find_check(report, "scene_order").status == "FAIL"
    assert find_check(report, "clips_current").status == "FAIL"


def test_stale_assembly_refuses_to_run(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    from backend.models import ProjectStatus

    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    client = TestClient(api_module.app)
    client.post(f"/projects/{agent.project.id}/assembly/assemble")  # idempotent
    client.post(f"/projects/{agent.project.id}/qc/run")

    agent.invalidate_assembly("a scene clip was regenerated")
    # Regeneration moved us back to GENERATING_VIDEOS: the state guard refuses.
    assert client.post(f"/projects/{agent.project.id}/qc/run").status_code == 409

    # Even if the UI returns us to READY_FOR_REVIEW, QC must still refuse
    # because the assembly itself is stale (must-reassemble gate).
    agent.project.status = ProjectStatus.READY_FOR_REVIEW
    save_project(agent.project)
    resp = client.post(f"/projects/{agent.project.id}/qc/run")
    assert resp.status_code == 409
    assert "reassemble" in resp.json()["detail"].lower()
    status = client.get(f"/projects/{agent.project.id}/qc/status").json()
    assert status["stale"] is True
    assert status["can_run"] is False


def test_changed_video_generation_fails_service(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    new_clip = render_at(tmp_path / "new_scene0.mp4", duration=3)
    set_scene_clip(agent, 0, new_clip)  # new attempt id, all scenes still exist
    service = QCService()
    report = service.assemble_report(agent.project, record)
    assert report.status == QCStatus.FAILED
    check = find_check(report, "clips_current")
    assert check.status == "FAIL"
    assert "out of date" in check.message.lower()


def test_duration_mismatch_fails(tmp_path):
    clip = render_at(tmp_path / "clip3s.mp4", duration=3)
    final = render_at(tmp_path / "final9s.mp4", duration=9)
    settings = qc_settings(tmp_path, {"QC_DURATION_TOLERANCE": "0.5", "QC_DURATION_WARN_TOLERANCE": "1.0"})
    agent = fabricate_project(tmp_path, final, clip_path=clip)
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.FAILED
    check = find_check(report, "duration_consistency")
    assert check.status == "FAIL"


def test_duration_mismatch_warns(tmp_path):
    clip = render_at(tmp_path / "clip3s.mp4", duration=3)
    final = render_at(tmp_path / "final4s.mp4", duration=4)
    settings = qc_settings(tmp_path, {"QC_DURATION_TOLERANCE": "0.1", "QC_DURATION_WARN_TOLERANCE": "5.0"})
    agent = fabricate_project(tmp_path, final, clip_path=clip)
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.WARNINGS
    assert find_check(report, "duration_consistency").status == "WARN"


# =========================================================================
# 14-16. Overall verdicts: PASSED / WARNINGS / FAILED
# =========================================================================


def test_successful_qc_overall(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    report = run_qc(agent)
    assert report.status == QCStatus.PASSED
    assert report.summary == "All automated checks passed."
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    assert agent.project.qc_reports[-1].qc_id == report.qc_id


def test_warning_qc_overall(tmp_path):
    final = render_at(tmp_path / "frozen.mp4", source="color=c=0x808080", duration=4)
    # constant-color source -> no motion between frames -> blocking WARN
    settings = qc_settings(tmp_path, {"QC_MIN_FILE_SIZE": "100"})
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.WARNINGS
    freeze = find_check(report, "visual_freeze")
    assert freeze.status == "WARN"
    assert freeze.blocking is True
    assert report.summary.startswith("Automated checks passed with warnings")


def test_failed_qc_overall(tmp_path):
    final = render_at(tmp_path / "black.mp4", source="color=c=black", duration=4)
    settings = qc_settings(tmp_path, {"QC_MIN_FILE_SIZE": "100"})
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.FAILED
    assert find_check(report, "visual_black").status == "FAIL"
    assert report.summary.startswith("Automated QC failed")


def test_qc_never_auto_publishes(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    report = run_qc(agent)
    assert report.status == QCStatus.PASSED
    # QC ended at READY_FOR_REVIEW (human still reviews), never publishing.
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    with pytest.raises(InvalidTransitionError):
        agent.metadata_generated("T", "D", ["#s"])  # not in GENERATING_METADATA


# =========================================================================
# 17-20. Persistence, restart, invalidation, rerun
# =========================================================================


def test_qc_persistence(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    report = run_qc(agent)
    reloaded = load_project(agent.project.id)
    assert len(reloaded.qc_reports) == 1
    assert reloaded.qc_reports[0].qc_id == report.qc_id
    assert reloaded.qc_reports[0].status == QCStatus.PASSED
    assert reloaded.qc_reports[0].assembly_id == report.assembly_id
    assert reloaded.current_qc_id == report.qc_id
    assert reloaded.qc_result.passed is True  # legacy summary kept in sync


def test_restart_preserves_qc(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    reloaded = MasterAgent.load(agent.project.id)
    assert reloaded.project.status == ProjectStatus.READY_FOR_REVIEW
    report = reloaded.latest_qc_report()
    assert report is not None and report.status == QCStatus.PASSED
    assert reloaded.project.current_qc_id == report.qc_id
    assert AssemblyService().is_stale(reloaded.project) is False


def test_running_report_not_current_and_rerun_recovers(tmp_path):
    # A crash that left a RUNNING report behind must not count as current.
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    assembly = agent.successful_assembly()
    orphan = QCReport(
        project_id=agent.project.id,
        assembly_id=assembly.assembly_id,
        status=QCStatus.RUNNING,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    agent.qc_running(orphan)
    reloaded = MasterAgent.load(agent.project.id)
    assert reloaded.latest_qc_report().status == QCStatus.RUNNING
    assert reloaded.latest_qc_report().is_current is False

    report = run_qc(reloaded)
    assert report.status == QCStatus.PASSED
    assert report.qc_id != orphan.qc_id
    assert reloaded.project.current_qc_id == report.qc_id
    # Two reports now: the orphan (RUNNING) and the completed current one.
    assert [r.status for r in reloaded.project.qc_reports] == [
        QCStatus.RUNNING,
        QCStatus.PASSED,
    ]


def test_qc_invalidation_after_scene_regeneration(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    report = run_qc(agent)
    assert report.status == QCStatus.PASSED

    agent.invalidate_assembly("a scene clip was regenerated")
    assert report.status == QCStatus.INVALID
    assert agent.project.current_qc_id is None
    assert agent.project.status == ProjectStatus.GENERATING_VIDEOS

    # QC cannot report the old result as current.
    reloaded = MasterAgent.load(agent.project.id)
    assert reloaded.latest_qc_report().status == QCStatus.INVALID
    assert reloaded.project.current_qc_id is None


def test_qc_rerun_after_reassembly(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    record1, _, _ = assemble_project(agent)
    report1 = run_qc(agent)

    agent.invalidate_assembly("regen scene 1")
    set_scene_clip(agent, 0, render_at(tmp_path / "new0.mp4", duration=3))
    record2, _, _ = assemble_project(agent)
    assert record2.assembly_id != record1.assembly_id
    assert report1.status == QCStatus.INVALID

    report2 = run_qc(agent)
    assert report2.status == QCStatus.PASSED
    assert report2.assembly_id == record2.assembly_id
    assert report2.qc_id != report1.qc_id
    assert agent.project.current_qc_id == report2.qc_id
    # History preserved: old report kept but INVALID, new one current.
    assert [r.status for r in agent.project.qc_reports] == [
        QCStatus.INVALID,
        QCStatus.PASSED,
    ]


# =========================================================================
# 21-22. API surface + state transitions
# =========================================================================


def test_api_qc_run_status_result(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    client = TestClient(api_module.app)
    client.post(f"/projects/{agent.project.id}/assembly/assemble")
    agent = MasterAgent.load(agent.project.id)

    resp = client.post(f"/projects/{agent.project.id}/qc/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == "PASSED"
    assert body["status"] == "READY_FOR_REVIEW"
    report = body["report"]
    assert report["status"] == "PASSED"
    assert report["assembly_id"] == agent.project.current_assembly_id

    status = client.get(f"/projects/{agent.project.id}/qc/status").json()
    assert status["qc_current"] is True
    assert status["can_run"] is True
    assert status["stale"] is False
    assert status["overall"] == "PASSED"

    result = client.get(f"/projects/{agent.project.id}/qc/result").json()
    assert result["qc_current"] is True
    assert result["report"]["status"] == "PASSED"
    assert result["exists"] is True
    assert result["final_asset_url"].startswith("/assets/")
    assert client.get(result["final_asset_url"]).status_code == 200


def test_api_qc_endpoint_requires_review_state(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=1, videos_generated=False)
    client = TestClient(api_module.app)
    resp = client.post(f"/projects/{agent.project.id}/qc/run")
    assert resp.status_code == 409


def test_state_transitions_around_qc(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    agent.start_quality_check()
    assert agent.project.status == ProjectStatus.QUALITY_CHECK
    report = QCReport(
        project_id=agent.project.id,
        assembly_id=agent.project.current_assembly_id,
        status=QCStatus.FAILED,
        completed_at=datetime.now(timezone.utc).isoformat(),
        findings=["Scene 3 clip is not the current generation"],
        summary="Automated QC failed",
    )
    agent.qc_complete(report)
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    assert agent.project.qc_result.passed is False
    assert agent.project.qc_result.issues == ["Scene 3 clip is not the current generation"]

    # Manual metadata start is the only (future) way forward; not automatic.
    agent.metadata_start()
    assert agent.project.status == ProjectStatus.GENERATING_METADATA


def test_qc_failed_checks_reported_not_hidden(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    record, _, _ = assemble_project(agent)
    Path(record.output_path).write_text("garbage")
    report = run_qc(agent)
    assert report.status == QCStatus.FAILED
    assert any("could not be opened" in c.message for c in report.checks)
    assert report.findings, "findings must be populated"
    assert report.completed_at is not None


# =========================================================================
# Mock end-to-end: assembly -> QC -> persistence -> invalidate -> rerun
# =========================================================================


def test_mock_end_to_end_seed_workflow(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    client = TestClient(api_module.app)
    pid = client.post("/projects", json={"idea": "Seed growing from seed to flowering plant"}).json()["id"]

    client.post(f"/projects/{pid}/planning")
    client.post(f"/projects/{pid}/plan")
    client.post(f"/projects/{pid}/story/approve")
    client.post(f"/projects/{pid}/images/generate")

    project = client.get(f"/projects/{pid}").json()
    selections = {s["id"]: "A" for s in project["scenes"]}
    client.post(f"/projects/{pid}/images/select", json={"selections": selections})

    client.post(f"/projects/{pid}/videos/generate")
    project = client.get(f"/projects/{pid}").json()
    assert project["status"] == "ASSEMBLING"

    body = client.post(f"/projects/{pid}/assembly/assemble").json()
    assert body["status"] == "READY_FOR_REVIEW"

    # 1) final MP4 passes technical QC
    qc = client.post(f"/projects/{pid}/qc/run").json()
    assert qc["overall"] == "PASSED", qc["report"]["findings"]
    assert qc["status"] == "READY_FOR_REVIEW"

    # 2) QC record persists across restart
    project = client.get(f"/projects/{pid}").json()
    assert project["current_qc_id"] == qc["report"]["qc_id"]
    assert project["qc_reports"][-1]["status"] == "PASSED"

    # 3) regenerating one scene invalidates assembly + QC
    scene_id = project["scenes"][0]["id"]
    client.post(f"/projects/{pid}/videos/regenerate", json={"scene_id": scene_id})
    status = client.get(f"/projects/{pid}/qc/status").json()
    assert status["stale"] is True
    assert client.post(f"/projects/{pid}/qc/run").status_code == 409
    project = client.get(f"/projects/{pid}").json()
    assert project["qc_reports"][-1]["status"] == "INVALID"

    # 4) reassembly creates a current assembly; QC reruns against the new one
    body = client.post(f"/projects/{pid}/assembly/reassemble").json()
    assert body["status"] == "READY_FOR_REVIEW"
    qc2 = client.post(f"/projects/{pid}/qc/run").json()
    assert qc2["overall"] == "PASSED"
    assert qc2["report"]["assembly_id"] == body["assembly"]["assembly_id"]
    assert qc2["report"]["qc_id"] != qc["report"]["qc_id"]
    result = client.get(f"/projects/{pid}/qc/result").json()
    assert result["qc_current"] is True

    client.delete(f"/projects/{pid}")
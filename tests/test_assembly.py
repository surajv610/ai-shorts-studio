"""Tests for the FFmpeg video assembly layer.

Without assembly the video pipeline ends at per-scene clips; these tests cover
the full assembly service: validation, ordering, normalization, concatenation,
final output validation, records, reuse, staleness on regeneration, reassembly,
restart recovery, failure + retry, state transitions, and serving the final
video through /assets. Everything runs offline against real FFmpeg-generated
mock clips — no external AI or video API is ever called.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_assembly_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from agents.image import ImageAgent
from agents.image_schemas import ImagePromptOutput
from agents.master import MasterAgent, InvalidTransitionError
from agents.video import VideoAgent
from agents.video_schemas import VideoPromptOutput
from backend.assembly import AssemblyError, AssemblyService, probe_video
from backend.config import Settings
from backend.models import AssemblyStatus, ProjectStatus, VideoGeneration
from backend.providers.registry import get_image_provider
from backend.providers.video import MockVideoProvider
from backend.storage import load_project, project_asset_dir


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
        "locked down with zero movement, warm golden-hour side lighting, "
        "photorealistic 85mm macro look, nothing else changes"
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
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return VideoPromptOutput(**VIDEO_PROMPT).model_dump()


def make_settings(tmp_path):
    return Settings(
        env={"STORAGE_PATH": str(tmp_path), "VIDEO_POLL_INTERVAL": "0.001"}
    )


def make_assembly_project(tmp_path, n=3, videos_generated=True):
    """A project with locked images + real mock clips for every scene.

    With ``videos_generated=True`` the agent reaches ASSEMBLING (all scenes
    have a successful clip); otherwise it stays in GENERATING_VIDEOS.
    """
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


def scene_index(agent, index):
    return agent.project.scenes[index]


def clips_ordered(agent):
    return [s.video_clip for s in agent.project.scenes]


def svc():
    return AssemblyService()


def duration_of(path):
    return probe_video(str(path)).duration


def render_at(tmp_path, width, height, duration=4, fps=24, with_audio=False):
    """Render a small real clip (optionally with audio) via the mock renderer."""
    out = Path(tmp_path) / f"clip_{uuid4().hex[:8]}.mp4"
    ffmpeg = MockVideoProvider(make_settings(tmp_path))
    from backend.ffmpeg import get_ffmpeg_info

    cmd = [
        get_ffmpeg_info().path,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3.5"]
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-movflags", "+faststart", str(out)]
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:300])
    return out


def set_scene_clip(agent, index, path, duration_seconds=4):
    """Point a scene's current clip at a specific MP4 file (as a new attempt)."""
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
    from backend.storage import save_project

    save_project(agent.project)


def assemble_project(agent):
    """Run the shared assemble->transition helper against an agent."""
    r, reused, error = svc().assemble(agent.project)
    agent.begin_assembly()
    if error:
        agent.assembly_failed(error, record=r)
        return r, reused, error
    agent.assembly_complete(record=r)
    return r, reused, error


# =========================================================================
# 1-5. Validation + preconditions
# =========================================================================


def test_all_scenes_ready_assemble_succeeds(tmp_path):
    agent = make_assembly_project(tmp_path, n=3)
    record, reused, error = svc().assemble(agent.project)
    assert error is None
    assert record.status == AssemblyStatus.SUCCEEDED
    assert reused is False
    assert record.scene_count == 3
    assert Path(record.output_path).is_file()


def test_plan_requires_locked_image(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    scene = scene_index(agent, 0)
    scene.selected_image = None
    scene.selected_image_id = None
    with pytest.raises(AssemblyError, match="Scene 1 has no locked"):
        svc().plan(agent.project)


def test_plan_missing_scene_clip_identifies_scene(tmp_path):
    agent = make_assembly_project(tmp_path, n=2, videos_generated=False)
    scene = scene_index(agent, 1)
    scene.video_generations = []
    scene.current_video_id = None
    scene.video_clip = None
    with pytest.raises(AssemblyError, match="Scene 2 has no successful video clip"):
        svc().plan(agent.project)


def test_plan_missing_video_file_identifies_scene(tmp_path):
    agent = make_assembly_project(tmp_path, n=2, videos_generated=False)
    scene = scene_index(agent, 0)
    gen = scene.video_generations[-1]
    gen.output_path = str(Path(tempfile.mkdtemp()) / "gone.mp4")
    gen.status = "SUCCEEDED"
    scene.video_clip = gen.output_path
    with pytest.raises(AssemblyError, match="Scene 1 video file is missing or unreadable"):
        svc().plan(agent.project)


def test_assemble_rejects_invalid_video(tmp_path):
    agent = make_assembly_project(tmp_path, n=1, videos_generated=False)
    junk = Path(tmp_path) / "not_a_video.mp4"
    junk.write_text("this is not a real mp4 file")
    set_scene_clip(agent, 0, junk)
    record, reused, error = svc().assemble(agent.project)
    assert record.status == AssemblyStatus.FAILED
    assert error and "could not be read" in error


def test_assemble_rejects_non_portrait_clip(tmp_path):
    agent = make_assembly_project(tmp_path, n=1, videos_generated=False)
    landscape = render_at(tmp_path, 640, 360)
    set_scene_clip(agent, 0, landscape)
    record, _, error = svc().assemble(agent.project)
    assert record.status == AssemblyStatus.FAILED
    assert error and "not vertical 9:16" in error


# =========================================================================
# 6-9. FFmpeg concatenation + normalization + output validation
# =========================================================================


def test_ffmpeg_concat_produces_valid_final(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, error = svc().assemble(agent.project)
    assert error is None
    final = Path(record.output_path)
    assert final.is_file() and final.stat().st_size > 0
    probe = probe_video(record.output_path)
    assert probe.video_codec == "h264"
    assert probe.duration > 0
    expected = sum(duration_of(p) for p in clips_ordered(agent))
    assert abs(record.output_duration - expected) < 1.0


def test_scene_order_is_storyboard_order(tmp_path):
    agent = make_assembly_project(tmp_path, n=3)
    # Rename each clip to a shuffled filename whose ALPHABETICAL order differs
    # from the storyboard order. The assembly must follow the storyboard.
    shuffled = ["m5.mp4", "a1.mp4", "z9.mp4"]  # alphabetical: a1, m5, z9
    clips_dir = Path(agent.project.scenes[0].video_generations[-1].output_path).parent
    for scene, name in zip(agent.project.scenes, shuffled):
        gen = scene.video_generations[-1]
        target = clips_dir / name
        target.write_bytes(Path(gen.output_path).read_bytes())
        gen.output_path = str(target)
        scene.video_clip = gen.output_path
    from backend.storage import save_project

    save_project(agent.project)

    record, _, error = svc().assemble(agent.project)
    assert error is None
    assert record.input_clip_ids == [s.current_video_id for s in agent.project.scenes]
    assert record.input_scene_ids == [s.id for s in agent.project.scenes]
    assert [Path(p).name for p in record.input_clip_paths] == shuffled
    # Sanity: storyboard order differs from alphabetical filename order.
    assert [Path(p).name for p in record.input_clip_paths] != sorted(shuffled)


def test_differing_formats_are_normalized(tmp_path):
    agent = make_assembly_project(tmp_path, n=2, videos_generated=False)
    # Scene 2 clip is higher resolution than scene 1's 9:16 clips.
    big = render_at(tmp_path, 640, 1136)
    set_scene_clip(agent, 1, big)
    agent.videos_generated()
    record, _, error = svc().assemble(agent.project)
    assert error is None
    assert record.raw["normalized"] is True
    assert record.output_width == 576 or record.output_width
    first_w = probe_video(agent.project.scenes[0].video_generations[-1].output_path).width
    assert record.output_width == first_w
    assert record.output_height == first_w * 16 // 9
    final_probe = probe_video(record.output_path)
    assert (final_probe.width, final_probe.height) == (record.output_width, record.output_height)


def test_compatible_clips_are_not_reencoded(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, error = svc().assemble(agent.project)
    assert error is None
    assert record.raw["normalized"] is False
    assert record.raw["normalized_clip_count"] == 0


def test_final_output_is_vertical_h264(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, error = svc().assemble(agent.project)
    assert error is None
    probe = probe_video(record.output_path)
    assert probe.video_codec == "h264"
    assert probe.height > probe.width
    ratio = probe.width / probe.height
    assert abs(ratio - 9 / 16) < 0.02


# =========================================================================
# 10-16. Records, reuse, staleness, reassembly, restart
# =========================================================================


def test_assembly_record_persists(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    assert record.status == AssemblyStatus.SUCCEEDED
    assert record.input_scene_ids == [s.id for s in agent.project.scenes]
    assert len(record.input_clip_ids) == 2
    reloaded = load_project(agent.project.id)
    assert len(reloaded.assemblies) == 1
    assert reloaded.assemblies[0].status == AssemblyStatus.SUCCEEDED
    assert reloaded.final_video == record.output_path


def test_existing_assembly_is_reused(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, reused, _ = assemble_project(agent)
    assert reused is False
    record2, reused2, _ = assemble_project(agent)
    assert reused2 is True
    assert record2.assembly_id == record.assembly_id
    assert len(agent.project.assemblies) == 1  # no duplicate record


def test_stale_detection_after_regeneration(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    assert svc().is_stale(agent.project) is False

    agent.invalidate_assembly()
    scene = scene_index(agent, 0)
    set_scene_clip(agent, 0, render_at(tmp_path, 576, 1024, 3))
    assert scene.current_video_id != record.input_clip_ids[0]
    assert svc().is_stale(agent.project) is True


def test_regeneration_invalidates_old_assembly(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    agent.invalidate_assembly("a scene clip was regenerated")
    assert record.status == AssemblyStatus.STALE
    assert "regenerated" in record.stale_reason
    # The final file still exists on disk for preview but is flagged stale.
    assert Path(record.output_path).is_file()


def test_reassembly_uses_updated_clip_only(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    old_ids = list(record.input_clip_ids)
    agent.invalidate_assembly()
    new_scene_0 = render_at(tmp_path, 576, 1024, 3)
    set_scene_clip(agent, 0, new_scene_0)
    scene_1_id = scene_index(agent, 1).current_video_id

    record2, reused2, _ = assemble_project(agent)
    assert reused2 is False
    assert record2.input_clip_ids[0] != old_ids[0]
    assert record2.input_clip_ids[1] == scene_1_id  # scene 2 clip unchanged
    assert [a.status for a in agent.project.assemblies] == [
        AssemblyStatus.STALE,
        AssemblyStatus.SUCCEEDED,
    ]
    # New final duration reflects only the swapped scene: 3s instead of 4s.
    expected = 3 + duration_of(scene_index(agent, 1).video_clip)
    assert abs(record2.output_duration - expected) < 1.0
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW


def test_restart_preserves_assembly(tmp_path):
    agent = make_assembly_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    reloaded = MasterAgent.load(agent.project.id)
    assert reloaded.project.status == ProjectStatus.READY_FOR_REVIEW
    assert reloaded.project.final_video == record.output_path
    assert len(reloaded.project.assemblies) == 1
    assert Path(reloaded.project.final_video).is_file()
    assert svc().is_stale(reloaded.project) is False


# =========================================================================
# 17-18. Failure + retry + state transitions
# =========================================================================


def test_assembly_failure_marks_project(tmp_path):
    agent = make_assembly_project(tmp_path, n=1, videos_generated=False)
    junk = Path(tmp_path) / "broken.mp4"
    junk.write_text("junk not a video")
    set_scene_clip(agent, 0, junk)
    agent.videos_generated()
    # Build failures return a FAILED record rather than raising.
    record, reused, error = svc().assemble(agent.project)
    assert record.status == AssemblyStatus.FAILED
    assert reused is False and error
    r, reused, error = assemble_project(agent)
    assert agent.project.status == ProjectStatus.ASSEMBLY_FAILED
    assert agent.project.errors


def test_retry_allowed_after_assembly_failure(tmp_path):
    agent = make_assembly_project(tmp_path, n=1, videos_generated=False)
    junk = Path(tmp_path) / "broken.mp4"
    junk.write_text("junk not a video")
    set_scene_clip(agent, 0, junk)
    agent.videos_generated()
    record, reused, error = svc().assemble(agent.project)
    assemble_project(agent)
    assert agent.project.status == ProjectStatus.ASSEMBLY_FAILED

    # Fix the clip (same generation path) and retry without clip regen.
    good = render_at(tmp_path, 576, 1024, 3)
    Path(scene_index(agent, 0).video_clip).write_bytes(good.read_bytes())
    agent.begin_assembly()  # ASSEMBLY_FAILED -> ASSEMBLING
    assert agent.project.status == ProjectStatus.ASSEMBLING
    record2, reused2, error = svc().assemble(agent.project)
    assemble_project(agent)
    assert error is None
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    assert record2.status == AssemblyStatus.SUCCEEDED
    # History preserved: failed first, then succeeded.
    assert [a.status for a in agent.project.assemblies] == [
        AssemblyStatus.FAILED,
        AssemblyStatus.SUCCEEDED,
    ]


def test_master_state_transitions_round_trip(tmp_path):
    agent = make_assembly_project(tmp_path, n=1)
    assert agent.project.status == ProjectStatus.ASSEMBLING
    agent.assembly_failed("boom")
    assert agent.project.status == ProjectStatus.ASSEMBLY_FAILED
    agent.begin_assembly()
    assert agent.project.status == ProjectStatus.ASSEMBLING
    agent.assembly_complete("final.mp4")
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    # Regen path: stale + back to generating, then re-assembling.
    agent.invalidate_assembly()
    assert agent.project.status == ProjectStatus.GENERATING_VIDEOS
    agent.videos_generated()
    assert agent.project.status == ProjectStatus.ASSEMBLING
    # QC (future phase) still reachable from READY_FOR_REVIEW.
    agent.assembly_complete("final.mp4")
    agent.start_quality_check()
    assert agent.project.status == ProjectStatus.QUALITY_CHECK


def test_unexpected_state_blocks_assembly(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_assembly_project(tmp_path, n=1, videos_generated=False)
    from backend.storage import save_project

    agent.project.status = ProjectStatus.PLANNING
    save_project(agent.project)
    client = TestClient(api_module.app)
    resp = client.post(f"/projects/{agent.project.id}/assembly/assemble")
    assert resp.status_code == 409


# =========================================================================
# 19-20. Mock workflow + /assets serving + API surface
# =========================================================================


def test_mock_workflow_full_pipeline(tmp_path):
    agent = make_assembly_project(tmp_path, n=3)
    record, reused, error = svc().assemble(agent.project)
    assert error is None and record.status == AssemblyStatus.SUCCEEDED
    assert record.scene_count == 3
    final = probe_video(record.output_path)
    assert final.height > final.width  # vertical
    assert abs(final.duration - sum(duration_of(p) for p in clips_ordered(agent))) < 1.0


def test_final_video_served_through_assets(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_assembly_project(tmp_path, n=2)
    assemble_project(agent)
    client = TestClient(api_module.app)
    result = client.get(f"/projects/{agent.project.id}/assembly/result").json()
    assert result["exists"] is True
    url = result["final_asset_url"]
    assert url.startswith("/assets/")
    resp = client.get(url)
    assert resp.status_code == 200
    assert len(resp.content) > 0


def test_api_assemble_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_assembly_project(tmp_path, n=2)
    client = TestClient(api_module.app)
    resp = client.post(f"/projects/{agent.project.id}/assembly/assemble")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "READY_FOR_REVIEW"
    assert body["reused"] is False
    assert body["assembly"]["status"] == "SUCCEEDED"


def test_api_reassemble_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_assembly_project(tmp_path, n=2)
    client = TestClient(api_module.app)
    client.post(f"/projects/{agent.project.id}/assembly/assemble")
    status = client.get(f"/projects/{agent.project.id}/assembly/status").json()
    assert status["stale"] is False

    # Regenerate scene 1 via the API: invalidates + returns to GENERATING_VIDEOS.
    scene_id = agent.project.scenes[0].id
    from backend.providers.video import MockVideoProvider

    agent2 = MasterAgent.load(agent.project.id)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=MockVideoProvider(make_settings(tmp_path))
    )
    import backend.api as api_module

    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    resp = client.post(f"/projects/{agent.project.id}/videos/regenerate", json={"scene_id": scene_id})
    assert resp.status_code == 200
    status = client.get(f"/projects/{agent.project.id}/assembly/status").json()
    assert status["stale"] is True

    resp = client.post(f"/projects/{agent.project.id}/assembly/reassemble")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "READY_FOR_REVIEW"
    assert body["reused"] is False
    result = client.get(f"/projects/{agent.project.id}/assembly/result").json()
    assert result["assembly"]["status"] == "SUCCEEDED"
    assert result["exists"] is True


def test_api_assembly_status_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_assembly_project(tmp_path, n=2)
    client = TestClient(api_module.app)
    status = client.get(f"/projects/{agent.project.id}/assembly/status").json()
    assert status["status"] == "ASSEMBLING"
    assert status["readiness"]["ready"] is True
    assert status["readiness"]["scene_count"] == 2
    assert status["can_assemble"] is True
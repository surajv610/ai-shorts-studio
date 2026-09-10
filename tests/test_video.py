"""Tests for the Video Agent.

Every test uses injected fakes (FakeLLM for animation prompts and real
MockVideoProvider / simple fakes for the provider) so NO real/paid Veo API is
ever called. Storage is redirected to a temp dir for the whole module.
"""

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_video_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from agents.image import ImageAgent
from agents.image_schemas import ImagePromptOutput
from agents.master import MasterAgent, InvalidTransitionError
from agents.video import VideoAgent
from agents.video_schemas import VideoPromptOutput, VIDEO_PROMPT_SCHEMA
from backend.config import Settings
from backend.models import ProjectStatus
from backend.providers.base import (
    AsyncJob,
    AsyncJobStatus,
    ProviderGenerationError,
)
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
def fast_video_poll(monkeypatch):
    """Any VideoAgent without explicit settings uses fast mock polling."""
    monkeypatch.setenv("VIDEO_POLL_INTERVAL", "0.001")
    monkeypatch.setenv("VIDEO_POLL_TIMEOUT", "5")


VIDEO_PROMPT = dict(
    global_continuity=(
        "same single plant, same dark soil, same locked-down macro camera, "
        "same warm golden-hour lighting, photorealistic 85mm macro, no "
        "anything else in the frame"
    ),
    scene_action="the shoot pushes upward as the bud swells open",
    motion="very slow, continuous upward growth only",
    environment="identical dark soil, nothing else changes",
    camera="completely locked-down, zero camera movement",
    lighting="same warm golden-hour side lighting throughout",
    timelapse="4-second growth timelapse, chronological",
    negative_constraints="no people, no text, no flicker, no camera shake",
    prompt=(
        "Macro 4-second 9:16 timelapse of a single plant shoot pushing "
        "upward as its bud swells open on the same dark moist soil, camera "
        "locked down with zero movement, warm golden-hour side lighting, "
        "photorealistic 85mm macro look, rich earthy tones, nothing else "
        "changes, no people, no text, no flicker"
    ),
)


def make_prompt(**overrides):
    data = dict(VIDEO_PROMPT)
    data.update(overrides)
    return VideoPromptOutput(**data)


class FakeLLM:
    """Returns a valid video-prompt structured dict; records every call."""

    def __init__(self, output=None, error=None):
        self.output = output or make_prompt()
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.output.model_dump()


class FakeImageLLM:
    """Same interface as the image agent's LLM, for building the project."""

    def __call__(self, **kwargs):
        return ImagePromptOutput(
            global_continuity="plant, soil, camera, lighting",
            scene_action="seed cracking",
            prompt_a="Macro close-up of a seed cracking open on dark moist soil",
            prompt_b="Bird's-eye view of the seed on the same soil",
            composition_notes="alternate framing",
        ).model_dump()


class CountingVideoProvider:
    """Wraps a real MockVideoProvider and counts submit_async calls."""

    def __init__(self, settings, **kwargs):
        self.inner = MockVideoProvider(settings, **kwargs)
        self.name = self.inner.name
        self.model = self.inner.model
        self.submits = []

    def capabilities(self):
        return self.inner.capabilities()

    def supports(self, capability):
        return self.inner.supports(capability)

    def submit_async(self, image_reference, prompt, options=None):
        self.submits.append({"prompt": prompt, "image": image_reference})
        return self.inner.submit_async(image_reference, prompt, options=options)

    def poll_async(self, job, options=None):
        return self.inner.poll_async(job, options=options)

    def download_result(self, job, options=None):
        return self.inner.download_result(job, options=options)


class FailSubmitProvider:
    """submit_async always raises — exercises per-scene isolation."""

    name = "fail"
    model = "fail-model"

    def capabilities(self):
        return {"image_to_video": True}

    def supports(self, capability):
        return capability == "image_to_video"

    def submit_async(self, image_reference, prompt, options=None):
        raise ProviderGenerationError("quota exhausted for video generation")

    def poll_async(self, job, options=None):
        return job

    def download_result(self, job, options=None):
        raise ProviderGenerationError("no result")


class StuckProvider:
    """poll_async never reaches a terminal state — forces a timeout."""

    name = "stuck"
    model = "stuck-model"

    def capabilities(self):
        return {"image_to_video": True}

    def supports(self, capability):
        return capability == "image_to_video"

    def submit_async(self, image_reference, prompt, options=None):
        return AsyncJob(
            provider=self.name,
            job_id="stuck-job-1",
            status=AsyncJobStatus.PROCESSING,
        )

    def poll_async(self, job, options=None):
        job.status = AsyncJobStatus.PROCESSING
        return job

    def download_result(self, job, options=None):
        raise ProviderGenerationError("no result")


def make_settings(tmp_path, **env):
    base = {"STORAGE_PATH": str(tmp_path), "VIDEO_POLL_INTERVAL": "0.001"}
    base.update(env)
    return Settings(env=base)


def make_video_project(tmp_path, n=3):
    """A project already in GENERATING_VIDEOS with per-scene locked images."""
    agent = MasterAgent.create("Seed growing from seed to flowering plant")
    agent.start_planning()
    storyboard = {
        "title": "Seed to Bloom",
        "summary": "A seed grows to a flower.",
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
    }
    scenes = [
        {"description": f"Scene {i} visual", "duration_seconds": 4}
        for i in range(1, n + 1)
    ]
    agent.submit_storyboard(
        storyboard=storyboard,
        scenes=scenes,
        project_bible={
            "subject": "seed",
            "environment": "dark soil",
            "time_lighting": "evening golden hour",
            "camera": "locked-down macro",
            "lens_look": "85mm macro shallow DOF",
            "visual_style": "Photorealistic",
            "negative_constraints": "no people, no text overlays",
            "aspect_ratio": "9:16",
            "target_duration": 30.0,
        },
    )
    agent.approve_story()
    agent.images_generated()

    from backend.providers.registry import get_image_provider

    img = ImageAgent(
        llm=FakeImageLLM(),
        image_provider=get_image_provider(make_settings(tmp_path, IMAGE_PROVIDER="mock")),
    )
    agent = MasterAgent.load(agent.project.id)
    img.run(agent.project)
    agent.select_images({s.id: "A" for s in agent.project.scenes})
    return agent


def scene_of(project_or_agent, index=0):
    project = getattr(project_or_agent, "project", project_or_agent)
    return project.scenes[index]


# =========================================================================
# 1-3. Prompt generation
# =========================================================================


def test_video_prompt_generation_uses_full_context(tmp_path):
    agent = make_video_project(tmp_path)
    llm = FakeLLM()
    vid = VideoAgent(
        llm=llm, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)

    assert len(llm.calls) == 3  # one per scene during the initial run
    for call in llm.calls:
        msg = call["user_message"]
        assert "Seed growing from seed to flowering plant" in msg
        assert "PROJECT BIBLE" in msg
        assert "9:16" in msg
        assert "dark soil" in msg
        assert call["json_schema"] is VIDEO_PROMPT_SCHEMA


def test_video_prompt_includes_previous_and_next_scene(tmp_path):
    agent = make_video_project(tmp_path)
    llm = FakeLLM()
    vid = VideoAgent(
        llm=llm, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)

    scene2_call = llm.calls[1]["user_message"]
    assert "action 1" in scene2_call and "PREVIOUS SCENE" in scene2_call
    assert "action 3" in scene2_call and "NEXT SCENE" in scene2_call


def test_video_prompt_generation_is_schema_validated(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    bad = FakeLLM(output=make_prompt(prompt="too short"))
    vid = VideoAgent(
        llm=bad, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    summary = vid.run(agent.project)
    assert summary["scenes_failed"] == 1
    assert summary["scenes_ready"] == 0
    gen = scene_of(agent.project).video_generations[-1]
    assert gen.status == "FAILED"


# =========================================================================
# 4-9. Generation flow
# =========================================================================


def test_scene_generation_stores_clip_under_project_videos(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    provider = CountingVideoProvider(make_settings(tmp_path))
    vid = VideoAgent(llm=FakeLLM(), video_provider=provider)
    vid.run(agent.project)

    scene = scene_of(agent.project)
    gen = scene.video_generations[-1]
    assert gen.status == "SUCCEEDED"
    assert gen.provider == "mock"
    assert gen.job_id
    assert gen.output_path
    assert gen.output_path.startswith(str(project_asset_dir(agent.project.id, "videos")))
    assert Path(gen.output_path).is_file()
    assert Path(gen.output_path).suffix.lower() == ".mp4"
    assert scene.video_clip == gen.output_path
    assert scene.current_video_id == gen.attempt_id


def test_async_polling_simulates_progress(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    provider = CountingVideoProvider(make_settings(tmp_path))
    job = provider.submit_async("img.png", "grow", options={"duration_seconds": 1})
    assert job.status == AsyncJobStatus.PENDING
    job = provider.poll_async(job)
    assert job.status == AsyncJobStatus.PROCESSING
    job = provider.poll_async(job)
    assert job.status == AsyncJobStatus.SUCCEEDED
    assert job.is_terminal
    assert Path(provider.download_result(job).video).is_file()


def test_run_returns_summary_of_readiness(tmp_path):
    agent = make_video_project(tmp_path, n=3)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    summary = vid.run(agent.project)
    assert summary == {
        "scenes_ready": 3,
        "scenes_generated": 3,
        "scenes_failed": 0,
        "scenes_pending": 0,
        "scenes_needs_selection": 0,
    }


def test_failed_generation_isolated_per_scene(tmp_path):
    agent = make_video_project(tmp_path, n=3)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=FailSubmitProvider()
    )
    summary = vid.run(agent.project)
    assert summary["scenes_failed"] == 3
    assert summary["scenes_ready"] == 0
    for scene in agent.project.scenes:
        gen = scene.video_generations[-1]
        assert gen.status == "FAILED"
        assert "quota exhausted" in (gen.error or "")


def test_timeout_marks_scene_failed(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    vid = VideoAgent(
        llm=FakeLLM(),
        video_provider=StuckProvider(),
        settings=make_settings(
            tmp_path, VIDEO_POLL_TIMEOUT="0.05", VIDEO_POLL_INTERVAL="0.01"
        ),
    )
    vid.run(agent.project)
    gen = scene_of(agent.project).video_generations[-1]
    assert gen.status == "FAILED"
    assert "timed out" in (gen.error or "").lower()


def test_provider_submission_error_sanitized_and_recorded(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    vid = VideoAgent(llm=FakeLLM(), video_provider=FailSubmitProvider())
    vid.run(agent.project)
    gen = scene_of(agent.project).video_generations[-1]
    assert gen.status == "FAILED"
    assert (gen.error or "").endswith("generation failure") is False  # not raw
    assert "quota" in (gen.error or "")


# =========================================================================
# 10-12. Retry + regeneration
# =========================================================================


def test_retry_failed_reuses_prompt(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    llm = FakeLLM()
    vid = VideoAgent(
        llm=llm,
        video_provider=FailSubmitProvider(),
        settings=make_settings(tmp_path),
    )
    vid.run(agent.project)

    scene = scene_of(agent.project)
    failed_gen = scene.video_generations[-1]
    llm.calls.clear()

    ok_provider = CountingVideoProvider(make_settings(tmp_path))
    vid2 = VideoAgent(llm=llm, video_provider=ok_provider, settings=make_settings(tmp_path))
    vid2.retry_failed(agent.project, scene.id)

    assert len(llm.calls) == 0  # reused the failed prompt, no new LLM call
    latest = scene_of(agent.project).video_generations[-1]
    assert latest.status == "SUCCEEDED"
    assert latest.attempt_id != failed_gen.attempt_id
    assert latest.prompt == failed_gen.prompt


def test_regenerate_clip_reuses_prompt_without_llm(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    llm = FakeLLM()
    vid = VideoAgent(
        llm=llm, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)
    scene = scene_of(agent.project)
    original = scene.video_generations[-1]
    llm.calls.clear()

    vid2 = VideoAgent(
        llm=llm, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid2.regenerate_clip(agent.project, scene.id)

    assert len(llm.calls) == 0
    gens = scene_of(agent.project).video_generations
    assert len(gens) == 2
    latest = gens[-1]
    assert latest.status == "SUCCEEDED"
    assert latest.prompt == original.prompt
    # History preserved: the previous successful clip still exists on disk.
    assert Path(original.output_path).is_file()
    # Current pointer moved to the new attempt.
    assert scene_of(agent.project).current_video_id == latest.attempt_id


def test_regenerate_prompt_creates_new_prompt(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    llm = FakeLLM()
    vid = VideoAgent(
        llm=llm, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)
    original = scene_of(agent.project).video_prompt
    llm.calls.clear()
    llm.output = make_prompt(
        prompt=(
            "Macro 4-second 9:16 timelapse of the same plant as its flower "
            "fully opens for the first time, camera locked down, warm light"
        )
    )

    vid2 = VideoAgent(
        llm=llm, video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid2.regenerate_prompt_clip(agent.project, scene_of(agent.project).id)

    assert len(llm.calls) == 1
    scene = scene_of(agent.project)
    assert scene.video_prompt != original
    assert scene.video_prompt == llm.output.prompt
    assert scene.video_generations[-1].prompt == llm.output.prompt


# =========================================================================
# 13-15. Persistence + restart recovery
# =========================================================================


def test_generation_history_persisted(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)
    agent.project = load_project(agent.project.id)

    scene = scene_of(agent.project)
    assert len(scene.video_generations) == 1
    gen = scene.video_generations[0]
    assert gen.status == "SUCCEEDED"
    assert scene.video_prompt
    assert scene.video_clip == gen.output_path
    assert scene.current_video_id == gen.attempt_id


def test_restart_resumes_only_incomplete_scenes(tmp_path):
    agent = make_video_project(tmp_path, n=3)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)

    snapshot = load_project(agent.project.id)
    llm = FakeLLM()
    provider = CountingVideoProvider(make_settings(tmp_path))
    vid2 = VideoAgent(llm=llm, video_provider=provider, settings=make_settings(tmp_path))
    summary = vid2.run(snapshot)

    assert summary["scenes_ready"] == 3
    assert summary["scenes_generated"] == 0
    assert len(llm.calls) == 0
    assert provider.submits == []


def test_completed_scenes_never_regenerated(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    provider = CountingVideoProvider(make_settings(tmp_path))
    VideoAgent(
        llm=FakeLLM(), video_provider=provider, settings=make_settings(tmp_path)
    ).run(agent.project)
    first_count = len(provider.submits)

    VideoAgent(
        llm=FakeLLM(), video_provider=provider, settings=make_settings(tmp_path)
    ).run(agent.project)

    assert len(provider.submits) == first_count == 2


# =========================================================================
# 16-17. Ordering + run summary
# =========================================================================


def test_scenes_generated_in_storyboard_order(tmp_path):
    agent = make_video_project(tmp_path, n=3)
    provider = CountingVideoProvider(make_settings(tmp_path))
    VideoAgent(
        llm=FakeLLM(), video_provider=provider, settings=make_settings(tmp_path)
    ).run(agent.project)

    scene_paths = [s.selected_image for s in agent.project.scenes]
    submitted = [Path(s["image"]) for s in provider.submits]
    assert submitted == [Path(p).resolve() for p in scene_paths]


# =========================================================================
# 18. State transitions
# =========================================================================


def test_transition_to_assembling_only_when_all_scenes_done(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    vid.run(agent.project)
    assert agent.project.status == ProjectStatus.GENERATING_VIDEOS  # not auto-moved


def test_videos_generated_requires_all_scenes(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    agent.project.scenes[0].video_clip = "/tmp/clip1.mp4"
    with pytest.raises(InvalidTransitionError):
        agent.videos_generated()
    assert agent.project.status == ProjectStatus.GENERATING_VIDEOS


def test_videos_generated_transitions_when_all_clips_present(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    for scene in agent.project.scenes:
        scene.video_clip = f"/tmp/clip-{scene.id}.mp4"
    agent.videos_generated()
    assert agent.project.status == ProjectStatus.ASSEMBLING


# =========================================================================
# 19-20. Provider errors + mock provider behaviour
# =========================================================================


def test_mock_provider_produces_executable_mp4(tmp_path):
    settings = make_settings(tmp_path)
    provider = MockVideoProvider(settings)
    job = provider.submit_async("img.png", "grow", options={"duration_seconds": 1})
    job = provider.poll_async(job)
    job = provider.poll_async(job)
    result = provider.download_result(job)
    assert result.status == "SUCCEEDED"
    assert Path(result.video).is_file()
    data = Path(result.video).read_bytes()
    assert data  # non-empty file


def test_mock_provider_failure_injection(tmp_path):
    settings = make_settings(tmp_path)
    provider = MockVideoProvider(settings, fail_after=1)
    job = provider.submit_async("img.png", "grow")
    job = provider.poll_async(job)
    assert job.status == AsyncJobStatus.FAILED
    assert job.error
    with pytest.raises(ProviderGenerationError):
        provider.download_result(job)


def test_scene_without_locked_image_needs_selection(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    agent.project.scenes[0].selected_image = None
    agent.project.scenes[0].selected_image_id = None
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    summary = vid.run(agent.project)
    assert summary["scenes_needs_selection"] == 1
    assert scene_of(agent.project, 0).video_generations == []


# =========================================================================
# 21. API surface
# =========================================================================


def test_api_generate_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    client = TestClient(api_module.app)

    resp = client.post(f"/projects/{agent.project.id}/videos/generate")
    assert resp.status_code == 200
    body = resp.json()
    scenes = body["scenes"]
    assert all(
        s["video_generations"][-1]["status"] == "SUCCEEDED" for s in scenes
    )
    assert all(s["video_clip"] for s in scenes)


def test_api_generate_scene_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    client = TestClient(api_module.app)

    scene_id = agent.project.scenes[0].id
    resp = client.post(
        f"/projects/{agent.project.id}/videos/generate-scene",
        json={"scene_id": scene_id},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "SUCCEEDED"


def test_api_regenerate_and_retry_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    scene_id = agent.project.scenes[0].id
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    client = TestClient(api_module.app)

    # Generate just this scene so it has a prompt to reuse (project stays
    # GENERATING_VIDEOS because scene B remains incomplete).
    r0 = client.post(
        f"/projects/{agent.project.id}/videos/generate-scene",
        json={"scene_id": scene_id},
    )
    assert r0.status_code == 200
    assert r0.json()["status"] == "SUCCEEDED"

    r1 = client.post(
        f"/projects/{agent.project.id}/videos/regenerate",
        json={"scene_id": scene_id},
    )
    assert r1.status_code == 200
    assert isinstance(r1.json()["status"], str)

    r2 = client.post(
        f"/projects/{agent.project.id}/videos/regenerate-prompt",
        json={"scene_id": scene_id},
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "SUCCEEDED"

    r3 = client.post(
        f"/projects/{agent.project.id}/videos/retry",
        json={"scene_id": scene_id},
    )
    assert r3.status_code == 200
    assert r3.json()["status"] == "SUCCEEDED"


def test_api_video_status_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=1)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    client = TestClient(api_module.app)

    scene_id = agent.project.scenes[0].id
    resp = client.get(f"/projects/{agent.project.id}/videos/{scene_id}/status")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("NONE", "SUCCEEDED")


def test_api_video_endpoints_require_generating_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=1)
    agent.videos_generated({s.id: "/tmp/x.mp4" for s in agent.project.scenes})
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    client = TestClient(api_module.app)

    scene_id = agent.project.scenes[0].id
    resp = client.post(f"/projects/{agent.project.id}/videos/generate")
    assert resp.status_code == 409
    resp = client.post(
        f"/projects/{agent.project.id}/videos/generate-scene",
        json={"scene_id": scene_id},
    )
    assert resp.status_code == 409


def test_api_generate_transitions_when_all_scenes_done(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    vid = VideoAgent(
        llm=FakeLLM(), video_provider=CountingVideoProvider(make_settings(tmp_path))
    )
    monkeypatch.setattr(api_module, "get_video_agent", lambda: vid)
    client = TestClient(api_module.app)

    resp = client.post(f"/projects/{agent.project.id}/videos/generate")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ASSEMBLING"
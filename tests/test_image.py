"""Tests for the Image Agent.

Every test uses injected fakes (FakeLLM for prompt generation and the real
MockImageProvider for candidates) so NO real/paid image API is ever called.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_image_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from agents.image import ImageAgent
from agents.image_schemas import ImagePromptOutput
from agents.llm import ProviderError
from agents.master import MasterAgent
from backend.config import Settings
from backend.models import ProjectStatus
from backend.providers.base import GenerationOptions, ProviderGenerationError
from backend.providers.image import MockImageProvider
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


def make_prompt(**overrides):
    data = dict(
        global_continuity="same plant species, same dark soil, same locked-down macro camera, same warm golden-hour lighting, photorealistic 85mm macro look, no people, no text",
        scene_action="the seed has just cracked open and a root tip is emerging",
        prompt_a="Macro close-up of a seed cracking open on dark moist soil, a single thin root tip emerging downward, 9:16 vertical composition, locked-down camera, warm golden-hour side lighting, photorealistic 85mm macro look, rich earthy tones, no people, no text",
        prompt_b="Bird's-eye view of the same seed on the same soil with the root tip emerging, same lighting and style, 9:16 vertical composition, alternate top-down framing, no people, no text",
        composition_notes="B uses a top-down framing instead of the close-up side angle",
    )
    data.update(overrides)
    return ImagePromptOutput(**data)


class FakeLLM:
    """Returns a valid image-prompt structured dict; records every call."""

    def __init__(self, output=None, error=None):
        self.output = output or make_prompt()
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.output.model_dump()


class CountingProvider:
    """Wraps a real MockImageProvider and counts generate_images calls."""

    def __init__(self, settings):
        self.inner = MockImageProvider(settings)
        self.calls = []
        self.name = self.inner.name

    def generate_images(self, prompt, options=None, count=2):
        self.calls.append({"prompt": prompt, "options": options, "count": count})
        return self.inner.generate_images(prompt, options=options, count=count)

    def capabilities(self):
        return self.inner.capabilities()

    def supports(self, capability):
        return self.inner.supports(capability)


def make_settings(tmp_path):
    return Settings(env={"STORAGE_PATH": str(tmp_path)})


def make_project(tmp_path, n=3):
    """A project sitting in GENERATING_IMAGES with a storyboard + bible in place."""
    agent = MasterAgent.create("Seed growing from seed to flowering plant")
    agent.start_planning()
    scenes = [{"description": f"Scene {i} visual", "duration_seconds": 4} for i in range(1, n + 1)]
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
            "color_appearance": "rich greens and warm highlights",
            "physical_characteristics": "single seed to a bloom",
            "continuity_rules": "same framing, fixed camera",
            "negative_constraints": "no people, no text overlays",
            "aspect_ratio": "9:16",
            "target_duration": 30.0,
        },
    )
    agent.approve_story()
    return agent


def scene_of(project_or_agent, index=0):
    project = getattr(project_or_agent, "project", project_or_agent)
    return project.scenes[index]


def candidate(scene, cid):
    return next(c for c in scene.image_candidates if c.candidate_id == cid)


# =========================================================================
# 1. & 2. Prompt generation: context + structured output
# =========================================================================


def test_image_prompt_generation_uses_full_context(tmp_path):
    agent = make_project(tmp_path)
    llm = FakeLLM()
    img = ImageAgent(llm=llm, image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)

    assert len(llm.calls) == 3  # one per scene
    for call in llm.calls:
        assert "Seed growing from seed to flowering plant" in call["user_message"]
        assert "dark soil" in call["user_message"]
        assert "9:16" in call["user_message"]
        assert "PROJECT BIBLE" in call["user_message"]


def test_prompt_includes_previous_and_next_scene(tmp_path):
    agent = make_project(tmp_path)
    llm = FakeLLM()
    img = ImageAgent(llm=llm, image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)

    scene2_call = llm.calls[1]["user_message"]
    assert "action 1" in scene2_call and "PREVIOUS SCENE" in scene2_call
    assert "action 3" in scene2_call and "NEXT SCENE" in scene2_call


def test_structured_prompt_output_stored_per_candidate(tmp_path):
    agent = make_project(tmp_path, n=1)
    llm = FakeLLM()
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=llm, image_provider=provider)
    img.run(agent.project)

    scene = scene_of(agent.project)
    assert scene.image_prompt == llm.output.prompt_a
    a, b = candidate(scene, "A"), candidate(scene, "B")
    assert a.prompt == llm.output.prompt_a
    assert b.prompt == llm.output.prompt_b
    assert a.prompt != b.prompt


def test_invalid_prompt_output_fails_scene(tmp_path):
    agent = make_project(tmp_path, n=1)
    # Same prompt for A and B is invalid (B must differ).
    llm = FakeLLM(output=make_prompt(prompt_a="identical long prompt content", prompt_b="identical long prompt content"))
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=llm, image_provider=provider)
    summary = img.run(agent.project)

    assert summary["scenes_failed"] == 1
    assert summary["scenes_ready"] == 0
    assert len(provider.calls) == 0  # nothing sent to the image provider
    scene = scene_of(agent.project)
    assert not any(c.status == "SUCCEEDED" for c in scene.image_candidates)


def test_prompt_provider_error_marks_scene_failed(tmp_path):
    agent = make_project(tmp_path, n=1)
    llm = FakeLLM(error=ProviderError("LLM down"))
    img = ImageAgent(llm=llm, image_provider=CountingProvider(make_settings(tmp_path)))
    summary = img.run(agent.project)
    assert summary["scenes_failed"] == 1
    assert "LLM down" in candidate(scene_of(agent.project), "A").error


# =========================================================================
# 3. & 4. Two candidate images per scene
# =========================================================================


def test_two_candidate_images_per_scene(tmp_path):
    agent = make_project(tmp_path, n=1)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    img.run(agent.project)

    scene = scene_of(agent.project)
    assert len(scene.image_candidates) == 2
    assert {c.candidate_id for c in scene.image_candidates} == {"A", "B"}
    assert all(c.status == "SUCCEEDED" for c in scene.image_candidates)
    assert len(provider.calls) == 2
    assert provider.calls[0]["prompt"].startswith("Macro close-up")
    assert "top-down" in provider.calls[1]["prompt"]  # meaningfully different


def test_all_scenes_generated_in_order(tmp_path):
    agent = make_project(tmp_path, n=3)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    summary = img.run(agent.project)

    assert summary == {"scenes_ready": 3, "scenes_generated": 3, "scenes_failed": 0}
    assert len(provider.calls) == 6  # 3 scenes x 2 candidates
    for scene in agent.project.scenes:
        assert len([c for c in scene.image_candidates if c.status == "SUCCEEDED"]) == 2


# =========================================================================
# 4. Candidate A/B persistence
# =========================================================================


def test_candidate_records_persist_with_all_fields(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    project_id = agent.project.id
    scene_id = scene_of(agent.project).id

    reloaded = load_project(project_id)
    scene = scene_of(reloaded)
    assert len(scene.image_candidates) == 2
    for c in scene.image_candidates:
        assert c.project_id == project_id
        assert c.scene_id == scene_id
        assert c.candidate_id in ("A", "B")
        assert c.provider == "mock"
        assert c.model
        assert c.generation_id
        assert c.prompt
        assert c.storage_path
        assert c.status == "SUCCEEDED"
        assert c.error is None
        assert c.created_at
        assert Path(c.storage_path).exists()
        # Stored inside the project's own asset dir (stable across restarts).
        assert project_asset_dir(project_id, "images") == Path(c.storage_path).parent


# =========================================================================
# 5. & 6. Candidate selection & locking
# =========================================================================


def test_select_candidate_locks_scene(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)

    agent.select_image(scene_of(agent.project).id, "B")
    scene = scene_of(agent.project)
    assert scene.selected_image_id == "B"
    assert scene.selected_image == candidate(scene, "B").storage_path
    assert Path(scene.selected_image).exists()

    reloaded = MasterAgent.load(agent.project.id)
    assert scene_of(reloaded).selected_image_id == "B"


def test_select_invalid_candidate_raises(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    from agents.master import InvalidTransitionError

    with pytest.raises(InvalidTransitionError):
        agent.select_image(scene_of(agent.project).id, "C")


def test_locked_scenes_not_regenerated(tmp_path):
    agent = make_project(tmp_path, n=2)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    img.run(agent.project)
    calls_after_first = len(provider.calls)

    # Lock scene 1, then re-run: nothing new should be generated.
    agent.select_image(scene_of(agent.project, 0).id, "A")
    img.run(agent.project)
    assert len(provider.calls) == calls_after_first
    assert scene_of(agent.project, 0).selected_image_id == "A"


# =========================================================================
# 7. Candidate regeneration
# =========================================================================


def test_regenerate_single_candidate(tmp_path):
    agent = make_project(tmp_path, n=1)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    img.run(agent.project)

    scene = scene_of(agent.project)
    old_a = candidate(scene, "A")
    old_a_path = old_a.storage_path
    old_a_gen_id = old_a.generation_id
    old_b_path = candidate(scene, "B").storage_path

    img.regenerate_candidate(agent.project, scene.id, "A", model=None)

    assert len(provider.calls) == 3  # 2 initial + 1 for A
    new_a = candidate(scene_of(agent.project), "A")
    assert new_a.storage_path != old_a_path
    assert new_a.generation_id != old_a_gen_id
    # B untouched (same record, same path).
    assert candidate(scene_of(agent.project), "B").storage_path == old_b_path


def test_regenerate_preserves_existing_selection(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    scene = scene_of(agent.project)
    agent.select_image(scene.id, "B")

    img.regenerate_candidate(agent.project, scene.id, "A")

    scene = scene_of(agent.project)
    assert scene.selected_image_id == "B"
    assert scene.selected_image == candidate(scene, "B").storage_path


def test_regenerate_unknown_candidate_raises(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    with pytest.raises(Exception):
        img.regenerate_candidate(agent.project, scene_of(agent.project).id, "C")


# =========================================================================
# 8. & 9. Failed candidate / failed scene
# =========================================================================


class FlakyProvider:
    """Fails candidate B, keeps A. Counts calls."""

    name = "flaky"

    def __init__(self, settings, fail_all=False, fail_prompt_substring="top-down"):
        self.inner = MockImageProvider(settings)
        self.calls = []
        self.fail_all = fail_all
        self.fail_prompt_substring = fail_prompt_substring
        self.enabled = True

    def generate_images(self, prompt, options=None, count=2):
        self.calls.append(prompt)
        if self.enabled and (self.fail_all or self.fail_prompt_substring in prompt):
            raise ProviderGenerationError(f"generation rejected for prompt: {prompt[:30]}")
        return self.inner.generate_images(prompt, options=options, count=count)

    def capabilities(self):
        return self.inner.capabilities()


def test_failed_candidate_other_succeeds(tmp_path):
    agent = make_project(tmp_path, n=1)
    provider = FlakyProvider(make_settings(tmp_path), fail_prompt_substring="top-down")
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    summary = img.run(agent.project)

    assert summary["scenes_ready"] == 1  # B failed but A still allows selection
    scene = scene_of(agent.project)
    assert candidate(scene, "A").status == "SUCCEEDED"
    assert candidate(scene, "B").status == "FAILED"
    assert "generation rejected" in candidate(scene, "B").error
    assert scene.image_a and scene.image_b is None

    # A can be locked even though B failed.
    agent.select_image(scene.id, "A")
    assert scene.selected_image_id == "A"

    # Retry of B succeeds once the provider recovers.
    provider.enabled = False
    img.regenerate_candidate(agent.project, scene.id, "B")
    assert candidate(scene_of(agent.project), "B").status == "SUCCEEDED"


def test_failed_scene_allows_retry(tmp_path):
    agent = make_project(tmp_path, n=1)
    provider = FlakyProvider(make_settings(tmp_path), fail_all=True)
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    summary = img.run(agent.project)

    assert summary["scenes_ready"] == 0
    assert summary["scenes_failed"] == 1
    scene = scene_of(agent.project)
    assert all(c.status == "FAILED" for c in scene.image_candidates)
    assert all(c.error for c in scene.image_candidates)

    # Retrying calls the provider again (no auto loop, but retry allowed).
    calls = len(provider.calls)
    summary = img.run(agent.project)
    assert summary["scenes_ready"] == 0
    assert len(provider.calls) == calls + 2


# =========================================================================
# 10. & 16. Restart / recovery & no duplicate generation
# =========================================================================


def test_restart_resumes_without_regenerating_completed_scenes(tmp_path):
    agent = make_project(tmp_path, n=3)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)

    # Simulate a crash after the first scene was generated.
    img.generate_scene(agent.project, scene_of(agent.project, 0), index=0)
    save_initial(agent.project)

    calls_before = len(provider.calls)
    reloaded = MasterAgent.load(agent.project.id)
    summary = ImageAgent(llm=FakeLLM(), image_provider=provider).run(reloaded.project)

    assert summary["scenes_ready"] == 3
    # Scene 0 was NOT regenerated; only scenes 1 & 2 got their pairs.
    assert len(provider.calls) == calls_before + 4


def save_initial(project):
    from backend.storage import save_project

    save_project(project)


def test_run_is_idempotent_after_restart(tmp_path):
    agent = make_project(tmp_path, n=2)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    img.run(agent.project)
    first_call_count = len(provider.calls)

    reloaded = MasterAgent.load(agent.project.id)
    again = ImageAgent(llm=FakeLLM(), image_provider=provider).run(reloaded.project)
    assert again["scenes_generated"] == 0
    assert again["scenes_ready"] == 2
    assert len(provider.calls) == first_call_count
    # Storyboard + bible survived the restart.
    assert reloaded.project.storyboard["title"] == "Seed to Bloom"
    assert reloaded.project.project_bible["subject"] == "seed"


# =========================================================================
# 11. Scene ordering (stays in storyboard order)
# =========================================================================


def test_scenes_remain_in_storyboard_order(tmp_path):
    agent = make_project(tmp_path, n=3)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    # Storyboard scene numbers stay in order and every scene keeps its own pair.
    assert [s["scene_number"] for s in agent.project.storyboard["scenes"]] == [1, 2, 3]
    for scene in agent.project.scenes:
        assert len(scene.image_candidates) == 2
        assert {c.scene_id for c in scene.image_candidates} == {scene.id}
        assert scene.image_prompt  # resolved for each scene in order


# =========================================================================
# 12. Master Agent state transitions
# =========================================================================


def test_state_transitions_through_image_selection(tmp_path):
    agent = make_project(tmp_path, n=1)
    assert agent.project.status == ProjectStatus.GENERATING_IMAGES

    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    agent.images_generated()
    assert agent.project.status == ProjectStatus.WAITING_FOR_IMAGE_SELECTION

    agent.select_images({scene_of(agent.project).id: "A"})
    assert agent.project.status == ProjectStatus.GENERATING_VIDEOS
    assert agent.project.scenes[0].selected_image_id == "A"


def test_select_one_does_not_transition(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    img.run(agent.project)
    agent.images_generated()
    agent.select_image(scene_of(agent.project).id, "A")
    assert agent.project.status == ProjectStatus.WAITING_FOR_IMAGE_SELECTION


def test_approve_moves_to_generating_images(tmp_path):
    agent = make_project(tmp_path, n=1)
    assert agent.project.status == ProjectStatus.GENERATING_IMAGES  # after approve_story


# =========================================================================
# 13. 9:16 normalized aspect ratio through the provider abstraction
# =========================================================================


def test_agent_requests_normalized_9_16(tmp_path):
    agent = make_project(tmp_path, n=1)
    provider = CountingProvider(make_settings(tmp_path))
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    img.run(agent.project)

    assert len(provider.calls) == 2
    for call in provider.calls:
        options = call["options"]
        assert isinstance(options, GenerationOptions)
        assert options.aspect_ratio_value() == "9:16"
    assert agent.project.settings.aspect_ratio == "9:16"


# =========================================================================
# 14. Provider errors surface as failed candidates (no crash)
# =========================================================================


def test_provider_error_stored_per_candidate(tmp_path):
    agent = make_project(tmp_path, n=1)
    provider = FlakyProvider(make_settings(tmp_path), fail_all=True)
    img = ImageAgent(llm=FakeLLM(), image_provider=provider)
    summary = img.run(agent.project)
    assert summary["scenes_failed"] == 1
    for c in scene_of(agent.project).image_candidates:
        assert "rejected" in c.error


# =========================================================================
# 15. Mock image provider used in tests
# =========================================================================


def test_mock_image_provider_writes_real_pngs(tmp_path):
    agent = make_project(tmp_path, n=1)
    img = ImageAgent(
        llm=FakeLLM(),
        image_provider=MockImageProvider(make_settings(tmp_path)),
    )
    img.run(agent.project)
    for c in scene_of(agent.project).image_candidates:
        data = Path(c.storage_path).read_bytes()
        assert data.startswith(b"\x89PNG")


# =========================================================================
# API-level integration (inject fake agent, image provider is mock)
# =========================================================================


def test_api_generate_select_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_project(tmp_path, n=2)

    fake_llm = FakeLLM()
    provider = CountingProvider(make_settings(tmp_path))
    fake_img = ImageAgent(llm=fake_llm, image_provider=provider)
    monkeypatch.setattr(api_module, "get_image_agent", lambda: fake_img)

    client = TestClient(api_module.app)

    resp = client.post(f"/projects/{agent.project.id}/images/generate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "WAITING_FOR_IMAGE_SELECTION"
    assert len(body["scenes"]) == 2
    for scene in body["scenes"]:
        assert len(scene["image_candidates"]) == 2

    scene_id = body["scenes"][0]["id"]
    resp = client.post(
        f"/projects/{agent.project.id}/images/select-one",
        json={"scene_id": scene_id, "candidate": "A"},
    )
    assert resp.status_code == 200
    assert resp.json()["scenes"][0]["selected_image_id"] == "A"
    assert resp.json()["status"] == "WAITING_FOR_IMAGE_SELECTION"

    resp = client.post(
        f"/projects/{agent.project.id}/images/select",
        json={"selections": {scene["id"]: "A" for scene in body["scenes"]}},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "GENERATING_VIDEOS"

    resp = client.post(f"/projects/{agent.project.id}/images/select", json={"selections": {}})
    assert resp.status_code == 409  # already past selection


def test_api_regenerate_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_project(tmp_path, n=1)
    fake_llm = FakeLLM()
    provider = CountingProvider(make_settings(tmp_path))
    fake_img = ImageAgent(llm=fake_llm, image_provider=provider)
    monkeypatch.setattr(api_module, "get_image_agent", lambda: fake_img)

    client = TestClient(api_module.app)
    client.post(f"/projects/{agent.project.id}/images/generate")

    counts = len(provider.calls)
    scene_id = agent.project.scenes[0].id
    resp = client.post(
        f"/projects/{agent.project.id}/images/regenerate",
        json={"scene_id": scene_id, "candidate_id": "B"},
    )
    assert resp.status_code == 200
    assert len(provider.calls) == counts + 1
    assert resp.json()["scenes"][0]["image_candidates"][1]["candidate_id"] == "B"


def test_api_generate_requires_selection_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_project(tmp_path, n=1)
    fake_img = ImageAgent(llm=FakeLLM(), image_provider=CountingProvider(make_settings(tmp_path)))
    monkeypatch.setattr(api_module, "get_image_agent", lambda: fake_img)
    client = TestClient(api_module.app)
    agent.images_generated()
    resp = client.post(f"/projects/{agent.project.id}/images/generate")
    assert resp.status_code == 200  # allowed from WAITING_FOR_IMAGE_SELECTION too
    resp = client.post(f"/projects/{agent.project.id}/images/generate")
    assert resp.status_code == 200
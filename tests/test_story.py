import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_story_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from backend.models import ProjectStatus
from agents.master import MasterAgent, InvalidTransitionError
from agents.story import StoryAgent
from agents.story_schemas import (
    StoryboardOutput,
    StoryScene,
    ProjectBible,
    ValidationError,
    validate_scenes,
    validate_project_bible,
    validate_total_duration,
    validate_storyboard_output,
)


@pytest.fixture(autouse=True)
def clean_storage():
    storage_mod.STORAGE_DIR = TEST_STORAGE
    storage_mod.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for f in storage_mod.STORAGE_DIR.glob("*.json"):
        f.unlink()
    yield
    for f in storage_mod.STORAGE_DIR.glob("*.json"):
        f.unlink()


def make_bible(**overrides):
    defaults = dict(
        subject="seed",
        environment="close-up on dark soil",
        time_lighting="warm golden hour",
        camera="locked down macro",
        lens_look="85mm macro, shallow DOF",
        visual_style="Photorealistic",
        color_appearance="rich greens, warm highlights",
        physical_characteristics="single green sprout",
        continuity_rules="same camera angle throughout",
        negative_constraints="no people, no text overlays",
        aspect_ratio="9:16",
        target_duration=30,
    )
    defaults.update(overrides)
    return ProjectBible(**defaults)


def make_scenes(n=4):
    scenes = []
    for i in range(1, n + 1):
        scenes.append(
            StoryScene(
                scene_id=i,
                scene_number=i,
                title=f"Scene {i}",
                visual_description=f"visual {i}",
                action=f"action {i}",
                estimated_duration=30 / n,
                continuity_notes=f"notes {i}",
            )
        )
    return scenes


def make_output(**overrides):
    defaults = dict(
        title="Seed to Sprout",
        summary="A seed grows into a sprout.",
        scenes=make_scenes(),
        project_bible=make_bible(),
    )
    defaults.update(overrides)
    return StoryboardOutput(**defaults)


class FakeLLM:
    """Reusable fake LLM returning a configurable StoryboardOutput dict."""

    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if isinstance(self.output, StoryboardOutput):
            return self.output.model_dump()
        return self.output


VALID_OUTPUT = make_output()


# ---------- validation ----------

def test_valid_output_passes():
    assert validate_storyboard_output(make_output()) == []


def test_scene_duration_validation():
    scenes = make_scenes()
    scenes[0].estimated_duration = -5
    issues = validate_scenes(scenes)
    assert any("estimated_duration" in i for i in issues)


def test_scene_action_required():
    scenes = make_scenes()
    scenes[1].action = " "
    issues = validate_scenes(scenes)
    assert any("action is empty" in i for i in issues)


def test_scene_numbers_must_be_sequential():
    scenes = make_scenes()
    scenes[2].scene_number = 9
    issues = validate_scenes(scenes)
    assert any("sequential" in i for i in issues)


def test_project_bible_must_be_complete():
    bible = make_bible(subject="")
    issues = validate_project_bible(bible)
    assert any("subject" in i for i in issues)


def test_aspect_ratio_validation():
    bible = make_bible(aspect_ratio="5:4")
    issues = validate_project_bible(bible)
    assert any("aspect_ratio" in i for i in issues)


def test_total_duration_validation():
    scenes = make_scenes()
    total = sum(s.estimated_duration for s in scenes)
    issues = validate_total_duration(scenes, total)
    assert issues == []


def test_total_duration_deviation_rejected():
    scenes = make_scenes()
    issues = validate_total_duration(scenes, 100)  # way off from ~30
    assert any("total duration" in i for i in issues)


def test_empty_storyboard_rejected():
    out = make_output(scenes=[])
    issues = validate_storyboard_output(out)
    assert any("at least one scene" in i for i in issues)


# ---------- StoryAgent.generate ----------

def test_generate_builds_valid_output():
    agent = StoryAgent(llm=FakeLLM(output=VALID_OUTPUT))
    out = agent.generate("seed grows", "Timelapse", "Photorealistic", 30, "9:16")
    assert out.title == "Seed to Sprout"
    assert len(out.scenes) == 4
    assert out.project_bible.aspect_ratio == "9:16"


def test_generate_injects_idea_and_settings_into_prompt():
    agent = StoryAgent(llm=FakeLLM(output=VALID_OUTPUT))
    agent.generate("my idea", "Timelapse", "Cinematic", 30, "9:16")
    user_msg = agent._llm.calls[0]["user_message"]
    assert "my idea" in user_msg
    assert "Cinematic" in user_msg
    assert "9:16" in user_msg


def test_generate_includes_regeneration_feedback():
    agent = StoryAgent(llm=FakeLLM(output=VALID_OUTPUT))
    agent.generate("idea", "t", "s", 30, "9:16", feedback="make it brighter")
    assert "make it brighter" in agent._llm.calls[0]["user_message"]


def test_generate_rejects_invalid_model_output():
    bad = make_output(scenes=[])
    agent = StoryAgent(llm=FakeLLM(output=bad))
    with pytest.raises(ValidationError):
        agent.generate("idea", "t", "s", 30, "9:16")


def test_generate_raises_when_total_duration_off():
    out = make_output(scenes=make_scenes())
    agent = StoryAgent(llm=FakeLLM(output=out))
    with pytest.raises(ValidationError):
        agent.generate("idea", "t", "s", 100, "9:16")


def test_generate_raises_on_provider_error():
    from agents.llm import ProviderError
    agent = StoryAgent(llm=FakeLLM(error=ProviderError("boom")))
    with pytest.raises(ValidationError):
        agent.generate("idea", "t", "s", 30, "9:16")


# ---------- MasterAgent integration ----------

def test_plan_story_moves_to_approval_and_stores_output():
    agent = MasterAgent.create("seed grows at sunrise")
    agent.start_planning()
    agent.plan_story(story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))
    assert agent.project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL
    assert agent.project.storyboard is not None
    assert agent.project.project_bible["subject"] == "seed"
    assert len(agent.project.scenes) == 4
    assert agent.is_paused


def test_plan_story_requires_planning_state():
    agent = MasterAgent.create("idea")
    with pytest.raises(InvalidTransitionError):
        agent.plan_story(story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))


def test_approve_after_plan_story_continues():
    agent = MasterAgent.create("idea")
    agent.start_planning()
    agent.plan_story(story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))
    agent.approve_story()
    assert agent.project.status == ProjectStatus.GENERATING_IMAGES


def test_reject_goes_back_to_planning():
    agent = MasterAgent.create("idea")
    agent.start_planning()
    agent.plan_story(story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))
    agent.reject_story("too bland")
    assert agent.project.status == ProjectStatus.PLANNING
    assert agent.project.storyboard is not None  # preserved until regeneration


def test_regenerate_replaces_story_and_pauses():
    agent = MasterAgent.create("idea")
    agent.start_planning()
    v1 = make_output(title="Version One")
    v2 = make_output(title="Version Two")
    fake = FakeLLM()
    fake.output = v1
    agent.plan_story(story_agent=StoryAgent(llm=fake))
    assert agent.project.storyboard["title"] == "Version One"

    fake.output = v2
    agent.regenerate_story(reason="make it punchier", story_agent=StoryAgent(llm=fake))
    assert agent.project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL
    assert agent.project.storyboard["title"] == "Version Two"
    # feedback captured in prompt
    assert "make it punchier" in fake.calls[-1]["user_message"]


def test_regenerate_from_approval_state():
    agent = MasterAgent.create("idea")
    agent.start_planning()
    agent.plan_story(story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))
    v2 = make_output(title="Regenerated")
    agent.regenerate_story(
        reason="retry", story_agent=StoryAgent(llm=FakeLLM(output=v2))
    )
    assert agent.project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL
    assert agent.project.storyboard["title"] == "Regenerated"


def test_no_images_generated_during_regeneration():
    """Reject/regenerate must not call image or video generation."""
    agent = MasterAgent.create("idea")
    agent.start_planning()
    agent.plan_story(story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))
    agent.regenerate_story(reason="again", story_agent=StoryAgent(llm=FakeLLM(output=VALID_OUTPUT)))
    # Still stuck at approval, never advanced toward images
    assert agent.project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL
    assert all(s.image_a is None and s.video_clip is None for s in agent.project.scenes)

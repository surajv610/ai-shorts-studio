import sys
import tempfile
import os
from pathlib import Path

# Ensure the project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point storage at a temp dir for tests
import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from datetime import datetime, timezone

from backend.models import ProjectStatus, ProjectSettings, QCReport, QCStatus
from agents.master import (
    MasterAgent,
    InvalidTransitionError,
    ProjectNotFoundError,
    can_transition,
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


@pytest.fixture
def agent():
    a = MasterAgent.create("A city timelapse at night")
    return a


def run_until(agent: MasterAgent, target: ProjectStatus):
    """Advance the agent through the happy-path states up to a target."""
    agent.start_planning()
    if target == ProjectStatus.PLANNING:
        return agent
    agent.submit_storyboard(
        storyboard={"summary": "s"},
        scenes=[{"description": "Scene 1", "duration_seconds": 4}],
        project_bible={"world": "city"},
    )
    if target == ProjectStatus.WAITING_FOR_STORY_APPROVAL:
        return agent
    agent.approve_story()
    if target == ProjectStatus.GENERATING_IMAGES:
        return agent
    agent.images_generated([{"scene_id": agent.project.scenes[0].id, "image_a": "a.png", "image_b": "b.png"}])
    if target == ProjectStatus.WAITING_FOR_IMAGE_SELECTION:
        return agent
    agent.select_images({agent.project.scenes[0].id: "a.png"})
    if target == ProjectStatus.GENERATING_VIDEOS:
        return agent
    agent.videos_generated({agent.project.scenes[0].id: "c.mp4"})
    if target == ProjectStatus.ASSEMBLING:
        return agent
    agent.assembly_complete("final.mp4")
    if target == ProjectStatus.READY_FOR_REVIEW:
        return agent
    agent.start_quality_check()
    if target == ProjectStatus.QUALITY_CHECK:
        return agent
    agent.qc_complete(
        QCReport(
            project_id=agent.project.id,
            assembly_id=agent.project.current_assembly_id or 'asm-legacy',
            status=QCStatus.PASSED,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    if target == ProjectStatus.GENERATING_METADATA:
        return agent
    agent.metadata_start()
    agent.metadata_generated("Title", "Desc", ["#shorts"])
    if target == ProjectStatus.WAITING_FOR_FINAL_APPROVAL:
        return agent
    agent.final_approve()
    return agent


# ---------- valid transitions ----------

def test_happy_path_full_workflow(agent):
    run_until(agent, ProjectStatus.COMPLETED)
    assert agent.project.status == ProjectStatus.COMPLETED
    assert agent.project.final_video == "final.mp4"
    assert agent.project.youtube_metadata.title == "Title"


def test_valid_transition_check():
    assert can_transition(ProjectStatus.DRAFT, ProjectStatus.PLANNING)
    assert can_transition(ProjectStatus.WAITING_FOR_STORY_APPROVAL, ProjectStatus.GENERATING_IMAGES)
    assert can_transition(ProjectStatus.WAITING_FOR_FINAL_APPROVAL, ProjectStatus.COMPLETED)


def test_cancel_from_any_active_state(agent):
    run_until(agent, ProjectStatus.WAITING_FOR_STORY_APPROVAL)
    agent.cancel()
    assert agent.project.status == ProjectStatus.CANCELLED


def test_fail_transition(agent):
    agent.start_planning()
    agent.fail("provider error")
    assert agent.project.status == ProjectStatus.FAILED
    assert agent.project.errors


def test_rework_loop_back_from_approval(agent):
    run_until(agent, ProjectStatus.WAITING_FOR_STORY_APPROVAL)
    agent.reject_story("rewrite")
    assert agent.project.status == ProjectStatus.PLANNING


def test_qc_fail_returns_to_review(agent):
    run_until(agent, ProjectStatus.QUALITY_CHECK)
    agent.qc_complete(
        QCReport(
            project_id=agent.project.id,
            assembly_id=agent.project.current_assembly_id or 'asm-legacy',
            status=QCStatus.FAILED,
            completed_at=datetime.now(timezone.utc).isoformat(),
            findings=["scene 1 clip is not the current generation"],
        )
    )
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    assert agent.project.qc_result.passed is False
    assert agent.project.qc_result.issues == ["scene 1 clip is not the current generation"]


# ---------- invalid transitions ----------

def test_invalid_transition_from_draft():
    assert not can_transition(ProjectStatus.DRAFT, ProjectStatus.COMPLETED)


def test_invalid_transition_raises(agent):
    with pytest.raises(InvalidTransitionError):
        agent.approve_story()  # still DRAFT


def test_cannot_advance_from_terminal_states(agent):
    run_until(agent, ProjectStatus.COMPLETED)
    with pytest.raises(InvalidTransitionError):
        agent.cancel()


# ---------- persistence ----------

def test_persistence_roundtrip(agent):
    project_id = agent.project.id
    agent.start_planning()
    reloaded = MasterAgent.load(project_id)
    assert reloaded.project.status == ProjectStatus.PLANNING
    assert reloaded.project.id == project_id


def test_load_missing_project_raises():
    with pytest.raises(ProjectNotFoundError):
        MasterAgent.load("does-not-exist")


def test_project_has_unique_ids():
    a1 = MasterAgent.create("idea one")
    a2 = MasterAgent.create("idea two")
    assert a1.project.id != a2.project.id


# ---------- pause / resume ----------

def test_pauses_at_story_approval(agent):
    run_until(agent, ProjectStatus.WAITING_FOR_STORY_APPROVAL)
    assert agent.is_paused
    assert agent.project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL
    # Cannot skip ahead while paused
    with pytest.raises(InvalidTransitionError):
        agent.images_generated([])
    # Resume by approving
    agent.approve_story()
    assert not agent.is_paused


def test_pauses_at_image_selection(agent):
    run_until(agent, ProjectStatus.WAITING_FOR_IMAGE_SELECTION)
    assert agent.is_paused
    # Must select images before continuing
    with pytest.raises(InvalidTransitionError):
        agent.select_images({})  # no selections given
    agent.select_images({agent.project.scenes[0].id: "a.png"})
    assert not agent.is_paused


def test_pauses_at_final_approval(agent):
    run_until(agent, ProjectStatus.WAITING_FOR_FINAL_APPROVAL)
    assert agent.is_paused
    agent.final_approve()
    assert not agent.is_paused
    assert agent.project.status == ProjectStatus.COMPLETED


def test_state_timestamps_recorded(agent):
    run_until(agent, ProjectStatus.WAITING_FOR_FINAL_APPROVAL)
    ts = agent.project.state_timestamps
    assert ProjectStatus.DRAFT.value in ts
    assert ProjectStatus.PLANNING.value in ts
    assert ProjectStatus.GENERATING_IMAGES.value in ts
    assert ProjectStatus.GENERATING_VIDEOS.value in ts
    assert ProjectStatus.WAITING_FOR_FINAL_APPROVAL.value in ts

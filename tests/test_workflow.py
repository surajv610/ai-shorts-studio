"""Tests for the deterministic production-workflow engine + summary endpoints.

Covers the Unified Daily Production Workflow phase: the 8-stage pipeline
(STORY, IMAGES, VIDEOS, ASSEMBLY, QC, METADATA, FINAL_REVIEW, READY_TO_UPLOAD),
the Next-Action engine (deterministic, NO LLM), resume/recovery, human
checkpoints, stale-state propagation, multi-project isolation, dashboard cards,
deletion, error UX inputs, generation estimates, and a full mock end-to-end run.

Everything runs offline against mock providers — no external AI/paid APIs.
"""

import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_workflow_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from backend.models import (
    AssemblyRecord,
    AssemblyStatus,
    ImageCandidate,
    MetadataRecord,
    MetadataStatus,
    Project,
    ProjectStatus,
    QCCheck,
    QCReport,
    QCStatus,
    Scene,
    VideoGeneration,
)
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
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    monkeypatch.setenv("VIDEO_PROVIDER", "mock")
    monkeypatch.setenv("VIDEO_POLL_INTERVAL", "0.001")
    monkeypatch.setenv("VIDEO_POLL_TIMEOUT", "10")


@pytest.fixture
def c():
    from fastapi.testclient import TestClient

    import backend.api as api_module

    return TestClient(api_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def client():
    from fastapi.testclient import TestClient

    import backend.api as api_module

    return TestClient(api_module.app, raise_server_exceptions=False)


def stash(project: Project) -> str:
    save_project(project)
    return project.id


def make_project(status=ProjectStatus.DRAFT, n=3, **kw) -> str:
    """A project in an arbitrary persisted state (bypasses transitions)."""
    p = Project(idea=kw.pop("idea", "Seed growing from seed to flowering plant"))
    p.status = status
    p.name = kw.pop("name", "Seed Timelapse")
    if kw.pop("story", False):
        p.storyboard = {
            "title": "Seed to Bloom",
            "summary": "A seed grows into a flower.",
            "scenes": [{"scene_number": i, "title": f"Scene {i}",
                        "visual_description": f"Scene {i} visual",
                        "estimated_duration": 4,
                        "action": f"action {i}",
                        "continuity_notes": "same plant"} for i in range(1, n + 1)],
        }
    for i in range(1, n + 1):
        s = Scene(description=f"Scene {i} visual", duration_seconds=4)
        if kw.get("images"):
            s.image_a = _path("images", s.id, "a.png")
            s.image_b = _path("images", s.id, "b.png")
            s.image_candidates = [
                ImageCandidate(
                    project_id=p.id, scene_id=s.id, candidate_id="A",
                    status="SUCCEEDED", storage_path=s.image_a, provider="mock",
                ),
                ImageCandidate(
                    project_id=p.id, scene_id=s.id, candidate_id="B",
                    status="SUCCEEDED", storage_path=s.image_b, provider="mock",
                ),
            ]
        if kw.get("selected"):
            s.selected_image = s.image_a
            s.selected_image_id = "A"
        if kw.get("clips"):
            s.video_clip = _path("videos", s.id, "clip.mp4")
            s.video_generations = [
                VideoGeneration(
                    project_id=p.id, scene_id=s.id, status="SUCCEEDED",
                    output_path=s.video_clip, provider="mock",
                )
            ]
            s.current_video_id = s.video_generations[0].attempt_id
        p.scenes.append(s)
    return stash(p)


def _path(kind, scene_id, name):
    return f"storage/projects/__probe__/{kind}/{scene_id}/{name}"


def next_action(pid, c=None):
    c = c or client()
    r = c.get(f"/projects/{pid}/next-action")
    assert r.status_code == 200, r.text
    return r.json()


def summary(pid, c=None):
    c = c or client()
    r = c.get(f"/projects/{pid}/summary")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Dashboard list + creation + deletion
# ---------------------------------------------------------------------------


def test_project_list_empty(c):
    assert c.get("/projects").json() == []


def test_project_card_created_draft(c):
    pid = c.post("/projects", json={"idea": "A tiny idea"}).json()["id"]
    cards = c.get("/projects").json()
    assert len(cards) == 1
    card = cards[0]
    assert card["id"] == pid
    assert card["status"] == "DRAFT"
    assert card["current_step"] == "Start Production"
    assert card["next_action"]["action"] == "START_PRODUCTION"
    assert card["progress"] == {"percent": 0, "done": 0, "total": 8}
    assert "thumbnail" in card and card["stale"] is False


def test_delete_project_removes_card(c):
    pid = c.post("/projects", json={"idea": "Soon deleted"}).json()["id"]
    assert len(c.get("/projects").json()) == 1
    assert c.delete(f"/projects/{pid}").json()["deleted"] is True
    assert c.get("/projects").json() == []
    assert c.get(f"/projects/{pid}").status_code == 404
    assert c.get(f"/projects/{pid}/summary").status_code == 404
    assert c.get(f"/projects/{pid}/next-action").status_code == 404


# ---------------------------------------------------------------------------
# 2. Next-action matrix (deterministic, no LLM / no state writes)
# ---------------------------------------------------------------------------


def test_next_action_matches_stage(c):
    pid = make_project(ProjectStatus.DRAFT)
    na = next_action(pid, c)
    assert na["action"] == "START_PRODUCTION" and na["step"] == "story"
    assert na["blocking"] is False and na["go"] is None

    pid = make_project(ProjectStatus.PLANNING)
    assert next_action(pid, c)["action"] == "GENERATE_STORY"

    pid = make_project(ProjectStatus.WAITING_FOR_STORY_APPROVAL)
    na = next_action(pid, c)
    assert na["action"] == "REVIEW_STORY" and na["blocking"] is True and na["go"] == ""

    # images: nothing generated yet
    pid = make_project(ProjectStatus.GENERATING_IMAGES, n=2, story=True)
    na = next_action(pid, c)
    assert na["action"] == "GENERATE_IMAGES" and na["operation"] == "images"
    assert na["label"].startswith("Generate Images (4 candidates across 2 scenes)")

    # images: candidates ready but no selection
    pid = make_project(ProjectStatus.WAITING_FOR_IMAGE_SELECTION, n=2, images=True, story=True)
    na = next_action(pid, c)
    assert na["action"] == "SELECT_IMAGES" and na["blocking"] is True and na["go"] == "images"

    # videos: nothing generated
    pid = make_project(ProjectStatus.GENERATING_VIDEOS, n=3, images=True, selected=True, story=True)
    na = next_action(pid, c)
    assert na["action"] == "GENERATE_VIDEOS" and na["operation"] == "videos"

    # videos: partial progress
    p = _load_meta(pid)
    p.scenes[0].video_clip = _path("videos", p.scenes[0].id, "clip.mp4")
    p.scenes[0].video_generations = [
        VideoGeneration(project_id=p.id, scene_id=p.scenes[0].id,
                        status="SUCCEEDED", output_path=p.scenes[0].video_clip)
    ]
    stash(p)
    na = next_action(pid, c)
    assert na["action"] == "CONTINUE_VIDEOS"

    # videos: all done -> assemble
    p = load_project(pid)
    for s in p.scenes:
        if not s.video_clip:
            s.video_clip = _path("videos", s.id, "clip.mp4")
            s.video_generations = [
                VideoGeneration(project_id=p.id, scene_id=s.id, status="SUCCEEDED",
                                output_path=s.video_clip)
            ]
    stash(p)
    na = next_action(pid, c)
    assert na["action"] == "ASSEMBLE_VIDEO" and na["operation"] is None


def test_next_action_terminal_states(c):
    pid = make_project(ProjectStatus.FAILED)
    na = next_action(pid, c)
    assert na["action"] == "RECOVER" and na["blocking"] is True

    pid = make_project(ProjectStatus.CANCELLED)
    na = next_action(pid, c)
    assert na["action"] == "CANCELLED" and na["blocking"] is True

    pid = make_project(ProjectStatus.COMPLETED)
    na = next_action(pid, c)
    assert na["action"] == "READY_TO_UPLOAD"


def test_assembly_failed_next_action_is_retry(c):
    pid = make_project(ProjectStatus.GENERATING_VIDEOS, n=2, images=True, selected=True, clips=True, story=True)
    p = load_project(pid)
    p.status = ProjectStatus.ASSEMBLY_FAILED
    p.assemblies = [
        AssemblyRecord(project_id=pid, status=AssemblyStatus.FAILED,
                       error="boom", scene_count=2)
    ]
    stash(p)
    na = next_action(pid, c)
    assert na["action"] == "REASSEMBLE_VIDEO" and na["label"] == "Retry Assembly"


def test_next_action_is_deterministic_and_write_free(c):
    pid = make_project(ProjectStatus.WAITING_FOR_IMAGE_SELECTION, n=2, images=True, story=True)
    before = load_project(pid).model_dump()
    a = next_action(pid, c)
    b = next_action(pid, c)
    assert a == b
    assert load_project(pid).model_dump() == before  # no writes, no LLM
    assert a["action"] == "SELECT_IMAGES"


def test_story_rejected_returns_to_generate(c):
    pid = make_project(ProjectStatus.PLANNING)
    assert next_action(pid, c)["action"] == "GENERATE_STORY"
    # a rejected story parks at PLANNING with no storyboard -> same next action
    assert next_action(pid, c)["blocking"] is False


# ---------------------------------------------------------------------------
# 3. Estimates (used for the generation confirmation dialogs)
# ---------------------------------------------------------------------------


def test_image_estimate_counts_missing_candidates(c):
    pid = make_project(ProjectStatus.GENERATING_IMAGES, n=3, story=True)
    est = summary(pid, c)["image_generation_estimate"]
    assert est == {"scenes": 3, "candidates_per_scene": 2, "total_candidates": 6}

    # one scene already has a successful A
    p = load_project(pid)
    p.scenes[0].image_candidates = [
        ImageCandidate(project_id=pid, scene_id=p.scenes[0].id, candidate_id="A",
                       status="SUCCEEDED", storage_path=_path("images", p.scenes[0].id, "a.png"))
    ]
    stash(p)
    est = summary(pid, c)["image_generation_estimate"]
    assert est == {"scenes": 3, "candidates_per_scene": 2, "total_candidates": 5}
    assert next_action(pid, c)["action"] == "GENERATE_IMAGES"


def test_video_estimate_counts_missing_clips(c):
    pid = make_project(ProjectStatus.GENERATING_VIDEOS, n=4, images=True, selected=True, clips=True, story=True)
    p = load_project(pid)
    p.scenes[2].video_clip = None
    p.scenes[2].video_generations = []
    stash(p)
    est = summary(pid, c)["video_generation_estimate"]
    assert est["scenes_to_generate"] == 1
    assert est["estimated_clips"] == 1
    assert est["existing_clips"] == 3
    assert next_action(pid, c)["action"] == "CONTINUE_VIDEOS"


# ---------------------------------------------------------------------------
# 4. Stale-state propagation
# ---------------------------------------------------------------------------


def test_scene_regeneration_propagates_stale_assembly_qc_metadata(c):
    """Regenerating a clip marks assembly + QC + metadata STALE so the next
    action is Reassemble -> Run QC -> Generate Metadata."""
    pid = _make_ready_project(c)
    p = load_project(pid)
    scene_id = p.scenes[0].id

    status = c.get(f"/projects/{pid}/qc/status").json()
    assert status["qc_current"] is True

    c.post(f"/projects/{pid}/videos/regenerate", json={"scene_id": scene_id})
    s = summary(pid, c)
    stage = {st["key"]: st for st in s["stages"]}
    assert stage["assembly"]["status"] == "stale"
    assert stage["qc"]["status"] == "stale"
    assert stage["metadata"]["status"] == "stale"
    assert stage["final_review"]["status"] == "waiting"
    assert next_action(pid, c)["action"] == "REASSEMBLE_VIDEO"

    # ... then reassemble, QC, metadata -> ready again
    c.post(f"/projects/{pid}/assembly/reassemble")
    assert next_action(pid, c)["action"] == "RUN_QC"
    c.post(f"/projects/{pid}/qc/run")
    assert next_action(pid, c)["action"] == "GENERATE_METADATA"
    c.post(f"/projects/{pid}/metadata/generate")
    assert next_action(pid, c)["action"] == "READY_TO_UPLOAD"
    s = summary(pid, c)
    assert s["progress"]["percent"] == 100
    assert all(st["status"] == "completed" for st in s["stages"])


# ---------------------------------------------------------------------------
# 5. QC FAILED + metadata blocking
# ---------------------------------------------------------------------------


def test_qc_failed_leads_to_final_review_and_blocks_metadata(c):
    pid = _assemble_project(c)
    record = c.get(f"/projects/{pid}/assembly/result").json()["assembly"]

    from agents.master import MasterAgent

    agent = MasterAgent.load(pid)
    agent.start_quality_check()
    report = QCReport(
        project_id=pid,
        assembly_id=record["assembly_id"],
        status=QCStatus.RUNNING,
        started_at="2026-01-01T00:00:00Z",
    )
    agent.qc_running(report)
    report.status = QCStatus.FAILED
    report.summary = "Video is 12s short of the target duration."
    report.checks = [
        QCCheck(check_name="duration", status="FAIL", severity="high",
                message="too short", measured_value=18, expected_value=30)
    ]
    report.findings = ["Video is too short."]
    report.completed_at = "2026-01-01T00:00:01Z"
    agent.qc_complete(report)

    s = summary(pid, c)
    assert s["qc"]["status"] == "FAILED"
    assert s["qc"]["current"] is True
    stage = {st["key"]: st for st in s["stages"]}
    assert stage["qc"]["status"] == "failed"
    assert s["progress"]["percent"] < 100
    assert stage["ready"]["status"] == "waiting"

    na = next_action(pid, c)
    assert na["action"] == "REVIEW_FINAL"
    assert na["blocking"] is True and na["go"] == "final"
    assert "QC found problems" in na["reason"]

    # metadata is blocked while QC FAILED (default config)
    ms = c.get(f"/projects/{pid}/metadata/status").json()
    assert ms["qc_blocked"] is True and ms["can_generate"] is False
    assert c.post(f"/projects/{pid}/metadata/generate").status_code == 409


def test_qc_warnings_still_ready(c):
    pid = _make_ready_project(c)
    from agents.master import MasterAgent

    agent = MasterAgent.load(pid)
    agent.start_quality_check()
    report = QCReport(
        project_id=pid,
        assembly_id=agent.successful_assembly().assembly_id,
        status=QCStatus.RUNNING,
    )
    agent.qc_running(report)
    report.status = QCStatus.WARNINGS
    report.summary = "Audio is missing but that is expected for this project."
    report.checks = [QCCheck(check_name="audio", status="WARN", severity="info", message="no audio")]
    agent.qc_complete(report)
    assert next_action(pid, c)["action"] == "READY_TO_UPLOAD"


# ---------------------------------------------------------------------------
# 6. Resume / persistence + multi-project isolation
# ---------------------------------------------------------------------------


def test_summary_survives_restart(c):
    pid = _make_ready_project(c)
    before = summary(pid, c)
    # simulate a process restart: fresh storage loader returns the same state
    from backend.storage import load_project

    reloaded = load_project(pid)
    assert reloaded is not None
    after = summary(pid, c)
    assert after["project_id"] == before["project_id"]
    assert after["next_action"] == before["next_action"]
    assert after["progress"] == before["progress"]
    assert after["qc"]["status"] == "PASSED"
    assert after["metadata"]["current"] is True


def test_two_projects_are_isolated(c):
    pid_a = _make_ready_project(c)
    pid_b = c.post("/projects", json={"idea": "Second, untouched idea"}).json()["id"]

    sa = summary(pid_a, c)
    sb = summary(pid_b, c)
    assert sa["project_id"] != sb["project_id"]
    assert sa["progress"]["percent"] == 100
    assert sb["progress"]["percent"] == 0
    assert sb["next_action"]["action"] == "START_PRODUCTION"
    assert sa["thumbnail"] != sb["thumbnail"]

    c.delete(f"/projects/{pid_b}")
    assert summary(pid_a, c)["project_id"] == pid_a  # untouched
    cards = c.get("/projects").json()
    assert [card["id"] for card in cards] == [pid_a]


# ---------------------------------------------------------------------------
# 7. Full mock end-to-end via the API (single-entry production workflow)
# ---------------------------------------------------------------------------


def test_mock_end_to_end_unified_workflow(c):
    pid = c.post("/projects", json={"idea": "Seed growing from seed to flowering plant"}).json()["id"]

    # DRAFT -> START_PRODUCTION
    na = next_action(pid, c)
    assert na["action"] == "START_PRODUCTION"
    c.post(f"/projects/{pid}/planning")
    c.post(f"/projects/{pid}/plan")
    assert next_action(pid, c)["action"] == "REVIEW_STORY"

    c.post(f"/projects/{pid}/story/approve")
    assert next_action(pid, c)["action"] == "GENERATE_IMAGES"
    c.post(f"/projects/{pid}/images/generate")
    assert next_action(pid, c)["action"] == "SELECT_IMAGES"

    project = c.get(f"/projects/{pid}").json()
    selections = {s["id"]: "A" for s in project["scenes"]}
    c.post(f"/projects/{pid}/images/select", json={"selections": selections})
    assert next_action(pid, c)["action"] == "GENERATE_VIDEOS"
    c.post(f"/projects/{pid}/videos/generate")
    assert next_action(pid, c)["action"] == "ASSEMBLE_VIDEO"

    c.post(f"/projects/{pid}/assembly/assemble")
    assert next_action(pid, c)["action"] == "RUN_QC"
    qc = c.post(f"/projects/{pid}/qc/run").json()
    assert qc["overall"] == "PASSED"
    assert next_action(pid, c)["action"] == "GENERATE_METADATA"
    gen = c.post(f"/projects/{pid}/metadata/generate").json()
    assert gen["status"] == "READY_FOR_REVIEW"

    na = next_action(pid, c)
    assert na["action"] == "READY_TO_UPLOAD"
    assert na["go"] == "final"
    s = summary(pid, c)
    assert s["progress"] == {"percent": 100, "done": 8, "total": 8}
    assert all(st["status"] == "completed" for st in s["stages"])
    assert s["final_video"]["exists"] is True
    assert s["final_video"]["url"].startswith("/assets/")
    assert s["metadata"]["current"] is True
    assert "title" in s["metadata"] and s["metadata"]["title"]
    assert {st["key"] for st in s["stages"]} == {
        "story", "images", "videos", "assembly", "qc", "metadata",
        "final_review", "ready",
    }

    c.delete(f"/projects/{pid}")


# ---------------------------------------------------------------------------
# 8. Summary payload contract (no secrets)
# ---------------------------------------------------------------------------


def test_summary_contract(c):
    pid = _make_ready_project(c)
    s = summary(pid, c)
    for key in (
        "project_id", "name", "idea", "status", "current_step", "progress",
        "stages", "next_action", "final_video", "qc", "metadata", "scenes",
        "thumbnail", "image_generation_estimate", "video_generation_estimate",
        "stale",
    ):
        assert key in s, key
    raw = c.get(f"/projects/{pid}").json()
    assert "youtube_metadata" in raw  # full project still available via /projects/:id
    # no secrets anywhere in the summary
    assert "api_key" not in str(s)
    assert "sk-" not in str(s)


# ---------------------------------------------------------------------------
# Shared state builders
# ---------------------------------------------------------------------------


def _load_meta(pid):
    return load_project(pid)


def _drive_to_assembly(c, pid):
    """Run Story -> Images -> Videos -> Assembly through the real API (mock)."""
    c.post(f"/projects/{pid}/planning")
    c.post(f"/projects/{pid}/plan")
    c.post(f"/projects/{pid}/story/approve")
    c.post(f"/projects/{pid}/images/generate")
    project = c.get(f"/projects/{pid}").json()
    selections = {s["id"]: "A" for s in project["scenes"]}
    c.post(f"/projects/{pid}/images/select", json={"selections": selections})
    c.post(f"/projects/{pid}/videos/generate")
    body = c.post(f"/projects/{pid}/assembly/assemble").json()
    assert body["status"] == "READY_FOR_REVIEW"
    assert body["assembly"]["status"] == "SUCCEEDED"
    return pid


def _assemble_project(c):
    pid = c.post("/projects", json={"idea": "Seed growing from seed to flowering plant"}).json()["id"]
    return _drive_to_assembly(c, pid)


def _make_ready_project(c):
    """A fully-produced project: story, images, videos, final video, QC, metadata."""
    pid = _assemble_project(c)
    assert c.post(f"/projects/{pid}/qc/run").json()["overall"] == "PASSED"
    assert c.post(f"/projects/{pid}/metadata/generate").json()["status"] == "READY_FOR_REVIEW"
    return pid
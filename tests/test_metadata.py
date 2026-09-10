"""Tests for the Metadata Agent + API + persistence + versioning.

Metadata produces the YouTube Shorts upload package for the CURRENT final
assembly: title, description, hashtags, keywords/category/content_summary. It
never uploads, schedules, publishes, or modifies the video / images / storyboard
/ Project Bible; the human reviews everything. All generation runs offline
against a counting fake LLM (mock provider) — no external AI/paid APIs.

Covers: output validity, strict validation (banned spam tags, dupes, length
caps), no-external-AI + auditable prompt, assembly prerequisite, staleness gate,
QC FAILED blocking (default) and override, QC WARNINGS tolerance, state
transitions (no auto-publish), persistence + restart recovery, regeneration
versioning (CURRENT -> DRAFT), assembly-change invalidation (STALE), user edits
with preserved AI original, API surface (generate/regenerate/status/result/
PATCH), immutability of the video, and legacy sync-path backward compatibility.
"""

import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.storage as storage_mod

TEST_STORAGE = Path(tempfile.mkdtemp(prefix="ai_shorts_metadata_test_"))
storage_mod.STORAGE_DIR = TEST_STORAGE

import pytest

from agents.image import ImageAgent
from agents.image_schemas import ImagePromptOutput
from agents.master import MasterAgent, InvalidTransitionError
from agents.metadata import MetadataAgent, build_user_message
from agents.metadata_schemas import (
    METADATA_SCHEMA,
    MetadataError,
    MetadataOutput,
    BANNED_HASHTAGS,
    validate_metadata_output,
)
from agents.video import VideoAgent
from agents.video_schemas import VideoPromptOutput
from backend.assembly import AssemblyService
from backend.config import Settings, get_settings
from backend.ffmpeg import get_ffmpeg_info
from backend.models import (
    AssemblyRecord,
    AssemblyStatus,
    MetadataStatus,
    ProjectStatus,
    MetadataRecord,
    QCReport,
    QCStatus,
    VideoGeneration,
)
from backend.providers.llm import _default_mock_metadata
from backend.providers.registry import get_image_provider
from backend.providers.video import MockVideoProvider
from backend.qc import QCService
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


@pytest.fixture(autouse=True)
def reload_settings():
    """Re-read env for every test so METADATA_* env tweaks don't leak."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


class CountingMetadataLLM:
    """Offline fake: records how it was called, returns the mock package."""

    def __init__(self):
        self.calls = 0
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return _default_mock_metadata()


def make_settings(tmp_path):
    return Settings(env={"STORAGE_PATH": str(tmp_path), "VIDEO_POLL_INTERVAL": "0.001"})


def qc_settings(tmp_path, extra=None):
    env = {"STORAGE_PATH": str(tmp_path), "VIDEO_POLL_INTERVAL": "0.001"}
    env.update(extra or {})
    return Settings(env=env)


def make_video_project(tmp_path, n=3, videos_generated=True):
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
    if source.startswith("testsrc"):
        graph = f"{source}=size={width}x{height}:rate={fps}:duration={duration}"
    else:
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
    proc = __import__("subprocess").run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:400])
    return Path(path)


def set_scene_clip(agent, index, path):
    scene = agent.project.scenes[index]
    gen = VideoGeneration(
        project_id=agent.project.id,
        scene_id=scene.id,
        provider="mock",
        model="mock-video-model",
        status="SUCCEEDED",
        prompt=VIDEO_PROMPT["prompt"],
        source_image_id=scene.selected_image_id or "A",
        source_image=scene.selected_image,
        duration_seconds=4,
        output_path=str(path),
    )
    scene.video_generations.append(gen)
    scene.current_video_id = gen.attempt_id
    scene.video_clip = gen.output_path
    save_project(agent.project)


def fabricate_project(tmp_path, final_path, *, n=1, clip_path=None, record=None):
    clip_path = clip_path or final_path
    agent = MasterAgent.create("Metadata fabricate")
    agent.start_planning()
    agent.submit_storyboard(
        storyboard={
            "title": "Meta",
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


def run_qc(agent, settings=None):
    qcs = QCService(settings=settings)
    assembly = agent.successful_assembly()
    assert assembly is not None
    agent.start_quality_check()
    report = QCReport(
        project_id=agent.project.id,
        assembly_id=assembly.assembly_id,
        status=QCStatus.RUNNING,
    )
    agent.qc_running(report)
    report = qcs.assemble_report(agent.project, assembly, report)
    agent.qc_complete(report)
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    return report


def run_metadata(agent, *, settings=None, llm=None, model="mock-model"):
    assembly = agent.successful_assembly()
    assert assembly is not None, "metadata needs a current final assembly"
    if agent.project.status != ProjectStatus.GENERATING_METADATA:
        agent.metadata_start()
    meta = MetadataAgent(llm=llm or CountingMetadataLLM(), settings=settings)
    output = meta.generate(agent.project, model=model)
    qc = agent.latest_qc_report()
    record = MetadataRecord(
        project_id=agent.project.id,
        assembly_id=assembly.assembly_id,
        qc_id=qc.qc_id if qc and qc.assembly_id == assembly.assembly_id else None,
        title=output.title,
        description=output.description,
        hashtags=list(output.hashtags),
        keywords=list(output.keywords),
        category=output.category or "",
        content_summary=output.content_summary or "",
        prompt=build_user_message(agent.project),
        provider=meta.provider_name(),
        model=model,
    )
    agent.metadata_running(record)
    agent.metadata_complete(record)
    return record, meta


def current(agent):
    return agent.current_metadata()


# =========================================================================
# 1-5. Output validity, mock generation, strict validation
# =========================================================================


def test_generated_package_is_valid(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    record, _ = run_metadata(agent)
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW

    assert record.title and len(record.title) <= 100
    assert "#" not in record.title
    assert record.description and len(record.description) <= 1000
    assert record.hashtags, "expected hashtags"
    for tag in record.hashtags:
        assert tag.startswith("#") and " " not in tag
    assert len(record.hashtags) == len({t.lower() for t in record.hashtags})
    assert "#Shorts" in record.hashtags
    assert record.keywords
    assert record.category
    assert record.content_summary


def test_generation_is_offline_and_auditable(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    llm = CountingMetadataLLM()
    record, meta = run_metadata(agent, llm=llm)

    assert llm.calls == 1, "exactly one LLM round-trip"
    kw = llm.last_kwargs
    assert kw["json_schema"]["name"] == "metadata_output"
    assert "YouTube Shorts upload package".lower() in "\n".join(
        [kw["system_prompt"], kw["user_message"]]
    ).lower()
    assert record.provider == "mock"
    assert record.model == "mock-model"

    prompt = record.prompt
    assert "Seed growing from seed to flowering plant" in prompt  # idea
    assert "subject: seed" in prompt  # Project Bible
    assert "scene 1: Scene 1 visual" in prompt  # scene summaries
    assert "final video" in prompt  # final probe info
    assert "QC" in prompt  # QC context


def test_generation_does_not_touch_video_or_world(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    record, _, _ = assemble_project(agent)
    run_qc(agent)
    final_bytes_before = Path(record.output_path).read_bytes()
    scenes_before = [s.model_dump() for s in agent.project.scenes]
    bible_before = dict(agent.project.project_bible or {})
    story_before = dict(agent.project.storyboard or {})

    run_metadata(agent)

    assert Path(record.output_path).read_bytes() == final_bytes_before
    saved = load_project(agent.project.id)
    assert [s.model_dump() for s in saved.scenes] == scenes_before
    assert dict(saved.project_bible or {}) == bible_before
    assert dict(saved.storyboard or {}) == story_before


def test_strict_rejects_banned_spam_hashtags():
    out = MetadataOutput(
        title="Cool clip", description="A clip about growth.",
        hashtags=["#viral", "#growth"],
    )
    issues = validate_metadata_output(out, strict=True)
    assert any("spam" in i for i in issues)
    issues_lax = validate_metadata_output(out, strict=False)
    assert not any("spam" in i for i in issues_lax)
    assert BANNED_HASHTAGS


def test_strict_rejects_duplicates_and_overcount(tmp_path):
    settings = qc_settings(tmp_path, {"METADATA_MAX_HASHTAGS": "3"})
    out = MetadataOutput(
        title="Ok title", description="Valid description body.",
        hashtags=["#a", "#a", "#Shorts", "#extra"],
    )
    issues = validate_metadata_output(out, settings, strict=True)
    assert any("duplicates" in i for i in issues)
    assert any("too many" in i for i in issues)

    tags = [f"#{i}" for i in range(3)]
    few = MetadataOutput(title="T", description="D", hashtags=tags)
    assert not validate_metadata_output(few, settings, strict=True)


def test_strict_enforces_length_caps(tmp_path):
    settings = qc_settings(tmp_path, {"METADATA_MAX_TITLE_LENGTH": "20",
                                      "METADATA_MAX_DESCRIPTION_LENGTH": "40"})
    long_title = MetadataOutput(
        title="X" * 30, description="desc body", hashtags=["#Shorts"],
    )
    issues = validate_metadata_output(long_title, settings, strict=True)
    assert any("title" in i for i in issues)

    stuffed = MetadataOutput(
        title="Fine", description=" ".join(["word"] * 6), hashtags=["#Shorts"],
    )
    issues = validate_metadata_output(stuffed, settings, strict=True)
    assert any("repeats" in i or "description" in i for i in issues)


def test_metadata_agent_rejects_strict_invalid_output(tmp_path):
    class BadLLM:
        def __call__(self, **kwargs):
            return MetadataOutput(title="Spam", description="d",
                                  hashtags=["#viral"]).model_dump()

    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    meta = MetadataAgent(llm=BadLLM(), settings=qc_settings(tmp_path))
    with pytest.raises(MetadataError):
        meta.generate(agent.project)


# =========================================================================
# 6-11. Prerequisites + QC gating
# =========================================================================


def test_api_requires_final_assembly():
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path := Path(tempfile.mkdtemp()), n=1,
                               videos_generated=False)
    client = TestClient(api_module.app)
    resp = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert resp.status_code == 409
    assert "assemble" in resp.json()["detail"].lower()


def test_api_refuses_stale_assembly(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    client = TestClient(api_module.app)
    ok = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert ok.status_code == 200

    agent.invalidate_assembly("a scene clip was regenerated")
    agent.project.status = ProjectStatus.READY_FOR_REVIEW
    save_project(agent.project)
    resp = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert resp.status_code == 409
    assert "reassemble" in resp.json()["detail"].lower()
    status = client.get(f"/projects/{agent.project.id}/metadata/status").json()
    assert status["stale"] is True
    assert status["can_generate"] is False


def test_qc_failed_blocks_generation_by_default(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    final = render_at(tmp_path / "black.mp4", source="color=c=black", duration=4)
    agent = fabricate_project(tmp_path, final)
    settings = qc_settings(tmp_path, {"QC_MIN_FILE_SIZE": "100"})
    report = run_qc(agent, settings=settings)
    assert report.status == QCStatus.FAILED

    client = TestClient(api_module.app)
    status = client.get(f"/projects/{agent.project.id}/metadata/status").json()
    assert status["qc_blocked"] is True
    assert status["can_generate"] is False
    resp = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert resp.status_code == 409
    assert "QC FAILED" in resp.json()["detail"]


def test_qc_failed_can_be_overridden(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    final = render_at(tmp_path / "black2.mp4", source="color=c=black", duration=4)
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent, settings=qc_settings(tmp_path, {"QC_MIN_FILE_SIZE": "100"}))
    assert report.status == QCStatus.FAILED

    monkeypatch.setenv("METADATA_BLOCK_ON_QC_FAILED", "false")
    get_settings.cache_clear()
    client = TestClient(api_module.app)
    resp = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qc_warning"] is True
    assert body["qc_overall"] == "FAILED"
    assert body["metadata"]["title"]


def test_qc_warnings_are_allowed_with_notice(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    final = render_at(tmp_path / "frozen.mp4", source="color=c=0x808080", duration=4)
    agent = fabricate_project(tmp_path, final)
    report = run_qc(agent, settings=qc_settings(tmp_path, {"QC_MIN_FILE_SIZE": "100"}))
    assert report.status == QCStatus.WARNINGS

    client = TestClient(api_module.app)
    resp = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qc_warning"] is True
    assert body["qc_overall"] == "WARNINGS"


def test_generation_allowed_without_qc(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    assert agent.latest_qc_report() is None
    record, _ = run_metadata(agent)
    assert record.qc_id is None
    assert record.status == MetadataStatus.CURRENT
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW


# =========================================================================
# 12-14. State transitions, persistence, restart recovery
# =========================================================================


def test_no_auto_publish_and_legacy_path_preserved(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    run_metadata(agent)
    assert agent.project.status == ProjectStatus.READY_FOR_REVIEW
    with pytest.raises(InvalidTransitionError):
        agent.final_approve()  # human still must approve manually

    # Legacy synchronous path still works end-to-end.
    agent.metadata_start()
    agent.metadata_generated("T", "D", ["#s"])
    assert agent.project.status == ProjectStatus.WAITING_FOR_FINAL_APPROVAL
    agent.final_approve()
    assert agent.project.status == ProjectStatus.COMPLETED


def test_metadata_persistence(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    record, _ = run_metadata(agent)

    reloaded = load_project(agent.project.id)
    assert reloaded.current_metadata_id == record.metadata_id
    assert [r.status for r in reloaded.metadata_records] == [MetadataStatus.CURRENT]
    assert reloaded.metadata_records[0].title == record.title
    assert reloaded.metadata_records[0].prompt
    assert reloaded.youtube_metadata.title == record.title
    assert reloaded.youtube_metadata.hashtags == record.hashtags
    assert reloaded.youtube_metadata.keywords == record.keywords
    assert reloaded.youtube_metadata.category == record.category


def test_restart_preserves_metadata(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    run_metadata(agent)

    reloaded = MasterAgent.load(agent.project.id)
    assert reloaded.project.status == ProjectStatus.READY_FOR_REVIEW
    record = reloaded.current_metadata()
    assert record is not None and record.status == MetadataStatus.CURRENT
    assert reloaded.project.current_metadata_id == record.metadata_id
    assert AssemblyService().is_stale(reloaded.project) is False


# =========================================================================
# 15-16. Regeneration versioning + assembly-change invalidation
# =========================================================================


def test_regeneration_keeps_history(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    first, _ = run_metadata(agent)

    second, _ = run_metadata(agent)  # regenerate for the same assembly
    assert second.metadata_id != first.metadata_id
    assert second.assembly_id == first.assembly_id

    reloaded = load_project(agent.project.id)
    assert reloaded.current_metadata_id == second.metadata_id
    by_id = {r.metadata_id: r.status for r in reloaded.metadata_records}
    assert by_id[second.metadata_id] == MetadataStatus.CURRENT
    assert by_id[first.metadata_id] == MetadataStatus.DRAFT  # history kept
    assert len(reloaded.metadata_records) == 2


def test_assembly_change_stales_metadata(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    record1, _, _ = assemble_project(agent)
    run_qc(agent)
    first, _ = run_metadata(agent)
    assert first.assembly_id == record1.assembly_id

    agent.invalidate_assembly("regen scene 1")
    reloaded = MasterAgent.load(agent.project.id)
    assert reloaded.current_metadata() is None
    assert reloaded.project.current_metadata_id is None
    assert reloaded.project.metadata_records[0].status == MetadataStatus.STALE
    assert "regen scene 1" in reloaded.project.metadata_records[0].raw.get("invalidated_reason", "")

    set_scene_clip(reloaded, 0, render_at(tmp_path / "new0.mp4", duration=3))
    record2, _, _ = assemble_project(reloaded)
    assert record2.assembly_id != record1.assembly_id

    second, _ = run_metadata(reloaded)
    assert second.assembly_id == record2.assembly_id
    by_id = {r.metadata_id: r.status for r in reloaded.project.metadata_records}
    assert by_id[second.metadata_id] == MetadataStatus.CURRENT
    assert by_id[first.metadata_id] == MetadataStatus.STALE


# =========================================================================
# 17-18. User edits (PATCH) with preserved AI original + API surface
# =========================================================================


def test_user_edits_preserve_ai_original(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    run_qc(agent)
    record, _ = run_metadata(agent)

    agent.update_metadata(
        record.metadata_id,
        title="My Edited Title",
        description="My own description.",
        hashtags=["#Mine", "#Shorts"],
        keywords=["edited"],
        category="People & Blogs",
        content_summary="Edited summary",
    )
    edited = current(agent)
    assert edited.title == "My Edited Title"
    assert edited.hashtags == ["#Mine", "#Shorts"]
    assert edited.is_edited is True
    assert edited.ai_original["title"] == record.title
    assert edited.ai_original["hashtags"] == record.hashtags
    assert agent.project.youtube_metadata.title == "My Edited Title"

    reloaded = load_project(agent.project.id)
    assert reloaded.youtube_metadata.title == "My Edited Title"
    rec = next(
        r for r in reloaded.metadata_records
        if r.metadata_id == agent.project.current_metadata_id
    )
    assert rec.ai_original["description"] == record.description


def test_patch_only_touches_provided_fields(tmp_path):
    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    record, _ = run_metadata(agent)
    agent.update_metadata(record.metadata_id, title="Only Title Changed")
    edited = current(agent)
    assert edited.title == "Only Title Changed"
    assert edited.description == record.description
    assert edited.hashtags == record.hashtags


def test_api_full_metadata_surface(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=2)
    assemble_project(agent)
    client = TestClient(api_module.app)
    client.post(f"/projects/{agent.project.id}/qc/run")

    generated = client.post(f"/projects/{agent.project.id}/metadata/generate")
    assert generated.status_code == 200
    body = generated.json()
    assert body["status"] == "READY_FOR_REVIEW"
    assert body["metadata"]["title"]
    assert body["qc_warning"] is False
    first_id = body["metadata"]["metadata_id"]

    status = client.get(f"/projects/{agent.project.id}/metadata/status").json()
    assert status["metadata_current"] is True
    assert status["can_generate"] is True
    assert status["stale"] is False
    assert status["metadata"]["metadata_id"] == first_id
    assert [r["metadata_id"] for r in status["metadata_records"]] == [first_id]

    result = client.get(f"/projects/{agent.project.id}/metadata/result").json()
    assert result["metadata_current"] is True
    assert result["metadata"]["metadata_id"] == first_id
    assert result["exists"] is True
    assert result["final_asset_url"].startswith("/assets/")
    assert client.get(result["final_asset_url"]).status_code == 200

    # Editing via PATCH (title + hashtags) keeps the AI original.
    patched = client.patch(
        f"/projects/{agent.project.id}/metadata/{first_id}",
        json={"title": "Edited via API", "hashtags": ["#Edited", "#Shorts"]},
    )
    assert patched.status_code == 200
    meta = patched.json()["metadata"]
    assert meta["title"] == "Edited via API"
    assert meta["is_edited"] is True
    assert meta["ai_original"]["title"] == body["metadata"]["title"]
    assert meta["ai_original"]["hashtags"] == body["metadata"]["hashtags"]

    # Regeneration appends a new CURRENT version; the old one becomes DRAFT.
    regenerated = client.post(f"/projects/{agent.project.id}/metadata/regenerate")
    assert regenerated.status_code == 200
    second_id = regenerated.json()["metadata"]["metadata_id"]
    assert second_id != first_id
    status = client.get(f"/projects/{agent.project.id}/metadata/status").json()
    by_id = {r["metadata_id"]: r["status"] for r in status["metadata_records"]}
    assert by_id[second_id] == "CURRENT"
    assert by_id[first_id] == "DRAFT"
    assert status["metadata"]["metadata_id"] == second_id

    # The final video is unchanged by generate/regenerate/PATCH.
    rec2 = load_project(agent.project.id)
    assert rec2.youtube_metadata.title == status["metadata"]["title"]


def test_api_patch_rejects_empty_title_and_caps(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=1)
    assemble_project(agent)
    client = TestClient(api_module.app)
    client.post(f"/projects/{agent.project.id}/metadata/generate")
    mid = client.get(f"/projects/{agent.project.id}/metadata/status").json()[
        "metadata"
    ]["metadata_id"]

    resp = client.patch(
        f"/projects/{agent.project.id}/metadata/{mid}", json={"title": "   "}
    )
    assert resp.status_code == 409
    resp = client.patch(
        f"/projects/{agent.project.id}/metadata/{mid}", json={"title": "T" * 500}
    )
    assert resp.status_code == 409


# =========================================================================
# 19-21. API gate states, immutability, mock end-to-end
# =========================================================================


def test_api_metadata_endpoint_requires_review_state(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    agent = make_video_project(tmp_path, n=1, videos_generated=False)
    client = TestClient(api_module.app)
    assert client.post(f"/projects/{agent.project.id}/metadata/generate").status_code == 409


def test_generate_edit_regenerate_do_not_change_video_bytes(tmp_path):
    agent = make_video_project(tmp_path, n=2)
    rec, _, _ = assemble_project(agent)
    run_qc(agent)
    before = Path(rec.output_path).read_bytes()

    first, _ = run_metadata(agent)
    agent.update_metadata(first.metadata_id, title="Edited title remains")
    second, _ = run_metadata(agent)
    assert Path(rec.output_path).read_bytes() == before
    assert second.metadata_id != first.metadata_id


def test_mock_end_to_end_metadata_workflow(tmp_path):
    from fastapi.testclient import TestClient

    import backend.api as api_module

    client = TestClient(api_module.app)
    pid = client.post(
        "/projects", json={"idea": "Seed growing from seed to flowering plant"}
    ).json()["id"]
    client.post(f"/projects/{pid}/planning")
    client.post(f"/projects/{pid}/plan")
    client.post(f"/projects/{pid}/story/approve")
    client.post(f"/projects/{pid}/images/generate")
    project = client.get(f"/projects/{pid}").json()
    selections = {s["id"]: "A" for s in project["scenes"]}
    client.post(f"/projects/{pid}/images/select", json={"selections": selections})
    client.post(f"/projects/{pid}/videos/generate")
    body = client.post(f"/projects/{pid}/assembly/assemble").json()
    assert body["status"] == "READY_FOR_REVIEW"

    # QC passes, then metadata generation (fully agent-driven, mock-only).
    qc = client.post(f"/projects/{pid}/qc/run").json()
    assert qc["overall"] == "PASSED"
    gen = client.post(f"/projects/{pid}/metadata/generate").json()
    assert gen["status"] == "READY_FOR_REVIEW"
    assert gen["qc_warning"] is False
    meta1 = gen["metadata"]
    assert meta1["title"] and "#" not in meta1["title"]
    assert all(t.startswith("#") for t in meta1["hashtags"])

    # Stable across restart; youtube_metadata working copy synced.
    project = client.get(f"/projects/{pid}").json()
    assert project["current_metadata_id"] == meta1["metadata_id"]
    assert project["youtube_metadata"]["title"] == meta1["title"]

    # Manual edit persists.
    patch = client.patch(
        f"/projects/{pid}/metadata/{meta1['metadata_id']}",
        json={"title": "Hand-tuned title", "hashtags": ["#Mine", "#Shorts"]},
    ).json()["metadata"]
    assert patch["title"] == "Hand-tuned title"
    assert patch["is_edited"] is True

    # Regeneration: new version, old one preserved as DRAFT history.
    meta2 = client.post(f"/projects/{pid}/metadata/regenerate").json()["metadata"]
    assert meta2["metadata_id"] != meta1["metadata_id"]
    status = client.get(f"/projects/{pid}/metadata/status").json()
    by_id = {r["metadata_id"]: r["status"] for r in status["metadata_records"]}
    assert by_id[meta2["metadata_id"]] == "CURRENT"
    assert by_id[meta1["metadata_id"]] == "DRAFT"

    # Regen a scene -> reassemble -> old metadata STALE for the new assembly.
    scene_id = project["scenes"][0]["id"]
    client.post(f"/projects/{pid}/videos/regenerate", json={"scene_id": scene_id})
    status = client.get(f"/projects/{pid}/metadata/status").json()
    assert status["stale"] is True
    assert client.post(f"/projects/{pid}/metadata/generate").status_code == 409
    body2 = client.post(f"/projects/{pid}/assembly/reassemble").json()
    assert body2["status"] == "READY_FOR_REVIEW"
    client.post(f"/projects/{pid}/qc/run")

    meta3 = client.post(f"/projects/{pid}/metadata/generate").json()["metadata"]
    assert meta3["assembly_id"] == body2["assembly"]["assembly_id"]
    assert meta3["metadata_id"] != meta1["metadata_id"] != meta2["metadata_id"]
    status = client.get(f"/projects/{pid}/metadata/status").json()
    by_id = {r["metadata_id"]: r["status"] for r in status["metadata_records"]}
    assert by_id[meta3["metadata_id"]] == "CURRENT"
    assert by_id[meta1["metadata_id"]] == "STALE"
    assert by_id[meta2["metadata_id"]] == "STALE"

    client.delete(f"/projects/{pid}")
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# No static mount: assets are served dynamically via /assets/{path} below.
from pydantic import BaseModel, Field

import backend.storage as storage_mod
from backend.assembly import AssemblyError
from backend.models import AssemblyStatus, ProjectSettings, ProjectStatus
from backend.storage import delete_project as storage_delete, list_projects
from backend.health import get_health
from agents.master import MasterAgent, InvalidTransitionError, ProjectNotFoundError
from agents.story import StoryAgent
from agents.story_schemas import ValidationError
from agents.video_schemas import VideoPromptError

app = FastAPI(title="AI Shorts Studio V1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated assets (per-project image/video files) so the frontend can
# display candidates directly. Resolved at request time so test suites that
# reassign STORAGE_DIR keep working (StaticFiles would pin the import-time dir).
@app.get("/assets/{asset_path:path}")
def get_asset(asset_path: str):
    from starlette.responses import FileResponse

    from backend.storage import STORAGE_DIR

    safe = asset_path.replace("..", "")
    candidate = (STORAGE_DIR / safe).resolve()
    if not str(candidate).startswith(str(STORAGE_DIR.resolve())):
        raise HTTPException(404, "Asset not found")
    if not candidate.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(candidate)


class ProjectCreate(BaseModel):
    name: str = ""
    idea: str
    content_type: str = "Timelapse"
    visual_style: str = "Photorealistic"
    duration_seconds: int = 30
    aspect_ratio: str = "9:16"


class SceneIn(BaseModel):
    description: str
    duration_seconds: int = 4


class StoryboardIn(BaseModel):
    storyboard: dict
    scenes: list[SceneIn]
    project_bible: Optional[dict] = None


class ImagePair(BaseModel):
    scene_id: str
    image_a: str
    image_b: str


class ImageSelection(BaseModel):
    selections: dict[str, str]


class ImageSelectionOne(BaseModel):
    scene_id: str
    candidate: str


class ImageRegenerate(BaseModel):
    scene_id: str
    candidate_id: str = Field(pattern="^[AB]$")


class VideoClips(BaseModel):
    clips: dict[str, str]


class VideoSceneRef(BaseModel):
    scene_id: str
    prompt: Optional[str] = Field(
        default=None,
        description="Optional animation prompt to reuse (instead of generating a new one).",
    )


class MetadataIn(BaseModel):
    title: str
    description: str
    hashtags: list[str]


class MetadataPatch(BaseModel):
    """User edits for a metadata record. Only supplied fields are changed."""

    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    category: Optional[str] = None
    content_summary: Optional[str] = None


class RejectIn(BaseModel):
    reason: str = ""


class FailIn(BaseModel):
    reason: str


def _load_or_404(project_id: str) -> MasterAgent:
    try:
        return MasterAgent.load(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


def get_image_agent():
    """Factory used by the image endpoints. Overridable in tests."""
    from agents.image import ImageAgent

    return ImageAgent()


def get_video_agent():
    """Factory used by the video endpoints. Overridable in tests."""
    from agents.video import VideoAgent

    return VideoAgent()


def get_assembly_service():
    """Factory used by the assembly endpoints. Overridable in tests."""
    from backend.assembly import AssemblyService

    return AssemblyService()


def get_qc_service():
    """Factory used by the QC endpoints. Overridable in tests."""
    from backend.qc import QCService

    return QCService()


def get_metadata_agent():
    """Factory used by the metadata endpoints. Overridable in tests."""
    from agents.metadata import MetadataAgent

    return MetadataAgent()


@app.get("/health")
def health():
    return get_health()


@app.get("/settings")
def settings():
    """Non-secret configuration summary for the frontend settings view."""
    from backend.config import get_settings
    from backend.health import (
        CONFIGURED,
        NOT_CONFIGURED,
        SUPPORTED_LLM_PROVIDERS,
        _status_for_provider,
    )

    s = get_settings()
    llm_status = _status_for_provider(s.llm_provider, s.llm_configured, SUPPORTED_LLM_PROVIDERS)
    return {
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "llm_configured": s.llm_configured,
        "llm_status": llm_status,
        "llm_credentialed": s.llm_credentialed,
        "image_provider": s.image_provider,
        "image_model": s.image_model,
        "image_configured": s.image_configured,
        "video_provider": s.video_provider,
        "video_model": s.video_model,
        "video_configured": s.video_configured,
        "database_url_set": bool(s.database_url),
    }


@app.get("/projects")
def get_projects():
    """Dashboard cards: each project plus its derived pipeline progress / step
    / next action / thumbnail (backed by the deterministic workflow engine)."""
    cards = []
    for entry in list_projects():
        card = _project_card(entry["id"])
        if card:
            cards.append(card)
    return cards


def _project_card(project_id: str) -> Optional[dict]:
    """Enrich one project with derived pipeline info for the dashboard card."""
    from backend.storage import load_project

    from backend.workflow import build_summary

    project = load_project(project_id)
    if project is None:
        return None
    summary = build_summary(project)
    return {
        "id": project.id,
        "name": project.name,
        "idea": project.idea,
        "status": project.status.value,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "settings": project.settings.model_dump(),
        "storyboard": project.storyboard,
        "progress": summary["progress"],
        "current_step": summary["current_step"],
        "next_action": summary["next_action"],
        "thumbnail": summary["thumbnail"],
        "stale": summary["stale"],
    }


@app.post("/projects")
def create_project(data: ProjectCreate):
    settings = ProjectSettings(
        content_type=data.content_type,
        visual_style=data.visual_style,
        duration_seconds=data.duration_seconds,
        aspect_ratio=data.aspect_ratio,
    )
    agent = MasterAgent.create(
        idea=data.idea,
        settings=settings,
        name=data.name,
    )
    return agent.project


@app.delete("/projects/{project_id}")
def delete_project(project_id: str):
    if not storage_delete(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True, "project_id": project_id}


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    return _load_or_404(project_id).project


@app.get("/projects/{project_id}/summary")
def project_summary(project_id: str):
    """Unified project summary for the workspace / final review / dashboard.

    Everything here is derived deterministically from persisted state — no LLM.
    Includes progress, per-stage statuses, the next action, final-video details,
    QC + metadata status, scene overview, and generation estimates. No secrets.
    """
    from backend.workflow import build_summary

    agent = _load_or_404(project_id)
    return build_summary(agent.project)


@app.get("/projects/{project_id}/next-action")
def project_next_action(project_id: str):
    """The single deterministic next production action for a project (no LLM)."""
    from backend.workflow import compute_next_action

    agent = _load_or_404(project_id)
    return compute_next_action(agent.project)


@app.post("/projects/{project_id}/planning")
def start_planning(project_id: str):
    agent = _load_or_404(project_id)
    try:
        return agent.start_planning()
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/plan")
def plan_story(project_id: str):
    """Run the Story Agent: PLANNING -> WAITING_FOR_STORY_APPROVAL."""
    agent = _load_or_404(project_id)
    try:
        return agent.plan_story()
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/projects/{project_id}/story/regenerate")
def regenerate_story(project_id: str, data: Optional[RejectIn] = None):
    """Regenerate the storyboard without creating images/videos."""
    agent = _load_or_404(project_id)
    try:
        return agent.regenerate_story(reason=(data.reason if data else ""))
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/projects/{project_id}/storyboard")
def submit_storyboard(project_id: str, data: StoryboardIn):
    agent = _load_or_404(project_id)
    try:
        scenes = [s.model_dump() for s in data.scenes]
        return agent.submit_storyboard(
            storyboard=data.storyboard,
            scenes=scenes,
            project_bible=data.project_bible,
        )
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/story/approve")
def approve_story(project_id: str):
    agent = _load_or_404(project_id)
    try:
        return agent.approve_story()
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/story/reject")
def reject_story(project_id: str, data: Optional[RejectIn] = None):
    agent = _load_or_404(project_id)
    try:
        return agent.reject_story(reason=(data.reason if data else ""))
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/images/generated")
def images_generated(project_id: str, data: list[ImagePair]):
    agent = _load_or_404(project_id)
    try:
        pairs = [p.model_dump() for p in data]
        return agent.images_generated(pairs)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/images/generate")
def generate_images(project_id: str):
    """Run the Image Agent across scenes that still need candidates.

    Scenes with a locked selection or existing successful candidates are not
    regenerated. Moves GENERATING_IMAGES -> WAITING_FOR_IMAGE_SELECTION when at
    least one scene is ready for the user to choose A or B. If every scene
    failed, the project stays put (retry by calling again).
    """
    from agents.image import ImageAgent

    agent = _load_or_404(project_id)
    project = agent.project
    if project.status not in (
        ProjectStatus.GENERATING_IMAGES,
        ProjectStatus.WAITING_FOR_IMAGE_SELECTION,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Image generation requires GENERATING_IMAGES, "
            f"current state is {project.status}",
        )
    summary = get_image_agent().run(project)
    if (
        summary["scenes_ready"] > 0
        and project.status == ProjectStatus.GENERATING_IMAGES
    ):
        try:
            agent.images_generated()
        except InvalidTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e))
    if summary["scenes_failed"]:
        agent._record_error(
            f"Image generation failed for {summary['scenes_failed']} scene(s)"
        )
    return project


@app.post("/projects/{project_id}/images/regenerate")
def regenerate_image(project_id: str, data: ImageRegenerate):
    """Regenerate a single candidate (A or B) for one scene, preserving the
    Project Bible, scene, continuity, and any existing selection."""
    from agents.image import ImagePromptError

    agent = _load_or_404(project_id)
    project = agent.project
    if project.status not in (
        ProjectStatus.GENERATING_IMAGES,
        ProjectStatus.WAITING_FOR_IMAGE_SELECTION,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Regeneration only available while selecting images, "
            f"current state is {project.status}",
        )
    try:
        get_image_agent().regenerate_candidate(
            project, data.scene_id, data.candidate_id
        )
    except ImagePromptError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return project


@app.post("/projects/{project_id}/images/select-one")
def select_one_image(project_id: str, data: ImageSelectionOne):
    """Lock one candidate (A or B) for a scene. No state transition."""
    agent = _load_or_404(project_id)
    try:
        return agent.select_image(data.scene_id, data.candidate)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/images/select")
def select_images(project_id: str, data: ImageSelection):
    agent = _load_or_404(project_id)
    try:
        return agent.select_images(data.selections)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/videos/generated")
def videos_generated(project_id: str, data: VideoClips):
    agent = _load_or_404(project_id)
    try:
        return agent.videos_generated(data.clips)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ---------------------------------------------------------------------------
# Video generation (Video Agent)
# ---------------------------------------------------------------------------

def _require_video_state(agent: MasterAgent) -> None:
    """Allow clip work while GENERATING_VIDEOS, and rework of a completed asset.

    After assembly a regenerated clip invalidates the final video (STALE), so
    the project returns to GENERATING_VIDEOS. A project that reached ASSEMBLING
    is locked from further clip changes (change via the Assembly screen).
    """
    if agent.project.status in (
        ProjectStatus.READY_FOR_REVIEW,
        ProjectStatus.ASSEMBLY_FAILED,
    ):
        agent.invalidate_assembly()
    if agent.project.status != ProjectStatus.GENERATING_VIDEOS:
        raise HTTPException(
            status_code=409,
            detail=(
                "Video generation is only available while the project is "
                f"GENERATING_VIDEOS; current state is {agent.project.status}"
            ),
        )


def _maybe_finish_videos(agent: MasterAgent) -> None:
    """Transition to ASSEMBLING only when every scene has a clip."""
    if all(s.video_clip for s in agent.project.scenes):
        agent.videos_generated()


@app.post("/projects/{project_id}/videos/generate")
def generate_videos(project_id: str):
    """Generate clips for every scene that does not yet have one (sync loop)."""
    agent = _load_or_404(project_id)
    _require_video_state(agent)
    project = agent.project
    summary = get_video_agent().run(project)
    if summary["scenes_failed"]:
        agent._record_error(
            f"Video generation failed for {summary['scenes_failed']} scene(s)"
        )
    if summary["scenes_ready"] == len(project.scenes) and not summary[
        "scenes_needs_selection"
    ]:
        try:
            _maybe_finish_videos(agent)
        except InvalidTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e))
    return project


@app.post("/projects/{project_id}/videos/generate-scene")
def generate_video_scene(project_id: str, data: VideoSceneRef):
    """Generate one scene's clip (from its locked image + existing prompt)."""
    agent = _load_or_404(project_id)
    _require_video_state(agent)
    project = agent.project
    try:
        status = get_video_agent().generate_scene_from_existing(project, data.scene_id)
    except VideoPromptError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        _maybe_finish_videos(agent)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"scene_id": data.scene_id, "status": status}


@app.post("/projects/{project_id}/videos/regenerate")
def regenerate_video_clip(project_id: str, data: VideoSceneRef):
    """Regenerate one scene's clip, reusing its existing animation prompt."""
    agent = _load_or_404(project_id)
    _require_video_state(agent)
    project = agent.project
    try:
        scene = get_video_agent().regenerate_clip(
            project, data.scene_id, prompt=data.prompt
        )
    except VideoPromptError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        _maybe_finish_videos(agent)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    status = scene.video_generations[-1].status if scene.video_generations else "PENDING"
    return {"scene_id": data.scene_id, "status": status}


@app.post("/projects/{project_id}/videos/regenerate-prompt")
def regenerate_video_prompt_and_clip(project_id: str, data: VideoSceneRef):
    """Regenerate the animation prompt via LLM, then the clip."""
    agent = _load_or_404(project_id)
    _require_video_state(agent)
    project = agent.project
    try:
        scene = get_video_agent().regenerate_prompt_clip(project, data.scene_id)
    except VideoPromptError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        _maybe_finish_videos(agent)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    status = scene.video_generations[-1].status if scene.video_generations else "PENDING"
    return {"scene_id": data.scene_id, "status": status}


@app.post("/projects/{project_id}/videos/retry")
def retry_video_scene(project_id: str, data: VideoSceneRef):
    """Retry a failed scene, reusing the failed generation's prompt."""
    agent = _load_or_404(project_id)
    _require_video_state(agent)
    project = agent.project
    try:
        scene = get_video_agent().retry_failed(
            project, data.scene_id, prompt=data.prompt
        )
    except VideoPromptError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        _maybe_finish_videos(agent)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    status = scene.video_generations[-1].status if scene.video_generations else "PENDING"
    return {"scene_id": data.scene_id, "status": status}


@app.get("/projects/{project_id}/videos/{scene_id}/status")
def video_scene_status(project_id: str, scene_id: str):
    """Latest video generation status/attempt details for one scene."""
    from backend.models import VideoGeneration

    agent = _load_or_404(project_id)
    scene = next((s for s in agent.project.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    gen = VideoGeneration.latest_for_scene(scene)
    if gen is None:
        return {"scene_id": scene_id, "status": "NONE", "attempts": 0}
    return {
        "scene_id": scene_id,
        "status": gen.status,
        "attempts": len(scene.video_generations),
        "error": gen.error,
        "provider": gen.provider,
        "job_id": gen.job_id,
        "duration_seconds": gen.duration_seconds,
        "output_path": gen.output_path,
    }


def _asset_url(path: Optional[str]) -> Optional[str]:
    """Absolute on-disk path -> servable /assets/<relative> URL."""
    if not path:
        return None
    if str(path).startswith("/assets/"):
        return str(path)
    parts = str(path).split(str(storage_mod.STORAGE_DIR), 1)
    if len(parts) != 2:
        return None
    return "/assets/" + parts[1].lstrip("/")


def _assembly_state_allowed(agent: MasterAgent) -> bool:
    return agent.project.status in (
        ProjectStatus.GENERATING_VIDEOS,
        ProjectStatus.ASSEMBLING,
        ProjectStatus.ASSEMBLY_FAILED,
        ProjectStatus.READY_FOR_REVIEW,
    )


def _run_assembly(agent: MasterAgent) -> dict:
    """Shared logic for /assembly/assemble and /assembly/reassemble."""
    if not _assembly_state_allowed(agent):
        raise HTTPException(
            status_code=409,
            detail=(
                "Assembly is only available while the project is in the "
                f"assembly phase; current state is {agent.project.status}"
            ),
        )
    svc = get_assembly_service()
    record, reused, error = svc.assemble(agent.project)

    if reused:
        agent.begin_assembly()
        agent.assembly_complete(record=record)
        return {
            "status": agent.project.status.value,
            "assembly": record.model_dump(),
            "reused": True,
            "error": None,
            "message": "Final video is already up to date.",
        }

    try:
        agent.begin_assembly()
        if error is not None:
            agent.assembly_failed(error, record=record)
            return {
                "status": agent.project.status.value,
                "assembly": record.model_dump(),
                "reused": False,
                "error": error,
                "message": "Assembly failed. Fix the issue and retry.",
            }
        agent.assembly_complete(record=record)
        return {
            "status": agent.project.status.value,
            "assembly": record.model_dump(),
            "reused": False,
            "error": None,
            "message": "Final video is ready.",
        }
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/assembly/assemble")
def assemble_project(project_id: str):
    """FFmpeg-only assembly of every scene clip into the final vertical MP4."""
    agent = _load_or_404(project_id)
    return _run_assembly(agent)


@app.post("/projects/{project_id}/assembly/reassemble")
def reassemble_project(project_id: str):
    """Reassemble after a regenerated scene made the final video stale."""
    agent = _load_or_404(project_id)
    return _run_assembly(agent)


@app.get("/projects/{project_id}/assembly/status")
def assembly_status(project_id: str):
    """Readiness + current/latest assembly summary for the Assembly screen."""
    from backend.ffmpeg import get_ffmpeg_info

    agent = _load_or_404(project_id)
    svc = get_assembly_service()
    readiness = svc.readiness(agent.project)
    latest = agent.latest_assembly()
    return {
        "project_id": project_id,
        "status": agent.project.status.value,
        "readiness": readiness,
        "assembly": latest.model_dump() if latest else None,
        "final_video": agent.project.final_video,
        "stale": svc.is_stale(agent.project),
        "ffmpeg_available": bool(get_ffmpeg_info()),
        "can_assemble": _assembly_state_allowed(agent),
    }


@app.get("/projects/{project_id}/assembly/result")
def assembly_result(project_id: str):
    """Final video file details + its on-disk path (served via /assets)."""
    agent = _load_or_404(project_id)
    svc = get_assembly_service()
    record = svc.live_record(agent.project) or agent.latest_assembly()
    exists = bool(record and record.output_path and Path(record.output_path).is_file())
    return {
        "project_id": project_id,
        "final_video": agent.project.final_video,
        "final_asset_url": _asset_url(agent.project.final_video),
        "assembly": record.model_dump() if record else None,
        "exists": exists,
    }


@app.post("/projects/{project_id}/assembly/complete")
def assembly_complete(project_id: str, final_video_path: str):
    agent = _load_or_404(project_id)
    try:
        return agent.assembly_complete(final_video_path)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/qc/run")
def run_qc(project_id: str):
    """Run automated QC against the CURRENT final assembly.

    Only reachable from READY_FOR_REVIEW (or QUALITY_CHECK, e.g. after a crash).
    Refuses (409) when the final video is stale or regenerated -- it must be
    reassembled before QC can check it. On completion the project returns to
    READY_FOR_REVIEW regardless of PASS/WARN/FAIL: QC reports, it never fixes
    or publishes.
    """
    from backend.qc import QCError

    from backend.models import QCReport, QCStatus

    agent = _load_or_404(project_id)
    if agent.project.status not in (
        ProjectStatus.READY_FOR_REVIEW,
        ProjectStatus.QUALITY_CHECK,
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "QC is only available after assembly, from READY_FOR_REVIEW; "
                f"current state is {agent.project.status}"
            ),
        )
    svc = get_qc_service()
    assembly = agent.successful_assembly()
    ok, reason = svc.can_run(agent.project, assembly)
    if not ok:
        raise HTTPException(status_code=409, detail=reason)

    if agent.project.status == ProjectStatus.READY_FOR_REVIEW:
        try:
            agent.start_quality_check()
        except InvalidTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e))

    report = QCReport(
        project_id=project_id,
        assembly_id=assembly.assembly_id,
        status=QCStatus.RUNNING,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    agent.qc_running(report)
    try:
        report = svc.assemble_report(agent.project, assembly, report)
        agent.qc_complete(report)
    except QCError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "status": agent.project.status.value,
        "overall": report.status.value,
        "summary": report.summary,
        "report": report.model_dump(),
    }


def _current_qc_report(agent: MasterAgent, assembly=None):
    """The latest terminal QC report tied to the current (successful) assembly."""
    assembly = assembly if assembly is not None else agent.successful_assembly()
    if assembly is None:
        return None
    for report in reversed(agent.project.qc_reports):
        if report.is_current and report.assembly_id == assembly.assembly_id:
            return report
    return None


@app.get("/projects/{project_id}/qc/status")
def qc_status(project_id: str):
    """Current QC state: staleness gate, latest report, and whether it is current."""
    agent = _load_or_404(project_id)
    assembly = agent.successful_assembly()
    stale = get_assembly_service().is_stale(agent.project)
    report = _current_qc_report(agent, assembly)
    allowed = agent.project.status in (
        ProjectStatus.READY_FOR_REVIEW,
        ProjectStatus.QUALITY_CHECK,
    )
    return {
        "project_id": project_id,
        "status": agent.project.status.value,
        "stale": stale,
        "assembly": assembly.model_dump() if assembly else None,
        "can_run": bool(allowed and not stale and assembly is not None),
        "qc": report.model_dump() if report else None,
        "qc_current": report is not None,
        "overall": report.status.value if report else None,
    }


@app.get("/projects/{project_id}/qc/result")
def qc_result(project_id: str):
    """Latest current QC report + the final video it checked."""
    agent = _load_or_404(project_id)
    assembly = agent.successful_assembly()
    report = _current_qc_report(agent, assembly)
    final = agent.project.final_video
    return {
        "project_id": project_id,
        "final_asset_url": _asset_url(final),
        "assembly": assembly.model_dump() if assembly else None,
        "report": report.model_dump() if report else None,
        "qc_current": report is not None,
        "exists": bool(final and Path(final).is_file()),
    }


@app.post("/projects/{project_id}/metadata/start")
def metadata_start(project_id: str):
    """READY_FOR_REVIEW -> GENERATING_METADATA (manual; then the agent flow)."""
    agent = _load_or_404(project_id)
    try:
        return agent.metadata_start()
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


def _metadata_assembly_or_409(agent: MasterAgent):
    """The current final assembly metadata can describe, or a 409."""
    assembly = agent.successful_assembly()
    stale = get_assembly_service().is_stale(agent.project)
    if assembly is None and not stale:
        raise HTTPException(
            status_code=409,
            detail="No final assembly exists yet - assemble the video first.",
        )
    if (
        stale
        or assembly is None
        or assembly.is_stale
        or agent.project.current_assembly_id != assembly.assembly_id
    ):
        raise HTTPException(
            status_code=409,
            detail="The final video is out of date - reassemble before generating metadata.",
        )
    return assembly


def _run_metadata_generation(agent: MasterAgent, project_id: str) -> dict:
    """Generate (or regenerate) the upload package for the current final video.

    Shared by /metadata/generate and /metadata/regenerate: both produce a new
    auditable version; older versions for the same assembly become DRAFT
    history. Never publishes anything.
    """
    from backend.config import get_settings
    from backend.models import MetadataRecord, MetadataStatus, QCStatus
    from agents.metadata import MetadataError, build_user_message as _build_msg

    assembly = _metadata_assembly_or_409(agent)
    settings = get_settings()
    qc_report = _current_qc_report(agent, assembly)
    qc_status = qc_report.status if qc_report else None
    if qc_status == QCStatus.FAILED and settings.metadata_block_on_qc_failed:
        raise HTTPException(
            status_code=409,
            detail="QC FAILED - fix the video and rerun QC before generating metadata.",
        )

    if agent.project.status == ProjectStatus.READY_FOR_REVIEW:
        agent.metadata_start()

    meta_agent = get_metadata_agent()
    try:
        output = meta_agent.generate(agent.project, model=settings.llm_model)
    except MetadataError as e:
        raise HTTPException(status_code=409, detail=str(e))

    record = MetadataRecord(
        project_id=project_id,
        assembly_id=assembly.assembly_id,
        qc_id=qc_report.qc_id if qc_report else None,
        title=output.title,
        description=output.description,
        hashtags=list(output.hashtags),
        keywords=list(output.keywords),
        category=output.category or "",
        content_summary=output.content_summary or "",
        prompt=_build_msg(agent.project),
        provider=meta_agent.provider_name(),
        model=settings.llm_model,
    )
    # Crash-safe: snapshot the DRAFT record before the completing transitions.
    agent.metadata_running(record)
    agent.metadata_complete(record)
    return {
        "project_id": project_id,
        "status": agent.project.status.value,
        "metadata": record.model_dump(),
        "qc_warning": qc_status not in (None, QCStatus.PASSED),
        "qc_overall": qc_status.value if qc_status else None,
    }


@app.post("/projects/{project_id}/metadata/generate")
@app.post("/projects/{project_id}/metadata/regenerate")
def metadata_generate(project_id: str):
    agent = _load_or_404(project_id)
    try:
        return _run_metadata_generation(agent, project_id)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/projects/{project_id}/metadata/status")
def metadata_status(project_id: str):
    """Metadata state: gate info, QC warnings, current version + full history."""
    from backend.config import get_settings
    from backend.models import QCStatus

    agent = _load_or_404(project_id)
    assembly = agent.successful_assembly()
    stale = get_assembly_service().is_stale(agent.project)
    record = agent.current_metadata()
    qc_report = _current_qc_report(agent, assembly)
    settings = get_settings()
    qc_status = qc_report.status if qc_report else None
    blocked = settings.metadata_block_on_qc_failed and qc_status == QCStatus.FAILED
    allowed = agent.project.status in (
        ProjectStatus.READY_FOR_REVIEW,
        ProjectStatus.GENERATING_METADATA,
    )
    return {
        "project_id": project_id,
        "status": agent.project.status.value,
        "stale": stale,
        "assembly": assembly.model_dump() if assembly else None,
        "can_generate": bool(
            allowed and not stale and assembly is not None and not blocked
        ),
        "qc_blocked": blocked,
        "qc_warning": qc_status in (QCStatus.WARNINGS, QCStatus.FAILED),
        "qc_overall": qc_status.value if qc_status else None,
        "metadata": record.model_dump() if record else None,
        "metadata_current": record is not None,
        "metadata_records": [r.model_dump() for r in agent.project.metadata_records],
    }


@app.get("/projects/{project_id}/metadata/result")
def metadata_result(project_id: str):
    """The current upload package + the final video it describes."""
    from backend.models import QCStatus

    agent = _load_or_404(project_id)
    assembly = agent.successful_assembly()
    record = agent.current_metadata()
    qc_report = _current_qc_report(agent, assembly)
    final = agent.project.final_video
    return {
        "project_id": project_id,
        "final_asset_url": _asset_url(final),
        "assembly": assembly.model_dump() if assembly else None,
        "metadata": record.model_dump() if record else None,
        "metadata_current": record is not None,
        "qc_overall": qc_report.status.value if qc_report else None,
        "exists": bool(final and Path(final).is_file()),
    }


@app.patch("/projects/{project_id}/metadata/{metadata_id}")
def metadata_patch(project_id: str, metadata_id: str, data: MetadataPatch):
    """Edit user-facing fields of a metadata record (no state transition).

    The AI-generated original is preserved in ``ai_original`` on the first edit
    so the audit trail keeps the untouched draft. Only structural sanity is
    enforced here (non-empty, length caps) — the user stays in control.
    """
    from backend.config import get_settings

    agent = _load_or_404(project_id)
    fields = {k: v for k, v in data.model_dump().items() if v is not None}
    settings = get_settings()
    title = fields.get("title")
    if title is not None and not str(title).strip():
        raise HTTPException(status_code=409, detail="title cannot be empty")
    if title is not None and len(str(title)) > settings.metadata_max_title_length:
        raise HTTPException(
            status_code=409,
            detail=f"title is too long (max {settings.metadata_max_title_length} chars)",
        )
    description = fields.get("description")
    if description is not None and not str(description).strip():
        raise HTTPException(status_code=409, detail="description cannot be empty")
    if description is not None and len(str(description)) > settings.metadata_max_description_length:
        raise HTTPException(
            status_code=409,
            detail=f"description is too long (max {settings.metadata_max_description_length} chars)",
        )
    try:
        agent.update_metadata(metadata_id, **fields)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    record = agent.current_metadata()
    return {
        "project_id": project_id,
        "metadata": record.model_dump() if record else None,
    }


@app.post("/projects/{project_id}/metadata")
def generate_metadata(project_id: str, data: MetadataIn):
    """Legacy synchronous metadata entry (backward compatibility)."""
    agent = _load_or_404(project_id)
    try:
        return agent.metadata_generated(
            title=data.title,
            description=data.description,
            hashtags=data.hashtags,
        )
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/final/approve")
def final_approve(project_id: str):
    agent = _load_or_404(project_id)
    try:
        return agent.final_approve()
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/final/reject")
def final_reject(project_id: str, data: Optional[RejectIn] = None):
    agent = _load_or_404(project_id)
    try:
        return agent.final_reject(reason=(data.reason if data else ""))
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/fail")
def fail(project_id: str, data: FailIn):
    agent = _load_or_404(project_id)
    try:
        return agent.fail(data.reason)
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/projects/{project_id}/cancel")
def cancel(project_id: str):
    agent = _load_or_404(project_id)
    try:
        return agent.cancel()
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))

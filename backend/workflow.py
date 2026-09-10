"""Deterministic production-workflow engine for the unified project workspace.

Everything in this module is a pure function over the persisted ``Project``
state — there is NO LLM call and no new AI agent here. The "Next Action" and
the pipeline progress/summary are derived from the state that the existing
Story / Image / Video / Assembly / QC / Metadata agents already persist through
the Master Agent.

Stage order (the unified dashboard pipeline):
    1. STORY
    2. IMAGES
    3. VIDEOS
    4. ASSEMBLY
    5. QC
    6. METADATA
    7. FINAL_REVIEW
    8. READY_TO_UPLOAD

Stage statuses used by the UI:
    completed  — the stage milestone has been reached (green check)
    current    — the stage is where the human/AI work is happening now
    waiting    — not reachable yet
    failed     — the stage hit a blocking problem (e.g. assembly failed)
    stale      — completed work was invalidated by a later change
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models import Project, ProjectStatus, QCStatus

STAGE_ORDER = [
    "story",
    "images",
    "videos",
    "assembly",
    "qc",
    "metadata",
    "final_review",
    "ready",
]

STAGE_LABELS = {
    "story": "Story",
    "images": "Images",
    "videos": "Videos",
    "assembly": "Assembly",
    "qc": "QC",
    "metadata": "Metadata",
    "final_review": "Final Review",
    "ready": "Ready to Upload",
}

_DONE = ("completed", "failed")  # milestones that count toward progress


def _successful_assembly(project: Project):
    """The live (non-stale) SUCCEEDED assembly — the on-disk final video."""
    for record in reversed(project.assemblies):
        if record.succeeded:
            return record
    return None


def _current_qc(project: Project, assembly):
    """The terminal QC report tied to the supplied (current) assembly."""
    if assembly is None:
        return None
    for report in reversed(project.qc_reports):
        if report.is_current and report.assembly_id == assembly.assembly_id:
            return report
    return None


def _current_metadata(project: Project):
    """The CURRENT metadata record, or None."""
    current_id = project.current_metadata_id
    if not current_id:
        return None
    for record in reversed(project.metadata_records):
        if (
            record.metadata_id == current_id
            and record.status.value == "CURRENT"
        ):
            return record
    for record in reversed(project.metadata_records):
        if record.metadata_id == current_id:
            return record
    return None


def _isolated_svc():
    from backend.assembly import AssemblyService

    return AssemblyService()


def _asset_url(path: Optional[str]) -> Optional[str]:
    """Absolute on-disk path -> servable /assets/<relative> URL."""
    if not path:
        return None
    if str(path).startswith("/assets/"):
        return str(path)
    from backend.storage import STORAGE_DIR

    parts = str(path).split(str(STORAGE_DIR), 1)
    if len(parts) != 2:
        return None
    return "/assets/" + parts[1].lstrip("/")


# ---------------------------------------------------------------------------
# Stage status derivation
# ---------------------------------------------------------------------------


def stage_statuses(project: Project) -> List[Dict[str, Any]]:
    """Compute status/detail for every pipeline stage (in pipeline order)."""
    svc = _isolated_svc()
    stale = svc.is_stale(project)
    assembly = _successful_assembly(project)
    qc_report = _current_qc(project, assembly)
    metadata = _current_metadata(project)

    statuses: Dict[str, Dict[str, Any]] = {}

    # --- STORY ------------------------------------------------------------
    if not project.storyboard or not project.scenes:
        if project.status in (ProjectStatus.DRAFT, ProjectStatus.PLANNING):
            statuses["story"] = _stage(
                "current", "The storyboard has not been generated yet."
            )
        elif project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL:
            statuses["story"] = _stage(
                "current", "The storyboard is waiting for your approval."
            )
        else:
            statuses["story"] = _stage("waiting")
    elif project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL:
        statuses["story"] = _stage(
            "current", "The storyboard is waiting for your approval."
        )
    elif project.status in (ProjectStatus.DRAFT, ProjectStatus.PLANNING):
        statuses["story"] = _stage(
            "current", "The storyboard was rejected — regenerate it."
        )
    else:
        statuses["story"] = _stage("completed", f"Approved · {len(project.scenes)} scenes", done=True)

    story_done = statuses["story"]["done"]

    # --- IMAGES -----------------------------------------------------------
    if not story_done:
        statuses["images"] = _stage("waiting")
    elif project.status == ProjectStatus.GENERATING_IMAGES:
        statuses["images"] = _stage("current", "Generate two candidates for every scene.")
    elif project.status == ProjectStatus.WAITING_FOR_IMAGE_SELECTION:
        statuses["images"] = _stage("current", "Pick A or B for every scene.")
    elif _all_selected(project):
        statuses["images"] = _stage(
            "completed", f"{len(project.scenes)} images locked", done=True
        )
    else:
        statuses["images"] = _stage("current", "Not every scene has a locked image.")

    images_done = statuses["images"]["done"]

    # --- VIDEOS -----------------------------------------------------------
    if not images_done:
        statuses["videos"] = _stage("waiting")
    elif project.status == ProjectStatus.GENERATING_VIDEOS:
        if not _all_clips(project):
            statuses["videos"] = _stage(
                "current",
                _clip_progress(project) + " — generate the remaining clips.",
            )
        else:
            statuses["videos"] = _stage(
                "completed", f"{len(project.scenes)} clips ready", done=True
            )
    elif _all_clips(project):
        statuses["videos"] = _stage(
            "completed", f"{len(project.scenes)} clips ready", done=True
        )
    else:
        statuses["videos"] = _stage("current", _clip_progress(project))

    videos_done = statuses["videos"]["done"]

    # --- ASSEMBLY ---------------------------------------------------------
    record = svc.live_record(project)
    if not videos_done:
        if project.status == ProjectStatus.ASSEMBLING:
            statuses["assembly"] = _stage("current", "Assembling the final video…")
        else:
            statuses["assembly"] = _stage("waiting")
    elif record is None:
        if project.status == ProjectStatus.ASSEMBLY_FAILED and project.assemblies:
            statuses["assembly"] = _stage(
                "failed", "The final video assembly failed — retry it."
            )
        elif project.status == ProjectStatus.ASSEMBLING:
            statuses["assembly"] = _stage("current", "Assembling the final video…")
        else:
            statuses["assembly"] = _stage("current", "Assemble the final video.")
    elif record.is_stale or stale:
        statuses["assembly"] = _stage(
            "stale",
            record.stale_reason
            or "A scene clip changed — the final video is out of date.",
        )
    elif record.succeeded:
        statuses["assembly"] = _stage(
            "completed", _assembly_metrics(record), done=True
        )
    elif project.status == ProjectStatus.ASSEMBLY_FAILED:
        statuses["assembly"] = _stage(
            "failed", "The final video assembly failed — retry it."
        )
    else:
        statuses["assembly"] = _stage("current", "Assemble the final video.")

    assembly_done = statuses["assembly"]["done"]

    # --- QC -----------------------------------------------------------------
    if not assembly_done and not (record is not None and (record.is_stale or stale)):
        statuses["qc"] = _stage("waiting")
    elif stale:
        statuses["qc"] = _stage(
            "stale", "The final video changed — rerun QC after reassembling."
        )
    elif qc_report is None:
        statuses["qc"] = _stage("current", "Run the automated quality check.")
    elif qc_report.status in (QCStatus.PASSED, QCStatus.WARNINGS):
        statuses["qc"] = _stage(
            "completed",
            f"QC {qc_report.status.value} · {_quantity(qc_report.checks, 'check')}",
            done=True,
        )
    else:
        statuses["qc"] = _stage(
            "failed",
            "QC found problems with the current final video. Review the findings.",
        )

    qc_has_report = qc_report is not None
    qc_done = statuses["qc"]["done"]

    # --- METADATA ---------------------------------------------------------
    metadata_current = bool(
        metadata
        and metadata.status.value == "CURRENT"
        and assembly is not None
        and metadata.assembly_id == assembly.assembly_id
    )
    if stale:
        statuses["metadata"] = _stage(
            "stale", "The final video changed — regenerate the package."
        )
    elif not assembly_done:
        statuses["metadata"] = _stage("waiting")
    elif metadata_current:
        statuses["metadata"] = _stage(
            "completed", "Upload package is current for the final video.", done=True
        )
    else:
        statuses["metadata"] = _stage(
            "current", "Generate the upload package for the final video."
        )

    metadata_done = statuses["metadata"]["done"]

    # --- FINAL REVIEW (human checkpoint) --------------------------------
    ready = _ready_to_upload(
        project, stale, assembly, qc_report, metadata, metadata_current
    )
    if not (story_done and images_done and videos_done and assembly_done and qc_done and metadata_done):
        statuses["final_review"] = _stage("waiting")
    elif ready:
        statuses["final_review"] = _stage(
            "completed", "Video, QC, and package all current.", done=True
        )
    else:
        statuses["final_review"] = _stage(
            "current", "Review the final video, QC findings, and upload package."
        )

    # --- READY TO UPLOAD --------------------------------------------------
    if ready:
        statuses["ready"] = _stage(
            "completed",
            "Your video and metadata are ready. Upload them manually to YouTube.",
            done=True,
        )
    elif statuses["final_review"]["done"]:
        statuses["ready"] = _stage("current")
    else:
        statuses["ready"] = _stage("waiting")

    return [
        {
            "key": key,
            "label": STAGE_LABELS[key],
            "status": statuses[key]["status"],
            "detail": statuses[key]["detail"],
            "done": statuses[key]["done"],
        }
        for key in STAGE_ORDER
    ]


def _stage(status: str, detail: str = "", done: bool = False) -> Dict[str, Any]:
    return {"status": status, "detail": detail, "done": done}


def _all_selected(project: Project) -> bool:
    return bool(project.scenes) and all(s.selected_image for s in project.scenes)


def _all_clips(project: Project) -> bool:
    return bool(project.scenes) and all(s.video_clip for s in project.scenes)


def _clip_progress(project: Project) -> str:
    ready = sum(1 for s in project.scenes if s.video_clip)
    return f"{ready} / {len(project.scenes)} clips ready"


def _assembly_metrics(record) -> str:
    size_kb = record.output_size_bytes / 1024 if record.output_size_bytes else 0
    if size_kb > 1024:
        size = f"{size_kb / 1024:.1f} MB"
    else:
        size = f"{size_kb:.0f} KB"
    duration = f"{record.output_duration:.0f}s" if record.output_duration else "—"
    res = f"{record.output_width}x{record.output_height}" if record.output_width else "—"
    return f"final video ready · {duration} · {res} · {size}"


def _quantity(items, noun: str) -> str:
    return f"{len(items)} {noun}{'' if len(items) == 1 else 's'}"


# ---------------------------------------------------------------------------
# Derived "ready to upload" + next-action
# ---------------------------------------------------------------------------


def _ready_to_upload(project, stale, assembly, qc_report, metadata, metadata_current) -> bool:
    """Derived, never stored: everything needed for manual YouTube upload exists
    and is current. It never uploads/publishes anything by itself."""
    if stale or assembly is None:
        return False
    if not (project.final_video and Path(project.final_video).is_file()):
        return False
    if qc_report is None or qc_report.status == QCStatus.FAILED:
        return False
    if not metadata_current or metadata is None:
        return False
    return True


def _image_estimate(project: Project) -> Dict[str, Any]:
    """Scenes that still need image candidates, and how many."""
    target = 0
    total = 0
    for scene in project.scenes:
        if scene.selected_image:
            continue
        has_a = any(c.candidate_id == "A" and c.succeeded for c in scene.image_candidates)
        has_b = any(c.candidate_id == "B" and c.succeeded for c in scene.image_candidates)
        missing = (0 if has_a else 1) + (0 if has_b else 1)
        if missing:
            target += 1
            total += missing
    return {
        "scenes": target,
        "candidates_per_scene": 2,
        "total_candidates": total,
    }


def _video_estimate(project: Project) -> Dict[str, Any]:
    needed = [s for s in project.scenes if s.selected_image and not s.video_clip]
    return {
        "scenes_to_generate": len(needed),
        "estimated_clips": len(needed),
        "existing_clips": sum(1 for s in project.scenes if s.video_clip),
        "total_scenes": len(project.scenes),
    }


def compute_next_action(project: Project, stages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The deterministic next production action for the project (no LLM)."""
    stages = stages if stages is not None else stage_statuses(project)
    s = {st["key"]: st["status"] for st in stages}
    svc = _isolated_svc()
    stale = svc.is_stale(project)
    assembly = _successful_assembly(project)
    qc_report = _current_qc(project, assembly)
    metadata = _current_metadata(project)
    metadata_current = bool(
        metadata
        and metadata.status.value == "CURRENT"
        and assembly is not None
        and metadata.assembly_id == assembly.assembly_id
    )

    def act(action, label, step, reason, blocking=False, go=None, operation=None):
        return {
            "action": action,
            "label": label,
            "step": step,
            "reason": reason,
            "blocking": blocking,
            "go": go,
            "operation": operation,
        }

    if project.status == ProjectStatus.CANCELLED:
        return act(
            "CANCELLED",
            "Project is cancelled",
            None,
            "This production was cancelled. Start a new one when you are ready.",
            blocking=True,
        )
    if project.status == ProjectStatus.FAILED:
        return act(
            "RECOVER",
            "Project failed",
            None,
            "The project hit a problem. Review the error log and retry, or start fresh.",
            blocking=True,
        )
    if project.status == ProjectStatus.COMPLETED:
        return act(
            "READY_TO_UPLOAD",
            "Ready to Upload",
            "final_review",
            "Your video and metadata are ready. Upload them manually to YouTube.",
            go="final",
        )

    # --- STORY ------------------------------------------------------------
    if s["story"] in ("current",):
        if project.status == ProjectStatus.DRAFT:
            return act(
                "START_PRODUCTION",
                "Start Production",
                "story",
                "Generate the storyboard from your idea.",
                operation="story",
            )
        if project.status == ProjectStatus.PLANNING:
            return act(
                "GENERATE_STORY",
                "Generate Story",
                "story",
                "Generate the storyboard from your idea.",
                operation="story",
            )
        return act(
            "REVIEW_STORY",
            "Review Story",
            "story",
            "The storyboard is waiting for your approval.",
            blocking=True,
            go="",
        )

    # --- IMAGES -----------------------------------------------------------
    if s["images"] in ("current",):
        est = _image_estimate(project)
        if est["scenes"]:
            return act(
                "GENERATE_IMAGES",
                f"Generate Images ({est['total_candidates']} candidates across {est['scenes']} scenes)",
                "images",
                "Generate two candidates for every scene that does not have them yet.",
                operation="images",
            )
        return act(
            "SELECT_IMAGES",
            "Select Images",
            "images",
            "Every scene has both candidates — pick A or B on the image screen.",
            blocking=True,
            go="images",
        )

    # --- VIDEOS -----------------------------------------------------------
    if s["videos"] in ("current",):
        est = _video_estimate(project)
        if est["scenes_to_generate"] == est["total_scenes"] and not est["existing_clips"]:
            return act(
                "GENERATE_VIDEOS",
                f"Generate Videos ({est['estimated_clips']} clips)",
                "videos",
                "Animate every locked image into a short clip.",
                operation="videos",
            )
        if est["scenes_to_generate"]:
            return act(
                "CONTINUE_VIDEOS",
                f"Continue Video Generation ({est['scenes_to_generate']} / {est['total_scenes']} clips remaining)",
                "videos",
                "Some scenes still need a clip.",
                operation="videos",
            )
        return act(
            "ASSEMBLE_VIDEO",
            "Assemble Video",
            "assembly",
            "Every scene has a clip — stitch them into the final vertical MP4.",
        )

    # --- ASSEMBLY ---------------------------------------------------------
    if s["assembly"] in ("stale",):
        return act(
            "REASSEMBLE_VIDEO",
            "Reassemble Video",
            "assembly",
            "A scene clip changed, so the final video is out of date. Reassemble it, then rerun QC and regenerate metadata.",
        )
    if s["assembly"] == "failed":
        return act(
            "REASSEMBLE_VIDEO",
            "Retry Assembly",
            "assembly",
            "The final video assembly failed. Retry — your scene clips are safe.",
        )
    if s["assembly"] in ("current", "waiting"):
        return act(
            "ASSEMBLE_VIDEO",
            "Assemble Video",
            "assembly",
            "Stitch the scene clips into the final MP4.",
        )

    # --- QC ---------------------------------------------------------------
    if s["qc"] in ("current",):
        return act(
            "RUN_QC",
            "Run Quality Check",
            "qc",
            "Automated check of the final video: duration, resolution, scenes, and integrity.",
        )
    if s["qc"] == "failed":
        return act(
            "REVIEW_FINAL",
            "Review Final Video",
            "final_review",
            "QC found problems with the current final video. Review the findings before uploading.",
            blocking=True,
            go="final",
        )

    # --- METADATA ---------------------------------------------------------
    if s["metadata"] == "stale":
        return act(
            "GENERATE_METADATA",
            "Generate Metadata",
            "metadata",
            "The final video changed — regenerate the upload package for the new video.",
            operation="metadata",
        )
    if s["metadata"] in ("current",):
        return act(
            "GENERATE_METADATA",
            "Generate Metadata",
            "metadata",
            "Generate the title, description, hashtags, and keywords for the final video.",
            operation="metadata",
        )

    # --- FINAL REVIEW / READY -------------------------------------------
    ready = _ready_to_upload(
        project, stale, assembly, qc_report, metadata, metadata_current
    )
    if ready:
        return act(
            "READY_TO_UPLOAD",
            "Ready to Upload",
            "final_review",
            "Your video and metadata are ready. Upload them manually to YouTube.",
            go="final",
        )
    return act(
        "REVIEW_FINAL",
        "Review Final Video",
        "final_review",
        "Review the final video and upload package before uploading.",
        blocking=True,
        go="final",
    )


# ---------------------------------------------------------------------------
# Summary (used by the dashboard cards + the project workspace/final review)
# ---------------------------------------------------------------------------


def build_summary(project: Project) -> Dict[str, Any]:
    """The full workspace summary for one project (id included for routing)."""
    stages = stage_statuses(project)
    status_map = {st["key"]: st["status"] for st in stages}
    done_count = sum(1 for st in stages if st["done"])
    percent = round(done_count * 100 / len(STAGE_ORDER)) if STAGE_ORDER else 0
    next_action = compute_next_action(project, stages)

    svc = _isolated_svc()
    stale = svc.is_stale(project)
    assembly = _successful_assembly(project)
    final_record = svc.live_record(project)
    qc_report = _current_qc(project, assembly)
    metadata = _current_metadata(project)
    metadata_current = bool(
        metadata
        and metadata.status.value == "CURRENT"
        and assembly is not None
        and metadata.assembly_id == assembly.assembly_id
    )

    final_video = None
    if project.final_video and Path(project.final_video).is_file():
        final_video = {
            "exists": True,
            "url": _asset_url(project.final_video),
            "duration": (final_record.output_duration if final_record else 0) or 0,
            "width": final_record.output_width if final_record else 0,
            "height": final_record.output_height if final_record else 0,
            "size_bytes": final_record.output_size_bytes if final_record else 0,
            "scene_count": final_record.scene_count if final_record else len(project.scenes),
        }
    elif project.final_video:
        final_video = {"exists": False, "url": None, "duration": 0,
                       "width": 0, "height": 0, "size_bytes": 0,
                       "scene_count": 0}

    top_scene = next(
        (s for s in project.scenes if s.selected_image and s.selected_image), None
    )
    thumbnail = (
        _asset_url(top_scene.selected_image)
        if top_scene and top_scene.selected_image
        else (final_video["url"] if final_video and final_video["exists"] else None)
    )

    scenes = []
    for i, scene in enumerate(project.scenes, start=1):
        attempts = [g.status for g in scene.video_generations]
        latest_video_status = attempts[-1] if attempts else None
        scenes.append({
            "number": i,
            "id": scene.id,
            "description": scene.description,
            "duration_seconds": scene.duration_seconds,
            "image_a": bool(
                any(c.candidate_id == "A" and c.succeeded for c in scene.image_candidates)
            ),
            "image_b": bool(
                any(c.candidate_id == "B" and c.succeeded for c in scene.image_candidates)
            ),
            "selected_image": bool(scene.selected_image),
            "selected_image_url": _asset_url(scene.selected_image),
            "has_clip": bool(scene.video_clip),
            "clip_url": _asset_url(scene.video_clip),
            "latest_video_status": latest_video_status,
        })

    qc_summary = None
    if qc_report is not None:
        qc_summary = {
            "status": qc_report.status.value,
            "current": True,
            "summary": qc_report.summary,
            "check_count": len(qc_report.checks),
            "finding_count": len(qc_report.findings),
        }
    elif assembly is not None:
        qc_summary = {"status": "PENDING", "current": False, "summary": "",
                      "check_count": 0, "finding_count": 0}

    metadata_summary = None
    if metadata_current and metadata is not None:
        metadata_summary = {
            "current": True,
            "metadata_id": metadata.metadata_id,
            "title": metadata.title,
            "description": metadata.description,
            "hashtags": list(metadata.hashtags),
            "keywords": list(metadata.keywords),
            "category": metadata.category,
            "content_summary": metadata.content_summary,
            "is_edited": metadata.is_edited,
        }

    return {
        "project_id": project.id,
        "name": project.name,
        "idea": project.idea,
        "status": project.status.value,
        "updated_at": project.updated_at,
        "current_step": next_action["label"],
        "progress": {"percent": percent, "done": done_count, "total": len(STAGE_ORDER)},
        "stages": stages,
        "next_action": next_action,
        "final_video": final_video,
        "qc": qc_summary,
        "metadata": metadata_summary,
        "scenes": scenes,
        "thumbnail": thumbnail,
        "image_generation_estimate": _image_estimate(project),
        "video_generation_estimate": _video_estimate(project),
        "stale": stale,
    }
"""Master Agent — orchestrates the full AI Shorts production pipeline."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.models import (
    AssemblyRecord,
    AssemblyStatus,
    Project,
    ProjectSettings,
    ProjectStatus,
    MetadataRecord,
    MetadataStatus,
    QCReport,
    QCResult,
    QCStatus,
    Scene,
    YouTubeMetadata,
)
from backend.storage import load_project, save_project

# Allowed transitions: current state -> set of valid next states
ALLOWED_TRANSITIONS: Dict[ProjectStatus, set[ProjectStatus]] = {
    ProjectStatus.DRAFT: {ProjectStatus.PLANNING, ProjectStatus.CANCELLED},
    ProjectStatus.PLANNING: {
        ProjectStatus.WAITING_FOR_STORY_APPROVAL,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.WAITING_FOR_STORY_APPROVAL: {
        ProjectStatus.GENERATING_IMAGES,
        ProjectStatus.PLANNING,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.GENERATING_IMAGES: {
        ProjectStatus.WAITING_FOR_IMAGE_SELECTION,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.WAITING_FOR_IMAGE_SELECTION: {
        ProjectStatus.GENERATING_VIDEOS,
        ProjectStatus.GENERATING_IMAGES,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.GENERATING_VIDEOS: {
        ProjectStatus.ASSEMBLING,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.ASSEMBLING: {
        ProjectStatus.READY_FOR_REVIEW,
        ProjectStatus.ASSEMBLY_FAILED,
        ProjectStatus.QUALITY_CHECK,
        ProjectStatus.GENERATING_VIDEOS,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.READY_FOR_REVIEW: {
        ProjectStatus.GENERATING_VIDEOS,
        ProjectStatus.ASSEMBLING,
        ProjectStatus.QUALITY_CHECK,
        ProjectStatus.GENERATING_METADATA,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.ASSEMBLY_FAILED: {
        ProjectStatus.ASSEMBLING,
        ProjectStatus.GENERATING_VIDEOS,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.QUALITY_CHECK: {
        ProjectStatus.READY_FOR_REVIEW,  # QC completes -> back to human review
        ProjectStatus.GENERATING_VIDEOS,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.GENERATING_METADATA: {
        ProjectStatus.READY_FOR_REVIEW,          # agent-driven flow (new)
        ProjectStatus.WAITING_FOR_FINAL_APPROVAL,  # legacy sync flow
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.WAITING_FOR_FINAL_APPROVAL: {
        ProjectStatus.COMPLETED,
        ProjectStatus.GENERATING_METADATA,
        ProjectStatus.ASSEMBLING,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.COMPLETED: set(),
    ProjectStatus.FAILED: set(),
    ProjectStatus.CANCELLED: set(),
}

# States where the workflow pauses for human input
PAUSE_STATES = {
    ProjectStatus.WAITING_FOR_STORY_APPROVAL,
    ProjectStatus.WAITING_FOR_IMAGE_SELECTION,
    ProjectStatus.WAITING_FOR_FINAL_APPROVAL,
    ProjectStatus.READY_FOR_REVIEW,
}


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


class ProjectNotFoundError(Exception):
    """Raised when a project ID does not exist."""


def can_transition(current: ProjectStatus, target: ProjectStatus) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, set())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MasterAgent:
    """Manages the lifecycle of a single project through the production pipeline."""

    def __init__(self, project: Project):
        self.project = project

    @classmethod
    def create(
        cls,
        idea: str,
        settings: Optional[ProjectSettings] = None,
        name: Optional[str] = None,
    ) -> "MasterAgent":
        display_name = (name or "").strip() or idea.strip()[:40]
        project = Project(
            name=display_name,
            idea=idea,
            settings=settings or ProjectSettings(),
            status=ProjectStatus.DRAFT,
        )
        project.state_timestamps[ProjectStatus.DRAFT.value] = _now()
        save_project(project)
        return cls(project)

    @classmethod
    def load(cls, project_id: str) -> "MasterAgent":
        project = load_project(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        return cls(project)

    def _transition_to(self, target: ProjectStatus) -> None:
        current = self.project.status
        if not can_transition(current, target):
            raise InvalidTransitionError(
                f"Cannot transition from {current.value} to {target.value}"
            )
        self.project.status = target
        self.project.updated_at = _now()
        self.project.state_timestamps[target.value] = _now()
        save_project(self.project)

    def _record_error(self, message: str) -> None:
        timestamped = f"[{_now()}] {message}"
        self.project.errors.append(timestamped)
        self.project.updated_at = _now()
        save_project(self.project)

    # --- Pipeline actions ---

    def start_planning(self) -> Project:
        """DRAFT -> PLANNING"""
        self._transition_to(ProjectStatus.PLANNING)
        return self.project

    def plan_story(self, story_agent=None) -> Project:
        """PLANNING -> WAITING_FOR_STORY_APPROVAL

        Calls the Story Agent to produce the storyboard + Project Bible,
        then stores them and pauses for human approval.
        """
        if self.project.status != ProjectStatus.PLANNING:
            raise InvalidTransitionError(
                f"plan_story requires PLANNING, current state is "
                f"{self.project.status.value}"
            )
        from agents.story import StoryAgent

        story_agent = story_agent or StoryAgent()
        output = story_agent.generate(
            idea=self.project.idea,
            content_type=self.project.settings.content_type,
            visual_style=self.project.settings.visual_style,
            target_duration=self.project.settings.duration_seconds,
            aspect_ratio=self.project.settings.aspect_ratio,
        )
        self._store_story(output)
        return self.project

    def _store_story(self, output) -> None:
        """Store a StoryboardOutput as the project's storyboard, scenes, and bible."""
        self.project.storyboard = {
            "title": output.title,
            "summary": output.summary,
            "scenes": [s.model_dump() for s in output.scenes],
        }
        self.project.project_bible = output.project_bible.model_dump()
        self.project.scenes = [
            Scene(
                description=s.visual_description,
                duration_seconds=max(1, int(round(s.estimated_duration))),
            )
            for s in output.scenes
        ]
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.WAITING_FOR_STORY_APPROVAL)

    def submit_storyboard(
        self,
        storyboard: Dict[str, Any],
        scenes: list[dict],
        project_bible: Optional[Dict[str, Any]] = None,
    ) -> Project:
        """PLANNING -> WAITING_FOR_STORY_APPROVAL"""
        self.project.storyboard = storyboard
        self.project.scenes = [Scene(**s) for s in scenes]
        if project_bible:
            self.project.project_bible = project_bible
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.WAITING_FOR_STORY_APPROVAL)
        return self.project

    def approve_story(self) -> Project:
        """WAITING_FOR_STORY_APPROVAL -> GENERATING_IMAGES"""
        self._transition_to(ProjectStatus.GENERATING_IMAGES)
        return self.project

    def reject_story(self, reason: str = "") -> Project:
        """WAITING_FOR_STORY_APPROVAL -> PLANNING"""
        self._record_error(f"Story rejected: {reason}" if reason else "Story rejected")
        self._transition_to(ProjectStatus.PLANNING)
        return self.project

    def regenerate_story(self, reason: str = "", story_agent=None) -> Project:
        """Regenerate the storyboard without creating images/videos.

        Works from either WAITING_FOR_STORY_APPROVAL or PLANNING. Moves to
        PLANNING first, re-runs the Story Agent with rejection feedback,
        and pauses again at WAITING_FOR_STORY_APPROVAL.
        """
        if self.project.status == ProjectStatus.WAITING_FOR_STORY_APPROVAL:
            self.reject_story(reason)
        elif self.project.status != ProjectStatus.PLANNING:
            raise InvalidTransitionError(
                "regenerate_story requires WAITING_FOR_STORY_APPROVAL or PLANNING, "
                f"current state is {self.project.status.value}"
            )

        from agents.story import StoryAgent

        story_agent = story_agent or StoryAgent()
        output = story_agent.generate(
            idea=self.project.idea,
            content_type=self.project.settings.content_type,
            visual_style=self.project.settings.visual_style,
            target_duration=self.project.settings.duration_seconds,
            aspect_ratio=self.project.settings.aspect_ratio,
            feedback=reason,
        )
        self._store_story(output)
        return self.project

    def images_generated(self, image_pairs: Optional[list[dict]] = None) -> Project:
        """GENERATING_IMAGES -> WAITING_FOR_IMAGE_SELECTION

        The Image Agent persists full candidate records directly on each scene
        (prompts, paths, status, errors). ``image_pairs`` is accepted for
        backward compatibility to set the legacy ``image_a``/``image_b`` paths,
        which the new flow keeps in sync automatically.
        """
        for pair in image_pairs or []:
            scene = next((s for s in self.project.scenes if s.id == pair["scene_id"]), None)
            if scene:
                scene.image_a = pair.get("image_a")
                scene.image_b = pair.get("image_b")
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.WAITING_FOR_IMAGE_SELECTION)
        return self.project

    def _apply_candidate_selection(self, scene: Scene, candidate_id: str) -> None:
        """Lock a candidate ('A' or 'B') for a scene based on stored records."""
        if candidate_id not in ("A", "B"):
            raise InvalidTransitionError(
                f"Candidate must be 'A' or 'B', got {candidate_id!r}"
            )
        candidate = next(
            (
                c
                for c in scene.image_candidates
                if c.candidate_id == candidate_id and c.status == "SUCCEEDED"
            ),
            None,
        )
        if candidate is None or not candidate.storage_path:
            raise InvalidTransitionError(
                f"Scene {scene.id} has no successful candidate {candidate_id}"
            )
        scene.selected_image_id = candidate_id
        scene.selected_image = candidate.storage_path

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return (
            value.startswith("/")
            or "/" in value
            or "\\" in value
            or value.endswith((".png", ".jpg", ".jpeg", ".webp"))
        )

    def _resolve_selection(self, scene: Scene, selection: str) -> None:
        """Resolve a user selection (candidate 'A'/'B' or legacy image path)."""
        if selection in ("A", "B"):
            self._apply_candidate_selection(scene, selection)
        elif self._looks_like_path(selection):
            scene.selected_image = selection
        else:
            raise InvalidTransitionError(
                f"Selection for scene {scene.id} must be 'A', 'B', or an "
                f"image path, got {selection!r}"
            )

    def select_image(self, scene_id: str, selection: str) -> Project:
        """Select/lock a candidate for one scene (no state transition).

        ``selection`` is 'A' or 'B' (resolved against stored candidates) or,
        for backward compatibility, an explicit image path.
        """
        scene = next((s for s in self.project.scenes if s.id == scene_id), None)
        if scene is None:
            raise InvalidTransitionError(f"Scene {scene_id} not found")
        self._resolve_selection(scene, selection)
        self.project.updated_at = _now()
        save_project(self.project)
        return self.project

    def select_images(self, selections: dict[str, str]) -> Project:
        """WAITING_FOR_IMAGE_SELECTION -> GENERATING_VIDEOS

        ``selections`` maps scene_id -> chosen candidate ('A'/'B') or legacy
        image path. Every scene must have a selected image.
        """
        for scene in self.project.scenes:
            if scene.id in selections:
                self._resolve_selection(scene, selections[scene.id])
            if not scene.selected_image:
                raise InvalidTransitionError(
                    f"Scene {scene.id} has no selected image"
                )
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.GENERATING_VIDEOS)
        return self.project

    def videos_generated(self, clips: Optional[dict] = None) -> Project:
        """GENERATING_VIDEOS -> ASSEMBLING

        Only transitions once EVERY scene has a completed video clip
        (``scene.video_clip``). The Video Agent persists full generation records
        on each scene; ``clips`` is accepted for backward compatibility to set
        the legacy ``video_clip`` paths, which the new flow keeps in sync.
        """
        for scene in self.project.scenes:
            if clips and scene.id in clips:
                scene.video_clip = clips[scene.id]
            if not scene.video_clip:
                raise InvalidTransitionError(
                    f"Scene {scene.id} has no completed video clip yet"
                )
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.ASSEMBLING)
        return self.project

    def video_scene_ready(self, scene_id: str) -> Project:
        """Record that one scene now has a completed clip (no transition)."""
        scene = next((s for s in self.project.scenes if s.id == scene_id), None)
        if scene is None:
            raise InvalidTransitionError(f"Scene {scene_id} not found")
        if not scene.video_clip:
            raise InvalidTransitionError(
                f"Scene {scene_id} has no completed video clip yet"
            )
        self.project.updated_at = _now()
        save_project(self.project)
        return self.project

    def begin_assembly(self) -> Project:
        """Move the project into the ASSEMBLING phase (ffmpeg-only work).

        Allowed from GENERATING_VIDEOS, ASSEMBLY_FAILED and READY_FOR_REVIEW
        (reassemble). Already ASSEMBLING is a no-op. Never regenerates clips.
        """
        if self.project.status == ProjectStatus.ASSEMBLING:
            return self.project
        self._transition_to(ProjectStatus.ASSEMBLING)
        return self.project

    def _append_assembly_record(self, record: AssemblyRecord) -> None:
        if any(r.assembly_id == record.assembly_id for r in self.project.assemblies):
            return
        # A new SUCCEEDED assembly replaces the on-disk final, so any QC result
        # and metadata package for an older assembly no longer describe the
        # current video.
        if record.status == AssemblyStatus.SUCCEEDED:
            previous = self.project.current_assembly_id
            if previous and previous != record.assembly_id:
                reason = f"superseded by assembly {record.assembly_id[:8]}"
                self.invalidate_qc(reason)
                self.invalidate_metadata(reason)
        self.project.assemblies.append(record)
        self.project.current_assembly_id = record.assembly_id
        self.project.updated_at = _now()

    def assembly_complete(
        self,
        final_video_path: Optional[str] = None,
        record: Optional[AssemblyRecord] = None,
    ) -> Project:
        """ASSEMBLING -> READY_FOR_REVIEW

        Records the successful assembly (preserving history: any previously
        SUCCEEDED assembly is marked STALE because the final file is replaced)
        and points ``project.final_video`` at the new output. ``record=`` may be
        omitted for backward compatibility (only sets ``final_video``).
        """
        if record is None and not final_video_path:
            raise InvalidTransitionError(
                "assembly_complete requires a final video path or an assembly record"
            )
        if record is not None:
            if record.status != AssemblyStatus.SUCCEEDED:
                raise InvalidTransitionError(
                    "assembly_complete requires a SUCCEEDED assembly record"
                )
            # The on-disk final file is replaced by this assembly, so any older
            # successful assembly is now stale (never the record just added).
            for old in self.project.assemblies:
                if old.succeeded and old.assembly_id != record.assembly_id:
                    old.status = AssemblyStatus.STALE
                    old.stale_reason = (
                        old.stale_reason
                        or f"superseded by assembly {record.assembly_id}"
                    )
            self._append_assembly_record(record)
            if not record.output_path:
                raise InvalidTransitionError(
                    "assembly_complete requires the record's output path"
                )
            self.project.final_video = record.output_path
        elif final_video_path:
            self.project.final_video = final_video_path
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.READY_FOR_REVIEW)
        return self.project

    def assembly_failed(
        self,
        error: str,
        record: Optional[AssemblyRecord] = None,
    ) -> Project:
        """ASSEMBLING -> ASSEMBLY_FAILED

        Records the failed assembly (history preserved) and errors on the
        project. The user can retry later without regenerating any clips.
        """
        if record is not None:
            record.status = AssemblyStatus.FAILED
            record.error = error
            record.completed_at = _now()
            self._append_assembly_record(record)
        self._record_error(f"Assembly failed: {error}")
        self._transition_to(ProjectStatus.ASSEMBLY_FAILED)
        return self.project

    def invalidate_assembly(self, reason: str = "") -> Project:
        """Mark the current successful assembly STALE and allow clip rework.

        Called when a scene clip is regenerated: the stored final video no
        longer matches the scene's current clip, so it must be reassembled.
        Returns the project to GENERATING_VIDEOS so clip generation endpoints
        keep working; the newer generation flow will move it back to ASSEMBLING
        once every scene has a current clip.
        """
        reason = reason or "a scene clip was regenerated"
        for record in self.project.assemblies:
            if record.succeeded and not record.is_stale:
                record.status = AssemblyStatus.STALE
                record.stale_reason = reason
        # A regenerated clip makes the assembly stale, so QC results and any
        # metadata package describing it are no longer current either.
        self.invalidate_qc(reason)
        self.invalidate_metadata(reason)
        if self.project.status != ProjectStatus.GENERATING_VIDEOS:
            if self.project.status in {
                ProjectStatus.READY_FOR_REVIEW,
                ProjectStatus.ASSEMBLING,
                ProjectStatus.ASSEMBLY_FAILED,
            }:
                self.project.updated_at = _now()
                save_project(self.project)
                self._transition_to(ProjectStatus.GENERATING_VIDEOS)
        else:
            self.project.updated_at = _now()
            save_project(self.project)
        return self.project

    def latest_assembly(self) -> Optional[AssemblyRecord]:
        return self.project.assemblies[-1] if self.project.assemblies else None

    def successful_assembly(self) -> Optional[AssemblyRecord]:
        """The most recent non-stale SUCCEEDED assembly (i.e. the live final)."""
        for record in reversed(self.project.assemblies):
            if record.succeeded:
                return record
        return None

    def start_quality_check(self) -> Project:
        """READY_FOR_REVIEW -> QUALITY_CHECK

        Enters the QC state before the automated checks run. sync anyway: the
        API moves QUALITY_CHECK -> READY_FOR_REVIEW once the report is stored.
        """
        self._transition_to(ProjectStatus.QUALITY_CHECK)
        return self.project

    def qc_running(self, report: QCReport) -> Project:
        """Persist a RUNNING QC report (no state transition).

        Used so a crash mid-run leaves an inspectable RUNNING record that is
        NOT considered current (only PASSED/WARNINGS/FAILED reports count) and
        can simply be superseded by rerunning QC.
        """
        if not any(r.qc_id == report.qc_id for r in self.project.qc_reports):
            self.project.qc_reports.append(report)
        self.project.current_qc_id = report.qc_id
        self.project.updated_at = _now()
        save_project(self.project)
        return self.project

    def qc_complete(self, report: QCReport) -> Project:
        """QUALITY_CHECK -> READY_FOR_REVIEW

        QC never publishes on its own: whether it PASSED, WARNED, or FAILED the
        project returns to READY_FOR_REVIEW so the human still reviews the
        final video. The report is persisted as history (reruns append).
        """
        if report.status not in {
            QCStatus.PASSED,
            QCStatus.WARNINGS,
            QCStatus.FAILED,
        }:
            raise InvalidTransitionError(
                f"qc_complete requires a finished report, got {report.status}"
            )
        if not report.completed_at:
            from datetime import datetime, timezone

            report.completed_at = datetime.now(timezone.utc).isoformat()
        if not any(r.qc_id == report.qc_id for r in self.project.qc_reports):
            self.project.qc_reports.append(report)
        else:
            # Keep the persisted report in sync with the final state of the run.
            idx = next(
                i for i, r in enumerate(self.project.qc_reports)
                if r.qc_id == report.qc_id
            )
            self.project.qc_reports[idx] = report
        self.project.current_qc_id = report.qc_id
        # Legacy summary for any older consumers.
        self.project.qc_result = QCResult(
            passed=report.status == QCStatus.PASSED,
            issues=list(report.findings),
            checked_at=report.completed_at,
        )
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.READY_FOR_REVIEW)
        return self.project

    def invalidate_qc(self, reason: str = "") -> Project:
        """Mark every prior QC report INVALID (a newer assembly is in place).

        QC results are only meaningful for the exact assembly they checked.
        As soon as a scene regenerates (assembly becomes STALE) or a new
        assembly supersedes the old final, older reports no longer describe
        the current video and must not be shown as the current result.
        """
        reason = reason or "the final video changed"
        for report in self.project.qc_reports:
            if report.status != QCStatus.INVALID:
                report.status = QCStatus.INVALID
                report.raw["invalidated_reason"] = reason
        if self.project.current_qc_id and all(
            r.qc_id != self.project.current_qc_id or r.status == QCStatus.INVALID
            for r in self.project.qc_reports
        ):
            self.project.current_qc_id = None
        self.project.updated_at = _now()
        return self.project

    def latest_qc_report(self) -> Optional[QCReport]:
        """The most recent QC report, or None."""
        if not self.project.qc_reports:
            return None
        for report in reversed(self.project.qc_reports):
            if report.is_current:
                return report
        return self.project.qc_reports[-1]

    def metadata_start(self) -> Project:
        """READY_FOR_REVIEW -> METADATA_GENERATION (manual).

        The project stays in READY_FOR_REVIEW after QC until the user chooses
        to generate metadata. The agent-driven flow returns the project to
        READY_FOR_REVIEW when generation completes.
        """
        self._transition_to(ProjectStatus.GENERATING_METADATA)
        return self.project

    def metadata_running(self, record: MetadataRecord) -> Project:
        """Persist a metadata record without a state transition (crash-safe).

        Mirrors ``qc_running``: the record is tracked before the LLM call
        finishes so a crash leaves an inspectable DRAFT that can simply be
        superseded by rerunning generation.
        """
        if record.status == MetadataStatus.CURRENT:
            record.status = MetadataStatus.DRAFT
        record_copy = record.model_copy(deep=True)
        if not any(r.metadata_id == record.metadata_id for r in self.project.metadata_records):
            self.project.metadata_records.append(record_copy)
        else:
            idx = next(
                i for i, r in enumerate(self.project.metadata_records)
                if r.metadata_id == record.metadata_id
            )
            self.project.metadata_records[idx] = record_copy
        self.project.current_metadata_id = record.metadata_id
        self.project.updated_at = _now()
        save_project(self.project)
        return self.project

    def metadata_complete(self, record: MetadataRecord) -> Project:
        """METADATA_GENERATION -> READY_FOR_REVIEW

        Persists the finished metadata as the CURRENT version for its assembly
        (older versions for the same assembly become DRAFT history). Never
        publishes: the project returns to READY_FOR_REVIEW so the human reviews
        the video + package before a manual YouTube upload.
        """
        record.status = MetadataStatus.CURRENT
        record.project_id = self.project.id
        if not record.created_at:
            record.created_at = _now()
        record_copy = record.model_copy(deep=True)
        if not any(r.metadata_id == record.metadata_id for r in self.project.metadata_records):
            self.project.metadata_records.append(record_copy)
        else:
            idx = next(
                i for i, r in enumerate(self.project.metadata_records)
                if r.metadata_id == record.metadata_id
            )
            self.project.metadata_records[idx] = record_copy
        # Older versions for the SAME assembly become DRAFT history.
        for old in self.project.metadata_records:
            if old.assembly_id == record.assembly_id and old.metadata_id != record.metadata_id:
                if old.status != MetadataStatus.STALE:
                    old.status = MetadataStatus.DRAFT
        self.project.current_metadata_id = record.metadata_id
        self._sync_youtube_metadata(record)
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.READY_FOR_REVIEW)
        return self.project

    def update_metadata(self, metadata_id: str, **fields) -> Project:
        """Edit user-facing metadata fields on a record (no state transition).

        Only known fields are accepted; the AI-generated original is preserved
        in ``ai_original`` on the first manual edit so the audit trail keeps the
        untouched draft.
        """
        record = next(
            (r for r in self.project.metadata_records if r.metadata_id == metadata_id),
            None,
        )
        if record is None:
            raise InvalidTransitionError(f"Metadata {metadata_id} not found")
        allowed = {
            "title", "description", "hashtags", "keywords",
            "category", "content_summary",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise InvalidTransitionError(
                f"Cannot edit metadata fields: {sorted(unknown)}"
            )
        if not fields:
            return self.project
        if record.ai_original is None:
            record.ai_original = {
                "title": record.title,
                "description": record.description,
                "hashtags": list(record.hashtags),
                "keywords": list(record.keywords),
                "category": record.category,
                "content_summary": record.content_summary,
            }
        for key, value in fields.items():
            setattr(record, key, value)
        record.is_edited = True
        if record.status not in (MetadataStatus.STALE,):
            record.status = MetadataStatus.CURRENT
            self.project.current_metadata_id = record.metadata_id
        self._sync_youtube_metadata(record)
        self.project.updated_at = _now()
        save_project(self.project)
        return self.project

    def invalidate_metadata(self, reason: str = "") -> Project:
        """Mark every metadata record STALE when the final video changes.

        Metadata describes a specific assembly. As soon as a scene regenerates
        (assembly STALE) or a new assembly supersedes the old final, no metadata
        version describes the current video — they are STALE (history kept) and
        the current id is cleared until the user regenerates for the new final.
        """
        reason = reason or "the final video changed"
        for record in self.project.metadata_records:
            if record.status != MetadataStatus.STALE:
                record.status = MetadataStatus.STALE
                record.raw["invalidated_reason"] = reason
        if self.project.current_metadata_id and all(
            r.metadata_id != self.project.current_metadata_id
            or r.status == MetadataStatus.STALE
            for r in self.project.metadata_records
        ):
            self.project.current_metadata_id = None
        self.project.updated_at = _now()
        return self.project

    def current_metadata(self) -> Optional[MetadataRecord]:
        """The CURRENT metadata record, or None."""
        current_id = self.project.current_metadata_id
        if not current_id:
            return None
        for record in reversed(self.project.metadata_records):
            if record.metadata_id == current_id and record.status == MetadataStatus.CURRENT:
                return record
        for record in reversed(self.project.metadata_records):
            if record.metadata_id == current_id:
                return record
        return None

    def _sync_youtube_metadata(self, record: MetadataRecord) -> None:
        """Keep the legacy working copy in sync with the CURRENT record."""
        self.project.youtube_metadata = YouTubeMetadata(
            title=record.title or None,
            description=record.description or None,
            hashtags=list(record.hashtags),
            keywords=list(record.keywords),
            category=record.category or None,
            content_summary=record.content_summary or None,
        )

    def metadata_generated(
        self,
        title: str,
        description: str,
        hashtags: list[str],
    ) -> Project:
        """GENERATING_METADATA -> WAITING_FOR_FINAL_APPROVAL (legacy path).

        Kept for backward compatibility (test harness / older callers). The
        agent-driven flow uses ``metadata_start`` + ``metadata_complete`` and
        returns to READY_FOR_REVIEW instead.
        """
        self.project.youtube_metadata = YouTubeMetadata(
            title=title, description=description, hashtags=hashtags
        )
        self.project.updated_at = _now()
        save_project(self.project)
        self._transition_to(ProjectStatus.WAITING_FOR_FINAL_APPROVAL)
        return self.project

    def final_approve(self) -> Project:
        """WAITING_FOR_FINAL_APPROVAL -> COMPLETED"""
        self._transition_to(ProjectStatus.COMPLETED)
        return self.project

    def final_reject(self, reason: str = "") -> Project:
        """WAITING_FOR_FINAL_APPROVAL -> GENERATING_METADATA"""
        self._record_error(
            f"Final rejected: {reason}" if reason else "Final rejected"
        )
        self._transition_to(ProjectStatus.GENERATING_METADATA)
        return self.project

    def fail(self, reason: str) -> Project:
        """Any active state -> FAILED"""
        self._record_error(reason)
        self._transition_to(ProjectStatus.FAILED)
        return self.project

    def cancel(self) -> Project:
        """Any active state -> CANCELLED"""
        self._transition_to(ProjectStatus.CANCELLED)
        return self.project

    @property
    def is_paused(self) -> bool:
        return self.project.status in PAUSE_STATES

    @property
    def is_terminal(self) -> bool:
        return self.project.status in {
            ProjectStatus.COMPLETED,
            ProjectStatus.FAILED,
            ProjectStatus.CANCELLED,
        }

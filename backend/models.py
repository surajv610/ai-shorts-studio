from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    DRAFT = "DRAFT"
    PLANNING = "PLANNING"
    WAITING_FOR_STORY_APPROVAL = "WAITING_FOR_STORY_APPROVAL"
    GENERATING_IMAGES = "GENERATING_IMAGES"
    WAITING_FOR_IMAGE_SELECTION = "WAITING_FOR_IMAGE_SELECTION"
    GENERATING_VIDEOS = "GENERATING_VIDEOS"
    ASSEMBLING = "ASSEMBLING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    ASSEMBLY_FAILED = "ASSEMBLY_FAILED"
    QUALITY_CHECK = "QUALITY_CHECK"
    GENERATING_METADATA = "GENERATING_METADATA"
    WAITING_FOR_FINAL_APPROVAL = "WAITING_FOR_FINAL_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AssemblyStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    STALE = "STALE"
    FAILED = "FAILED"


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    duration_seconds: int = 4
    image_a: Optional[str] = None
    image_b: Optional[str] = None
    selected_image: Optional[str] = None
    selected_image_id: Optional[str] = None
    video_clip: Optional[str] = None
    image_prompt: Optional[str] = None
    image_candidates: List["ImageCandidate"] = Field(default_factory=list)
    video_prompt: Optional[str] = None
    video_generations: List["VideoGeneration"] = Field(default_factory=list)
    current_video_id: Optional[str] = None


class ImageCandidate(BaseModel):
    """A single generated image candidate (A or B) for a scene."""

    candidate_id: str = "A"
    project_id: str = ""
    scene_id: str = ""
    provider: str = ""
    model: str = ""
    generation_id: str = ""
    prompt: str = ""
    storage_path: str = ""
    status: str = "PENDING"  # PENDING | SUCCEEDED | FAILED
    error: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"


class VideoGeneration(BaseModel):
    """A single image-to-video generation attempt for a scene.

    History is retained: regenerating a scene appends a new record and only
    updates ``Scene.current_video_id`` after the new attempt SUCCEEDs, so old
    (previously successful) clips are never deleted.
    """

    attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str = ""
    scene_id: str = ""
    provider: str = ""
    model: str = ""
    job_id: str = ""  # provider operation / generation id (e.g. Veo operation name)
    status: str = "PENDING"  # PENDING | SUBMITTED | PROCESSING | SUCCEEDED | FAILED | CANCELLED
    prompt: str = ""
    source_image_id: str = ""  # locked image candidate (A/B) or legacy ref
    source_image: str = ""  # resolved local path of the input image
    duration_seconds: int = 0
    output_path: str = ""  # local MP4 path once SUCCEEDED
    error: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw: Dict[str, Any] = Field(default_factory=dict)  # provider metadata (no secrets)

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"

    @staticmethod
    def latest_for_scene(scene: "Scene") -> Optional["VideoGeneration"]:
        gens = getattr(scene, "video_generations", None) or []
        return gens[-1] if gens else None


class AssemblyRecord(BaseModel):
    """One attempt at assembling the project's scene clips into a final MP4.

    History is preserved: reassembling appends a new record and marks the
    previously SUCCEEDED record STALE (the final file on disk is always the
    latest successful assembly). ``input_clip_ids`` records the exact
    VideoGeneration attempt ids that were concatenated so we can detect when a
    regenerated scene invalidates the current final video.
    """

    assembly_id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str = ""
    status: AssemblyStatus = AssemblyStatus.PENDING
    input_scene_ids: List[str] = Field(default_factory=list)
    input_clip_ids: List[str] = Field(default_factory=list)
    input_clip_paths: List[str] = Field(default_factory=list)
    scene_count: int = 0
    output_path: str = ""
    output_duration: float = 0.0
    output_width: int = 0
    output_height: int = 0
    output_codec: str = ""
    output_size_bytes: int = 0
    ffmpeg_version: str = ""
    error: Optional[str] = None
    stale_reason: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)  # e.g. action used, durations

    @property
    def succeeded(self) -> bool:
        return self.status == AssemblyStatus.SUCCEEDED

    @property
    def is_stale(self) -> bool:
        return self.status == AssemblyStatus.STALE


class ProjectSettings(BaseModel):
    content_type: str = "Timelapse"
    visual_style: str = "Photorealistic"
    duration_seconds: int = 30
    aspect_ratio: str = "9:16"


class YouTubeMetadata(BaseModel):
    """The current editable "working copy" of the YouTube Shorts package.

    Kept in sync with the CURRENT ``MetadataRecord`` (and user edits to it) so
    legacy consumers keep working. The full auditable history lives in
    ``Project.metadata_records``.
    """

    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    content_summary: Optional[str] = None


class MetadataStatus(str, Enum):
    DRAFT = "DRAFT"        # an older version for the same assembly (keep history)
    CURRENT = "CURRENT"    # the latest successful generation (or manually edited) version
    STALE = "STALE"        # superseded because the final video changed


class MetadataRecord(BaseModel):
    """One auditable metadata version tied to a specific assembly.

    History is preserved: regenerating appends a new record (the previous
    CURRENT becomes DRAFT while still on the same assembly). When the final
    video changes (a scene regenerates / a new assembly supersedes the old
    final), every record describing the old assembly becomes STALE.
    ``prompt``/``model``/``provider``/``qc_id`` make the version reproducible.
    The AI-generated original is preserved in ``ai_original`` the first time a
    user manually edits any field.
    """

    metadata_id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str = ""
    assembly_id: str = ""
    qc_id: Optional[str] = None
    status: MetadataStatus = MetadataStatus.DRAFT
    title: str = ""
    description: str = ""
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    category: str = ""
    content_summary: str = ""
    prompt: str = ""            # the exact metadata-generation prompt (auditable)
    model: str = ""             # provider model used (if available)
    provider: str = ""          # provider name (mock / openai / ...)
    ai_original: Optional[Dict[str, Any]] = None  # untouched AI output after first edit
    is_edited: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw: Dict[str, Any] = Field(default_factory=dict)  # extra metadata (no secrets)


class QCResult(BaseModel):
    passed: bool = False
    issues: List[str] = Field(default_factory=list)
    checked_at: Optional[str] = None


class QCStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    WARNINGS = "WARNINGS"
    FAILED = "FAILED"
    INVALID = "INVALID"  # superseded by a newer assembly / scene regeneration


class QCCheck(BaseModel):
    """Result of one automated QC check (PASS/WARN/FAIL).

    ``severity`` classifies how much a WARN/FAIL should block: ``info``
    (non-blocking, e.g. missing audio in V1), ``medium`` (blocking warning),
    ``high`` (failure).
    """

    check_name: str
    status: str = "PENDING"  # PASS | WARN | FAIL
    severity: str = "info"
    message: str = ""
    measured_value: Optional[Any] = None
    expected_value: Optional[Any] = None

    @property
    def blocking(self) -> bool:
        """True when this check must keep the video from being considered ready."""
        return self.status == "FAIL" or (
            self.status == "WARN" and self.severity in ("medium", "high")
        )


class QCReport(BaseModel):
    """One QC run against a specific (successful) assembly of the final video.

    History is preserved: rerunning QC appends a new report. When a scene is
    regenerated (assembly becomes STALE) or a new assembly supersedes the old
    final, prior reports are marked INVALID so only the report tied to the
    current assembly is ever considered current.
    """

    qc_id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str = ""
    assembly_id: str = ""
    status: QCStatus = QCStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checks: List[QCCheck] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    summary: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw: Dict[str, Any] = Field(default_factory=dict)  # e.g. qc_version, checkers

    @property
    def passed(self) -> bool:
        return self.status == QCStatus.PASSED

    @property
    def is_current(self) -> bool:
        return self.status in (QCStatus.PASSED, QCStatus.WARNINGS, QCStatus.FAILED)


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    idea: str
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    status: ProjectStatus = ProjectStatus.DRAFT
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    state_timestamps: Dict[str, str] = Field(default_factory=dict)
    storyboard: Optional[Dict[str, Any]] = None
    project_bible: Dict[str, Any] = Field(default_factory=dict)
    scenes: List[Scene] = Field(default_factory=list)
    image_generations: List[Dict[str, Any]] = Field(default_factory=list)
    video_generations: List[Dict[str, Any]] = Field(default_factory=list)
    youtube_metadata: YouTubeMetadata = Field(default_factory=YouTubeMetadata)
    metadata_records: List["MetadataRecord"] = Field(default_factory=list)
    current_metadata_id: Optional[str] = None
    qc_result: Optional[QCResult] = None
    qc_reports: List[QCReport] = Field(default_factory=list)
    current_qc_id: Optional[str] = None
    final_video: Optional[str] = None
    assemblies: List[AssemblyRecord] = Field(default_factory=list)
    current_assembly_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


# Resolve the forward references (ImageCandidate / VideoGeneration are declared
# after Scene).
Scene.model_rebuild()
Project.model_rebuild()

from typing import List, Optional

from pydantic import BaseModel, Field


class StoryScene(BaseModel):
    """One scene within a storyboard."""

    scene_id: int
    scene_number: int
    title: str
    visual_description: str
    action: str
    estimated_duration: float
    continuity_notes: str


class ProjectBible(BaseModel):
    """Guiding style/reference document for the whole video."""

    subject: str
    environment: str
    time_lighting: str
    camera: str
    lens_look: str
    visual_style: str
    color_appearance: str
    physical_characteristics: str
    continuity_rules: str
    negative_constraints: str
    aspect_ratio: str
    target_duration: float


class StoryboardOutput(BaseModel):
    """Structured schema the AI model must return for a storyboard."""

    title: str
    summary: str
    scenes: List[StoryScene]
    project_bible: ProjectBible


VALID_ASPECT_RATIOS = {"9:16", "1:1", "16:9", "4:5"}

MIN_SCENE_DURATION = 1.0
MAX_SCENE_DURATION = 30.0
DEFAULT_SCENE_DURATION_TOLERANCE = 0.2  # 20% around target


class ValidationError(Exception):
    """Raised when a storyboard/project bible fails validation."""


def validate_scenes(scenes: List[StoryScene]) -> List[str]:
    """Return list of human-readable issues. Empty list means valid."""
    issues: List[str] = []
    numbers = [s.scene_number for s in scenes]
    if len(numbers) != len(set(numbers)):
        issues.append("scene_number values must be unique")
    if sorted(numbers) != list(range(1, len(scenes) + 1)):
        issues.append("scene_number values must be sequential from 1")
    for s in scenes:
        if not s.title.strip():
            issues.append(f"scene {s.scene_number}: title is empty")
        if not s.visual_description.strip():
            issues.append(f"scene {s.scene_number}: visual_description is empty")
        if not s.action.strip():
            issues.append(f"scene {s.scene_number}: action is empty")
        if not (MIN_SCENE_DURATION <= s.estimated_duration <= MAX_SCENE_DURATION):
            issues.append(
                f"scene {s.scene_number}: estimated_duration {s.estimated_duration} "
                f"out of range [{MIN_SCENE_DURATION}, {MAX_SCENE_DURATION}]"
            )
        if not s.continuity_notes.strip():
            issues.append(f"scene {s.scene_number}: continuity_notes is empty")
    return issues


def validate_project_bible(bible: ProjectBible) -> List[str]:
    """Return list of human-readable issues. Empty list means valid."""
    issues: List[str] = []
    required_fields = [
        "subject",
        "environment",
        "time_lighting",
        "camera",
        "lens_look",
        "visual_style",
        "color_appearance",
        "physical_characteristics",
        "continuity_rules",
        "negative_constraints",
    ]
    for field in required_fields:
        if not getattr(bible, field, "").strip():
            issues.append(f"project_bible.{field} is empty")
    if bible.aspect_ratio not in VALID_ASPECT_RATIOS:
        issues.append(
            f"project_bible.aspect_ratio {bible.aspect_ratio!r} not in "
            f"valid set {sorted(VALID_ASPECT_RATIOS)}"
        )
    if bible.target_duration <= 0:
        issues.append("project_bible.target_duration must be positive")
    return issues


def validate_total_duration(scenes: List[StoryScene], target: float) -> List[str]:
    """Total scene duration should be within tolerance of the target."""
    total = sum(s.estimated_duration for s in scenes)
    if total <= 0:
        return ["total scene duration must be positive"]
    allowed = max(target * DEFAULT_SCENE_DURATION_TOLERANCE, 1.0)
    if abs(total - target) > allowed:
        return [
            f"total duration {total:.1f}s deviates from target {target:.1f}s "
            f"by more than tolerance {allowed:.1f}s"
        ]
    return []


def validate_storyboard_output(output: StoryboardOutput) -> List[str]:
    if not output.scenes:
        return ["storyboard must contain at least one scene"]
    issues = validate_scenes(output.scenes)
    issues += validate_project_bible(output.project_bible)
    return issues

"""Schemas for the Image Agent's structured image-prompt output."""

from typing import List

from pydantic import BaseModel, Field


class ImagePromptOutput(BaseModel):
    """Structured image prompt the LLM must return for one scene.

    The agent explicitly separates GLOBAL CONTINUITY (bible-level rules that
    must hold across every scene) from SCENE-SPECIFIC ACTION (what is new in
    this scene's progression). Two ready-to-use prompts are produced so the
    two candidates (A = primary composition, B = meaningfully different
    composition) are generated from distinct, auditable prompts.
    """

    global_continuity: str = Field(description="Anything that must stay identical across all scenes")
    scene_action: str = Field(description="What is happening in this scene only (progression step)")
    prompt_a: str = Field(description="Full image prompt for candidate A (primary composition)")
    prompt_b: str = Field(description="Full image prompt for candidate B (alternate composition)")
    composition_notes: str = Field(
        default="", description="Short note on how prompt_b differs compositionally from prompt_a"
    )


IMAGE_PROMPT_SCHEMA = {
    "name": "image_prompt_output",
    "schema": ImagePromptOutput.model_json_schema(),
}

MIN_PROMPT_LENGTH = 20
MAX_PROMPT_LENGTH = 4000


class ImagePromptError(Exception):
    """Raised when image-prompt generation or validation fails."""


def validate_image_prompt_output(output: ImagePromptOutput) -> List[str]:
    """Return human-readable issues. Empty list means valid."""
    issues: List[str] = []
    if not output.global_continuity.strip():
        issues.append("global_continuity is empty")
    if not output.scene_action.strip():
        issues.append("scene_action is empty")
    for field in ("prompt_a", "prompt_b"):
        value = getattr(output, field).strip()
        if not value:
            issues.append(f"{field} is empty")
        elif not (MIN_PROMPT_LENGTH <= len(value) <= MAX_PROMPT_LENGTH):
            issues.append(
                f"{field} length {len(value)} out of range "
                f"[{MIN_PROMPT_LENGTH}, {MAX_PROMPT_LENGTH}]"
            )
    if output.prompt_a.strip() == output.prompt_b.strip():
        issues.append("prompt_a and prompt_b must differ (candidate B needs a different composition)")
    return issues
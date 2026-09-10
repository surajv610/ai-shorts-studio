"""Schemas for the Video Agent's structured animation-prompt output."""

from typing import List

from pydantic import BaseModel, Field


class VideoPromptOutput(BaseModel):
    """Structured animation prompt the LLM must return for one scene.

    The agent explicitly separates GLOBAL CONTINUITY (bible-level rules that
    must hold across every scene of the clip sequence) from SCENE-SPECIFIC
    ACTION (what changes/progresses in THIS scene). The ``prompt`` field is the
    complete, ready-to-use animation prompt sent to the image-to-video model.
    """

    global_continuity: str = Field(
        description="What must remain identical across the whole video: subject identity, "
        "plant/species, environment, soil, camera perspective, visual style, lighting, "
        "and other important visual features."
    )
    scene_action: str = Field(
        description="The one progression step happening in this scene (e.g. the seed cracks "
        "open). Focus on WHAT CHANGES, not a re-description of the image."
    )
    motion: str = Field(
        description="How the subject moves in this scene (growth, swaying, particles), "
        "including realistic motion physics."
    )
    environment: str = Field(
        description="Environmental movement: soil/texture settling, light through dust, "
        "micro-particles, etc. Keep the environment stable unless the scene calls for it."
    )
    camera: str = Field(
        description="Camera behavior. Keep it locked/matching the source image unless the "
        "storyboard explicitly calls for movement. Never allow arbitrary camera drift."
    )
    lighting: str = Field(
        description="How lighting behaves/moves in time (golden-hour shift, shadows "
        "lengthening) while preserving the source lighting character."
    )
    timelapse: str = Field(
        default="",
        description="Timelapse behavior when applicable (e.g. slow chronological growth, "
        "nothing else changes around the plant). Empty when not a timelapse scene.",
    )
    negative_constraints: str = Field(
        default="",
        description="Things the model must NOT do: no new props, no hands/puppeteering, no "
        "duplicate plants, no morphing into different species, no text/watermarks, no "
        "jarring camera cuts, no scene rewrites.",
    )
    prompt: str = Field(
        description="The complete animation prompt to send to the image-to-video model. "
        "It combines continuity + scene action + motion + camera + lighting + constraints "
        "into one coherent instruction."
    )


VIDEO_PROMPT_SCHEMA = {
    "name": "video_prompt_output",
    "schema": VideoPromptOutput.model_json_schema(),
}

MIN_PROMPT_LENGTH = 20
MAX_PROMPT_LENGTH = 4000


class VideoPromptError(Exception):
    """Raised when animation-prompt generation or validation fails."""


def validate_video_prompt_output(output: VideoPromptOutput) -> List[str]:
    """Return human-readable issues. Empty list means valid."""
    issues: List[str] = []
    for field in ("global_continuity", "scene_action", "motion", "camera"):
        value = getattr(output, field).strip()
        if not value:
            issues.append(f"{field} is empty")
    prompt = output.prompt.strip()
    if not prompt:
        issues.append("prompt is empty")
    elif not (MIN_PROMPT_LENGTH <= len(prompt) <= MAX_PROMPT_LENGTH):
        issues.append(
            f"prompt length {len(prompt)} out of range "
            f"[{MIN_PROMPT_LENGTH}, {MAX_PROMPT_LENGTH}]"
        )
    return issues
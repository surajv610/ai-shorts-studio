"""Story Agent — turns a user idea into a structured storyboard + Project Bible."""

import logging
from typing import Optional

from agents.llm import call_structured, ProviderError
from agents.story_schemas import (
    StoryboardOutput,
    ValidationError,
    validate_storyboard_output,
    validate_total_duration,
)

logger = logging.getLogger(__name__)

STORYBOARD_SCHEMA = {
    "name": "storyboard_output",
    "schema": StoryboardOutput.model_json_schema(),
}

SYSTEM_PROMPT = (
    "You are the Story Agent for an AI Shorts production studio. "
    "You turn a single user idea into a structured, shot-by-shot storyboard "
    "and a Project Bible that keeps all downstream AI (image, video) consistent. "
    "Design for short vertical videos but apply principles that work for any format. "
    "Always return the exact JSON schema provided, with every field populated. "
    "The project Bible must be complete: subject, environment, time/lighting, camera, "
    "lens/look, visual style, color/appearance guidance, physical characteristics, "
    "continuity rules, negative constraints, aspect ratio, and target duration."
)

REJECTION_HINT = (
    ".\n\nNote: the previous storyboard was rejected. "
    "Regenerate a fresh version and address this feedback where possible."
)


def _build_user_message(
    idea: str,
    content_type: str,
    visual_style: str,
    target_duration: float,
    aspect_ratio: str,
    feedback: Optional[str] = None,
) -> str:
    msg = (
        f"USER IDEA: {idea}\n"
        f"VIDEO TYPE: {content_type}\n"
        f"VISUAL STYLE: {visual_style}\n"
        f"TARGET DURATION (seconds): {target_duration}\n"
        f"ASPECT RATIO: {aspect_ratio}\n"
        "Break the video into a small number of clear scenes. "
        "Sum of scene estimated_duration should match the target duration."
    )
    if feedback:
        msg += REJECTION_HINT.replace("\n", " ").replace(".**\n\n", " ") + " " + feedback
    return msg


class StoryAgent:
    """Generates and validates storyboards + project bibles."""

    def __init__(self, *, llm=call_structured):
        self._llm = llm

    def generate(
        self,
        idea: str,
        content_type: str,
        visual_style: str,
        target_duration: float,
        aspect_ratio: str,
        feedback: Optional[str] = None,
        model: Optional[str] = None,
    ) -> StoryboardOutput:
        try:
            raw = self._llm(
                system_prompt=SYSTEM_PROMPT,
                user_message=_build_user_message(
                    idea,
                    content_type,
                    visual_style,
                    target_duration,
                    aspect_ratio,
                    feedback,
                ),
                json_schema=STORYBOARD_SCHEMA,
                model=model,
            )
        except ProviderError as e:
            raise ValidationError(f"Story generation provider error: {e}")

        output = StoryboardOutput.model_validate(raw)

        issues = validate_storyboard_output(output)
        issues += validate_total_duration(output.scenes, target_duration)
        if issues:
            raise ValidationError("; ".join(issues))

        return output

    def validate(self, output: StoryboardOutput, target_duration: float) -> None:
        issues = validate_storyboard_output(output)
        issues += validate_total_duration(output.scenes, target_duration)
        if issues:
            raise ValidationError("; ".join(issues))

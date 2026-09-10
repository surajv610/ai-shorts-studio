"""Metadata Agent — produces the YouTube Shorts upload package.

The agent reads the CURRENT final assembly's context (idea + Project Bible +
storyboard + scene summaries + final video info + current QC result) and asks
the LLM for a structured, validated upload package:

  * title (concise, accurate, curiosity without clickbait, <=100 chars)
  * description (plain text, explains what the viewer sees)
  * hashtags (relevant, unique, starts with '#')
  * keywords / category / content_summary (optional suggestions)

It NEVER uploads, schedules, publishes, edits the video, regenerates images or
clips, or touches the storyboard / Project Bible. Human review + manual upload
is the whole point of the workflow. The generation prompt is built from facts
only and stored verbatim on the record so every version is auditable.

Gating (assembly prerequisites / QC block) lives in the API + master agent,
not here.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.assembly import probe_video
from backend.config import Settings, get_settings
from backend.models import Project
from backend.providers.base import ProviderError
from agents.llm import call_structured
from agents.metadata_schemas import (
    METADATA_SCHEMA,
    MetadataError,
    MetadataOutput,
    validate_metadata_output,
)

SYSTEM_PROMPT = (
    "You are the Metadata Agent for an AI Shorts production studio. You write "
    "the YouTube Shorts upload package for the CURRENT final video.\n"
    "TITLE: concise, natural-language, accurate, curiosity without clickbait; "
    "no ALL CAPS, no emoji spam, no hashtags, no false claims or misleading "
    "promises. Prefer around 60 characters, never over 100.\n"
    "DESCRIPTION: plain text that explains what the viewer is seeing, reads "
    "naturally, uses relevant search phrasing without keyword-stuffing, makes "
    "no unsupported claims, and may end with a short call to action. If the "
    "video is an AI-generated visualization, say so instead of implying real "
    "footage. Never invent locations, people, brands, statistics, awards, "
    "events, dates, or links.\n"
    "HASHTAGS: 5-12 short, relevant tags, each starting with '#', no "
    "duplicates, no spam/generic tags (#viral, #trending, #fyp, #explore). "
    "Include #Shorts when appropriate for a YouTube Short.\n"
    "KEYWORDS: a small list of genuinely useful search keywords (subject, "
    "action, format, topic), no duplicates.\n"
    "Return exactly the provided JSON schema with every field populated."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bible_lines(bible: dict) -> str:
    if not bible:
        return "  (none provided)"
    lines = []
    for key, value in bible.items():
        if value is None:
            continue
        label = str(key).replace("_", " ")
        lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def _scene_lines(project: Project) -> str:
    lines = []
    for index, scene in enumerate(project.scenes, start=1):
        clip = scene.video_clip or ""
        lines.append(
            f"  scene {index}: {scene.description}"
            f"{f'  [clip: {Path(clip).name}]' if clip else ''}"
        )
    return "\n".join(lines) if lines else "  (no scenes)"


def _final_video_lines(project: Project) -> str:
    final = project.final_video
    if not final or not Path(final).is_file():
        return "  final video: not yet available"
    try:
        info = probe_video(final)
        orientation = "vertical 9:16" if info.height > info.width else f"{info.width}x{info.height}"
        return (
            f"  final video: {info.duration:.1f}s, {info.width}x{info.height} "
            f"({orientation}), {info.video_codec or 'unknown codec'}"
        )
    except Exception as e:  # pragma: no cover - probe already validated by QC
        return f"  final video: available (probe failed: {e})"


def _qc_lines(project: Project) -> str:
    report = None
    for candidate in reversed(project.qc_reports):
        if candidate.is_current and candidate.assembly_id == project.current_assembly_id:
            report = candidate
            break
    if report is None:
        return "  QC: not run yet (metadata generation does not require it)"
    summary = (report.findings[:2] or ["no findings"]) if report.status.value != "PASSED" else ["all checks passed"]
    return f"  QC: {report.status.value} — " + "; ".join(summary)


def build_user_message(project: Project) -> str:
    """The exact message stored on the record for auditability."""
    settings = project.settings
    return (
        "PLATFORM: YouTube Shorts (vertical 9:16 short-form video)\n"
        f"PROJECT IDEA: {project.idea}\n"
        f"CONTENT TYPE: {settings.content_type or 'n/a'}\n"
        f"VISUAL STYLE: {settings.visual_style or 'n/a'}\n"
        f"TARGET DURATION: {settings.duration_seconds}s\n\n"
        "PROJECT BIBLE (source of truth):\n"
        f"{_bible_lines(project.project_bible)}\n\n"
        "STORYBOARD TITLE/SUMMARY:\n"
        f"  title: {((project.storyboard or {}).get('title') or 'n/a')}\n"
        f"  summary: {((project.storyboard or {}).get('summary') or 'n/a')}\n\n"
        "SCENE SUMMARIES:\n"
        f"{_scene_lines(project)}\n\n"
        "FINAL VIDEO:\n"
        f"{_final_video_lines(project)}\n\n"
        "CURRENT QC RESULT:\n"
        f"{_qc_lines(project)}\n\n"
        "Write the upload package for THIS specific final video. Do not rely on "
        "the project name alone; base every claim on the facts above."
    )


class MetadataAgent:
    """Generates a structured YouTube Shorts upload package for a project."""

    def __init__(
        self,
        *,
        llm=call_structured,
        settings: Optional[Settings] = None,
    ):
        self._llm = llm
        self._settings = settings or get_settings()

    def generate(self, project: Project, *, model: Optional[str] = None) -> MetadataOutput:
        """Ask the LLM for the upload package and validate it strictly."""
        user_message = build_user_message(project)
        try:
            raw = self._llm(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
                json_schema=METADATA_SCHEMA,
                model=model,
            )
        except ProviderError as e:
            raise MetadataError(f"Metadata provider error: {e}")
        output = MetadataOutput.model_validate(raw)
        issues = validate_metadata_output(output, self._settings, strict=True)
        if issues:
            raise MetadataError(
                "LLM returned invalid metadata: " + "; ".join(issues)
            )
        return output

    @staticmethod
    def provider_name(project: Project = None) -> str:
        from backend.providers.registry import get_llm_provider

        try:
            provider = get_llm_provider(get_settings())
            return getattr(provider, "name", "unknown")
        except Exception:
            return "unknown"
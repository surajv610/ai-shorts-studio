"""Schemas + validation for the Metadata Agent's structured output.

``MetadataOutput`` is the JSON the LLM must return for the YouTube Shorts upload
package. It is validated BEFORE anything is persisted: titles/descriptions are
length-capped, hashtags must be legal and unique, keywords must not be stuffed.
``strict`` mode is used at generation time (rejects banned spam tags); manual
user edits (PATCH) use the same validator leniently so the user stays in control.
"""

from collections import Counter
from typing import List, Optional

from pydantic import BaseModel, Field

from backend.config import Settings, get_settings

# Hashtags that are usually spammy / unrelated. V1 refuses to auto-generate them.
BANNED_HASHTAGS = {
    "viral", "trending", "fyp", "foryou", "explore", "viralvideo",
    "mustwatch", "explorepage", "reels",
}


class MetadataOutput(BaseModel):
    """Structured YouTube Shorts upload package returned by the LLM."""

    title: str = Field(description="Concise, accurate, curiosity-driven title (<=100 chars).")
    description: str = Field(description="Plain-text description explaining what the viewer sees.")
    hashtags: List[str] = Field(description="Relevant hashtags (each starting with '#').")
    keywords: List[str] = Field(default_factory=list, description="Internal search-keyword suggestions.")
    category: str = Field(default="", description="Suggested YouTube category (optional).")
    content_summary: str = Field(default="", description="One-line summary of what the video shows.")


METADATA_SCHEMA = {
    "name": "metadata_output",
    "schema": MetadataOutput.model_json_schema(),
}


class MetadataError(Exception):
    """Raised when metadata generation or validation fails."""


def validate_metadata_output(
    output: MetadataOutput,
    settings: Optional[Settings] = None,
    strict: bool = False,
) -> List[str]:
    """Return human-readable issues. Empty list means the package is valid.

    ``strict=True`` (generation): additionally rejects banned hashtags and
    keyword/description stuffing. User edits are validated leniently so the
    user can always override the draft.
    """
    settings = settings or get_settings()
    issues: List[str] = []

    title = output.title.strip()
    if not title:
        issues.append("title is empty")
    elif len(title) > settings.metadata_max_title_length:
        issues.append(
            f"title is {len(title)} chars (max {settings.metadata_max_title_length})"
        )
    if "#" in title or "  " in title or title != title.strip():
        issues.append("title must be natural language without hashtags or extra whitespace")

    description = output.description.strip()
    if not description:
        issues.append("description is empty")
    elif len(description) > settings.metadata_max_description_length:
        issues.append(
            f"description is {len(description)} chars "
            f"(max {settings.metadata_max_description_length})"
        )
    if strict:
        # Cheap keyword-stuffing check: no significant word repeated > 4x.
        words = [w.lower() for w in description.split() if len(w) > 2]
        repeated = {
            w: n for w, n in Counter(words).items() if n > 4
        }
        if repeated:
            issues.append(
                "description repeats words excessively: " + ", ".join(repeated)
            )

    tags: List[str] = []
    for tag in output.hashtags:
        cleaned = tag.strip()
        if not cleaned:
            issues.append("hashtags contains an empty tag")
            continue
        if not cleaned.startswith("#"):
            issues.append(f"hashtag {cleaned!r} does not begin with '#'")
        elif " " in cleaned:
            issues.append(f"hashtag {cleaned!r} must not contain spaces")
        else:
            tags.append(cleaned)
    lowered = [t.lower() for t in tags]
    if len(lowered) != len(set(lowered)):
        issues.append("hashtags contains duplicates")
    if len(tags) > settings.metadata_max_hashtags:
        issues.append(
            f"too many hashtags ({len(tags)}, max {settings.metadata_max_hashtags})"
        )
    if strict:
        banned = sorted({t[1:].lower() for t in tags} & BANNED_HASHTAGS)
        if banned:
            issues.append(
                "generic spam hashtags are not allowed: #" + ", #".join(banned)
            )

    keywords: List[str] = [k.strip() for k in output.keywords if k.strip()]
    if len(output.keywords) > settings.metadata_max_keywords:
        issues.append(
            f"too many keywords ({len(output.keywords)}, max {settings.metadata_max_keywords})"
        )
    lowered_kw = [k.lower() for k in keywords]
    if len(lowered_kw) != len(set(lowered_kw)):
        issues.append("keywords contains duplicates")
    if strict and len(keywords) > 8:
        issues.append("keywords list is too long to stay relevant (keep it small)")

    return issues
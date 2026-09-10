"""LLM client for the agent layer.

Earlier versions of this module read AI_* env vars directly. It now delegates
to the central provider layer in backend/providers so there is a single source
of truth for provider selection. It keeps backward compatibility with the
AI_PROVIDER/AI_API_KEY/AI_MODEL variables via backend.config fallbacks.
"""

from typing import Optional

from backend.config import get_settings
from backend.providers.base import ProviderError
from backend.providers.registry import get_llm_provider

__all__ = ["call_structured", "call_text", "ProviderError"]


def _provider():
    return get_llm_provider(get_settings())


def call_structured(
    system_prompt: str,
    user_message: str,
    json_schema: dict,
    model: Optional[str] = None,
) -> dict:
    """Call the LLM and force a JSON response matching the given schema.

    Returns a plain dict parsed from the model's structured output.
    """
    options = {"model": model} if model else None
    return _provider().generate_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        json_schema=json_schema,
        options=options,
    )


def call_text(prompt: str, model: Optional[str] = None) -> str:
    """Call the LLM with a plain text prompt and return the text reply."""
    options = {"model": model} if model else None
    return _provider().generate_text(prompt=prompt, options=options).text

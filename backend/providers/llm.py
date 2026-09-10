"""LLM providers: OpenAI-compatible client and a mock implementation."""

import json
from typing import Optional

from backend.config import Settings, get_settings
from backend.providers.base import (
    LLMProvider,
    LLMTextResult,
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderNotConfiguredError,
    ProviderQuotaError,
    ProviderRemoteError,
    ProviderSafetyError,
    ProviderUnavailableError,
)
from backend.providers.http import APIClient

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def map_llm_error(status: int, body, method: str, path: str) -> ProviderError:
    """Map OpenAI-style HTTP errors to specific ProviderError subclasses."""
    message = ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("code") or "")
    lowered = message.lower()
    safety = "content_filter" in lowered or "safety" in lowered

    if status in (401, 403):
        return ProviderAuthError(message or f"{method} {path}: authentication failed")
    if status == 429:
        return ProviderQuotaError(message or f"{method} {path}: rate limited")
    if status >= 500:
        return ProviderUnavailableError(message or f"{method} {path}: upstream error")
    if safety:
        return ProviderSafetyError(message)
    return ProviderInvalidRequestError(message or f"{method} {path} returned HTTP {status}")


class OpenAILLMProvider(LLMProvider):
    """LLM provider for OpenAI-compatible chat completions APIs."""

    name = "openai"

    def __init__(self, settings: Optional[Settings] = None, request_func=None):
        self.settings = settings or get_settings()
        api_key = self.settings.llm_api_key.get_secret_value()
        base_url = self.settings.llm_base_url or DEFAULT_LLM_BASE_URL
        self.model = self.settings.llm_model or DEFAULT_LLM_MODEL
        self._client = APIClient(
            base_url=base_url,
            api_key=api_key,
            timeout=self.settings.llm_timeout,
            retry_attempts=self.settings.retry_attempts,
            retry_backoff=self.settings.retry_backoff,
            retry_max_delay=self.settings.retry_max_delay,
            request_func=request_func,
            error_parser=map_llm_error,
        )

    def generate_text(
        self,
        prompt: str,
        options: Optional[dict] = None,
    ) -> LLMTextResult:
        options = options or {}
        model = options.get("model") or self.model
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": options.get("temperature", 0.7),
        }
        data = self._client.post_json("/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderRemoteError(f"Unexpected LLM response shape: {e}")
        return LLMTextResult(text=text, model=model, provider="openai")

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict,
        options: Optional[dict] = None,
    ) -> dict:
        options = options or {}
        model = options.get("model") or self.model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": options.get("temperature", 0.7),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "structured_output"),
                    "strict": True,
                    "schema": json_schema["schema"],
                },
            },
        }
        data = self._client.post_json("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ProviderRemoteError(f"Could not parse LLM structured response: {e}")


def _default_mock_image_prompt() -> dict:
    """A valid Image-Agent structured prompt used when running in mock mode."""
    return {
        "global_continuity": (
            "Same single plant on identical dark fertile soil, locked-down macro "
            "camera, warm golden-hour side lighting, photorealistic 85mm macro "
            "look, rich greens and warm highlights, no people, no text, no watermarks."
        ),
        "scene_action": "The plant visibly progresses to its next growth step.",
        "prompt_a": (
            "Vertical 9:16 macro close-up of the plant at this scene's growth "
            "step, on the same dark fertile soil, same locked-down camera and "
            "warm golden-hour side lighting, photorealistic 85mm macro look, "
            "rich greens with warm highlights, no people, no text, no watermarks."
        ),
        "prompt_b": (
            "Vertical 9:16 three-quarter angle of the plant at this scene's "
            "growth step, same soil, same lighting and photorealistic 85mm "
            "macro look, alternate framing from the left with a slightly wider "
            "crop, no people, no text, no watermarks."
        ),
        "composition_notes": "B uses a wider three-quarter angle vs A's direct close-up.",
    }


def _default_mock_structured() -> dict:
    """A plausible structured storyboard used when running in mock mode."""
    return {
        "title": "Seed to Bloom",
        "summary": "A single seed germinates, grows, buds, and flowers.",
        "scenes": [
            {
                "scene_id": 1,
                "scene_number": 1,
                "title": "Dormant Seed",
                "visual_description": "Close-up of a seed on dark soil, warm light.",
                "action": "Static macro shot.",
                "estimated_duration": 6.0,
                "continuity_notes": "Fixed camera, consistent lighting.",
            },
            {
                "scene_id": 2,
                "scene_number": 2,
                "title": "Sprouting",
                "visual_description": "A green sprout emerges upward.",
                "action": "Slow push-in.",
                "estimated_duration": 6.0,
                "continuity_notes": "Same framing.",
            },
            {
                "scene_id": 3,
                "scene_number": 3,
                "title": "Growing",
                "visual_description": "The plant grows taller with more leaves.",
                "action": "Camera tilts up.",
                "estimated_duration": 6.0,
                "continuity_notes": "Growth is monotonic.",
            },
            {
                "scene_id": 4,
                "scene_number": 4,
                "title": "Budding",
                "visual_description": "A bud forms at the top.",
                "action": "Slow drift toward bud.",
                "estimated_duration": 6.0,
                "continuity_notes": "Bud only after leaves.",
            },
            {
                "scene_id": 5,
                "scene_number": 5,
                "title": "Flowering",
                "visual_description": "The bud opens into a vibrant bloom.",
                "action": "Slow push-in.",
                "estimated_duration": 6.0,
                "continuity_notes": "Flower fully open by end.",
            },
        ],
        "project_bible": {
            "subject": "A single flowering plant grown from seed",
            "environment": "Macro on dark fertile soil",
            "time_lighting": "Warm golden-hour side lighting",
            "camera": "Locked-down tripod macro",
            "lens_look": "85mm macro, shallow depth of field",
            "visual_style": "Photorealistic",
            "color_appearance": "Rich greens to vibrant petals",
            "physical_characteristics": "One seed to a single-stem bloom",
            "continuity_rules": "Fixed framing, monotonic growth",
            "negative_constraints": "No people, text, or watermarks",
            "aspect_ratio": "9:16",
            "target_duration": 30.0,
        },
    }


def _default_mock_video_prompt() -> dict:
    """A valid Video-Agent structured prompt used when running in mock mode."""
    return {
        "global_continuity": (
            "Same single plant on identical dark fertile soil, locked-down macro "
            "camera, warm golden-hour side lighting, photorealistic 85mm macro "
            "look, rich greens and warm highlights; nothing else may change."
        ),
        "scene_action": (
            "The plant visibly progresses to its next growth step from the "
            "source image."
        ),
        "motion": "Very slow, continuous, monotonic growth only.",
        "environment": (
            "Identical dark fertile soil, no background or environment changes."
        ),
        "camera": "Completely locked-down macro camera, zero camera movement.",
        "lighting": "Same warm golden-hour side lighting throughout.",
        "timelapse": "Short growth timelapse, strictly chronological.",
        "negative_constraints": "No people, no text, no watermarks, no flicker, no camera shake.",
        "prompt": (
            "Vertical 9:16 macro timelapse of the same single plant on the same "
            "dark fertile soil progressing to its next growth step, camera "
            "locked down with zero movement, warm golden-hour side lighting, "
            "photorealistic 85mm macro look, rich greens with warm highlights, "
            "nothing else changes, no people, no text, no flicker."
        ),
    }


def _default_mock_metadata() -> dict:
    """A valid Metadata-Agent structured output used when running in mock mode.

    Example only — it is generated from the actual project facts at runtime and
    never hard-coded into the agent.
    """
    return {
        "title": "From Seed to Flower in One Timelapse 🌱",
        "description": (
            "Watch a seed transform through the stages of growth, from "
            "germination to a flowering plant, captured as a short vertical "
            "timelapse.\n\nA simple look at plant growth from beginning to "
            "bloom.\n\nFollow for more short visual experiments."
        ),
        "hashtags": ["#PlantGrowth", "#Timelapse", "#SeedToPlant", "#Nature", "#Shorts"],
        "keywords": [
            "seed germination",
            "plant growing timelapse",
            "seed to flower",
            "nature short",
            "vertical timelapse",
        ],
        "category": "Science & Technology",
        "content_summary": (
            "A vertical timelapse of a single seed growing into a flowering plant."
        ),
    }


class MockLLMProvider(LLMProvider):
    """In-memory LLM provider for development without API costs.

    Provides deterministic structured output so the full workflow can run.
    """

    name = "mock"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model = self.settings.llm_model or "mock-model"

    def generate_text(
        self,
        prompt: str,
        options: Optional[dict] = None,
    ) -> LLMTextResult:
        return LLMTextResult(
            text=f"(mock) Received prompt of {len(prompt)} chars.",
            model=self.model,
            provider="mock",
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict,
        options: Optional[dict] = None,
    ) -> dict:
        if json_schema.get("name") == "image_prompt_output":
            return _default_mock_image_prompt()
        if json_schema.get("name") == "video_prompt_output":
            return _default_mock_video_prompt()
        if json_schema.get("name") == "metadata_output":
            return _default_mock_metadata()
        return _default_mock_structured()

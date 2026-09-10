"""Provider registry and factory.

Selects concrete provider implementations based on environment configuration
(e.g. IMAGE_PROVIDER=mock). This is the single place that maps a provider
name string to an implementation, so agent code never concerns itself with a
specific vendor.

Also provides:
  * configuration validation (raise a clear ProviderNotConfiguredError when a
    real provider is selected but credentials/config are missing),
  * capability introspection (ask whether the configured provider supports a
    requested capability without instantiating a vendor adapter),
  * provider summary used by the settings endpoint (no secrets).
"""

from typing import Optional

from backend.config import Settings, get_settings
from backend.providers.base import (
    ImageProvider,
    LLMProvider,
    ProviderCapabilities,
    ProviderCapabilityError,
    ProviderNotConfiguredError,
    VideoProvider,
)
from backend.providers.gemini import GeminiLLMProvider
from backend.providers.google_image import GoogleImageProvider
from backend.providers.google_veo import GoogleVeoProvider
from backend.providers.image import MockImageProvider
from backend.providers.llm import MockLLMProvider, OpenAILLMProvider
from backend.providers.video import MockVideoProvider

# Supported provider names per capability.
SUPPORTED_LLM_PROVIDERS = {"gemini", "openai", "mock"}
SUPPORTED_IMAGE_PROVIDERS = {"mock", "google"}
SUPPORTED_VIDEO_PROVIDERS = {"mock", "google"}


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLMProvider(settings)
    if provider == "gemini":
        return GeminiLLMProvider(settings)
    if provider == "openai":
        if not settings.llm_api_key.get_secret_value():
            raise ProviderNotConfiguredError(
                "LLM_PROVIDER=openai but LLM_API_KEY is not set. "
                "Set LLM_API_KEY (and optionally LLM_MODEL / LLM_BASE_URL) or "
                "use LLM_PROVIDER=mock for development."
            )
        return OpenAILLMProvider(settings)
    raise ProviderNotConfiguredError(
        f"Unsupported LLM_PROVIDER={provider!r}. Supported: {sorted(SUPPORTED_LLM_PROVIDERS)}"
    )


def get_image_provider(settings: Optional[Settings] = None) -> ImageProvider:
    settings = settings or get_settings()
    provider = settings.image_provider
    if provider == "mock":
        return MockImageProvider(settings)
    if provider == "google":
        if not settings.image_api_key.get_secret_value():
            raise ProviderNotConfiguredError(
                "IMAGE_PROVIDER=google but no API key is set. "
                "Set IMAGE_API_KEY (or GEMINI_API_KEY) and optionally "
                "IMAGE_MODEL / IMAGE_BASE_URL, or use IMAGE_PROVIDER=mock "
                "for development."
            )
        return GoogleImageProvider(settings)
    raise ProviderNotConfiguredError(
        f"Unsupported IMAGE_PROVIDER={provider!r}. Supported: {sorted(SUPPORTED_IMAGE_PROVIDERS)}"
    )


def get_video_provider(settings: Optional[Settings] = None) -> VideoProvider:
    settings = settings or get_settings()
    provider = settings.video_provider
    if provider == "mock":
        return MockVideoProvider(settings)
    if provider == "google":
        if not settings.video_api_key.get_secret_value():
            raise ProviderNotConfiguredError(
                "VIDEO_PROVIDER=google but no API key is set. "
                "Set VIDEO_API_KEY (or GEMINI_API_KEY) and optionally "
                "VIDEO_MODEL / VIDEO_BASE_URL, or use VIDEO_PROVIDER=mock "
                "for development."
            )
        return GoogleVeoProvider(settings)
    raise ProviderNotConfiguredError(
        f"Unsupported VIDEO_PROVIDER={provider!r}. Supported: {sorted(SUPPORTED_VIDEO_PROVIDERS)}"
    )


def provider_summary(settings: Optional[Settings] = None) -> dict:
    """Human-readable summary of which providers are active (no secrets)."""
    settings = settings or get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_configured": settings.llm_configured,
        "image_provider": settings.image_provider,
        "image_configured": settings.image_configured,
        "video_provider": settings.video_provider,
        "video_configured": settings.video_configured,
    }


# -------------------------------------------------------------------------
# Capability introspection
# -------------------------------------------------------------------------


def provider_capabilities(
    kind: str, settings: Optional[Settings] = None
) -> ProviderCapabilities:
    """Return the declared capabilities for the configured provider.

    ``kind`` is one of "image", "video", "llm". Returns an empty capabilities
    object for unknown providers so callers can still check safely.
    """
    settings = settings or get_settings()
    if kind == "image":
        p = get_image_provider(settings)
    elif kind == "video":
        p = get_video_provider(settings)
    elif kind == "llm":
        try:
            p = get_llm_provider(settings)
        except ProviderNotConfiguredError:
            return ProviderCapabilities(name=settings.llm_provider, generates="text")
    else:
        return ProviderCapabilities(name="unknown")
    caps = getattr(p, "capabilities", None)
    return caps() if callable(caps) else ProviderCapabilities(name=p.name)


def provider_supports(
    kind: str, capability: str, settings: Optional[Settings] = None
) -> bool:
    """Return True if the configured provider supports a capability.

    Example::
        provider_supports("image", "aspect_ratio_9_16")

    Raises ProviderCapabilityError if the provider itself is not configured,
    so callers get a clear signal early rather than a generic failure later.
    """
    settings = settings or get_settings()
    caps = provider_capabilities(kind, settings)
    return caps.supports(capability)

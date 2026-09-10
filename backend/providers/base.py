"""Base types and interfaces shared by all generation providers.

This module defines:

* Status enums shared by synchronous and asynchronous generation.
* Standardized generation options (dimensions, aspect ratio, model, params).
* Capability metadata so callers can ask "does this provider support X?".
* Generic asynchronous job abstraction (submit -> poll -> result) so providers
  that require polling can be supported without changing agent code.
* Abstract provider interfaces for LLM, image, and video generation.
* A provider-error hierarchy, including provider-specific error information.

The actual provider adapters (OpenAI image, Runway, Luma, etc.) will be added
later. This module only defines the contracts and the mock-safe primitives.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


# =========================================================================
# Enums
# =========================================================================


class GenerationStatus(str, Enum):
    """Status of a synchronous generation result."""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class AsyncJobStatus(str, Enum):
    """Status of an asynchronous generation job (submit -> poll -> result)."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# =========================================================================
# Provider-specific errors
# =========================================================================


class ProviderError(Exception):
    """Base error for any provider failure."""


class ProviderNotConfiguredError(ProviderError):
    """Raised when a provider is selected but its credentials are missing."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderRemoteError(ProviderError):
    """Raised when a provider returns an error status."""


class ProviderValidationError(ProviderError):
    """Raised when a requested combination of options/capabilities is invalid."""


class ProviderCapabilityError(ProviderError):
    """Raised when the configured provider does not support a requested capability."""


class ProviderAuthError(ProviderError):
    """Raised when the provider rejects our credentials (401/403)."""


class ProviderInvalidRequestError(ProviderError):
    """Raised when the request itself is malformed or rejected (4xx client errors)."""


class ProviderQuotaError(ProviderError):
    """Raised when the provider reports quota exhaustion or rate limiting (429)."""


class ProviderRateLimitError(ProviderQuotaError):
    """A TRANSIENT rate limit (safe to retry after backoff).

    Carries ``category="rate_limit"`` and a public ``reason`` like
    ``"GEMINI_RATE_LIMITED"``. Callers can distinguish a temporary rate limit
    from quota exhaustion via ``isinstance(exc, ProviderQuotaExceededError)``.
    """

    category = "rate_limit"
    reason = "RATE_LIMITED"


class ProviderQuotaExceededError(ProviderQuotaError):
    """Quota exhausted (billing/resource) — NOT transient, do NOT retry.

    Carries ``category="quota_exhausted"`` and a public ``reason`` like
    ``"GEMINI_QUOTA_EXCEEDED"``. The application must stop and surface this to
    the user instead of silently switching providers.
    """

    category = "quota_exhausted"
    reason = "QUOTA_EXCEEDED"


class ProviderSafetyError(ProviderError):
    """Raised when the provider blocks / filters content for safety reasons."""


class ProviderUnavailableError(ProviderError):
    """Raised when the provider service is down/unreachable (5xx, network)."""


class ProviderGenerationError(ProviderError):
    """Raised when generation was submitted but the provider failed or returned nothing."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =========================================================================
# Generation options
# =========================================================================


class GenerationOptions(BaseModel):
    """Standardized, provider-agnostic options for a generation request.

    These are the option *names* used across adapters. Different providers map
    these to their own fields. ``params`` carries vendor-specific extras that a
    particular adapter understands.
    """

    model: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[str] = None  # e.g. "9:16", "1:1", "16:9"
    duration_seconds: Optional[int] = None
    prompt: Optional[str] = None  # forwards animation/prompt refinements
    quality: Optional[str] = None  # e.g. "standard", "hd", "auto"
    seed: Optional[int] = None
    params: Dict[str, object] = Field(default_factory=dict)

    def aspect_ratio_key(self) -> str:
        """Normalize an aspect ratio to a capability key like ``9_16``.

        Returns an empty string if no usable aspect ratio is available. Non-2D
        ratios degrade gracefully; e.g. "9:16" -> "9_16".
        """
        raw = (self.aspect_ratio or "").strip()
        if not raw:
            return ""
        # Accept "9:16", "9 / 16", "9x16", "0.9", etc. Keep only digits/underscore.
        parts = [p for p in raw.replace("x", ":").replace("/", ":").split(":") if p.strip()]
        if len(parts) >= 2 and all(p.isdigit() for p in parts):
            return f"{int(parts[0])}_{int(parts[1])}"
        return ""

    def aspect_ratio_value(self) -> Optional[str]:
        """Return a normalized ``"W:H"`` representation, e.g. ``"9:16"``.

        This is the format most providers (Gemini image + Veo) expect directly.
        Returns ``None`` when no usable ratio was provided.
        """
        key = self.aspect_ratio_key()
        return key.replace("_", ":") if key else None


# =========================================================================
# Capability metadata
# =========================================================================


class ProviderCapabilities(BaseModel):
    """Declarative capability metadata for a provider.

    Each capability is a lower_case string key, e.g. ``"text_to_image"``,
    ``"multi_candidate"``, ``"aspect_ratio_9_16"``, ``"image_to_video"``,
    ``"duration_control"``. Optional descriptive fields aid reporting and UI.
    """

    name: str
    generates: str = ""  # "image" | "video" | "text" | ""
    capabilities: Set[str] = Field(default_factory=set)
    supports_async: bool = False
    description: str = ""
    models: List[str] = Field(default_factory=list)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def supported_ratios(self) -> List[str]:
        return sorted(c for c in self.capabilities if c.startswith("aspect_ratio_"))


# -------------------------------------------------------------------------
# Common capability keys (documented, so adapters can declare them)
# -------------------------------------------------------------------------

# Image capabilities
CAP_TEXT_TO_IMAGE = "text_to_image"
CAP_MULTI_CANDIDATE = "multi_candidate"
CAP_ASYNC = "async"

# Generic none-specific capability used to check async support reading from
# the provider-level flag; kept here for documentation/detection helpers.


class ProviderCapabilitiesMixin:
    """Provides ``capabilities()`` and ``supports()`` helpers on providers.

    Subclasses declare a ``_capabilities`` attribute (a ``ProviderCapabilities``
    or a plain dict). The ``supports`` helper lets callers check, without
    knowing the concrete adapter, whether a capability is available.
    """

    _capabilities: ProviderCapabilities = ProviderCapabilities(name="base")

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def supports(self, capability: str) -> bool:
        return self._capabilities.supports(capability)


# =========================================================================
# Result models
# =========================================================================


class GenerationResult(BaseModel):
    """Common fields for any (synchronous) generation result."""

    generation_id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    model: str = ""
    prompt: str = ""
    status: GenerationStatus = GenerationStatus.PENDING
    error: Optional[str] = None
    provider_error: Optional[Dict[str, object]] = None
    created_at: str = Field(default_factory=_now)
    raw: Optional[Dict[str, object]] = None


class ImageGenerationResult(GenerationResult):
    """Result of generating one or more candidate images for a scene."""

    images: List[str] = Field(
        default_factory=list,
        description="Local paths/references to generated images.",
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Async job id if this result was produced via polling, else None.",
    )


class VideoGenerationResult(GenerationResult):
    """Result of generating one video clip from an image."""

    video: Optional[str] = Field(
        default=None, description="Local path/reference to the generated clip."
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Async job id if this result was produced via polling, else None.",
    )


class LLMTextResult(BaseModel):
    text: str
    model: str = ""
    provider: str = ""
    generation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=_now)


# =========================================================================
# Generic asynchronous job abstraction
# =========================================================================


class AsyncJob(BaseModel):
    """A submitted asynchronous job, returned after ``submit_*``.

    Carries a provider-agnostic ``job_id`` plus provider-specific state under
    ``raw``. Use ``AsyncPollingClient.poll()`` (or a concrete async-capable
    provider) to advance to a terminal status.
    """

    provider: str
    job_id: str
    status: AsyncJobStatus = AsyncJobStatus.PENDING
    error: Optional[str] = None
    provider_error: Optional[Dict[str, object]] = None
    created_at: str = Field(default_factory=_now)
    raw: Optional[Dict[str, object]] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (AsyncJobStatus.SUCCEEDED, AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED)


class AsyncPollingClient:
    """Generic async-job controller for providers that submit then poll.

    This is the abstraction that lets providers with asynchronous generation
    (submit -> job id -> poll status -> result) integrate without the agents
    knowing anything about polling.

    Concrete async-capable providers implement ``submit_*`` (returning an
    ``AsyncJob``) and ``poll_*`` (returning an updated ``AsyncJob``). The
    `wait` helpers loop over ``poll`` until a terminal state is reached.
    """

    def submit_async(
        self,
        kind: str,
        prompt: str,
        options: Optional[GenerationOptions] = None,
    ) -> AsyncJob:
        raise NotImplementedError

    def poll_async(self, job: AsyncJob) -> AsyncJob:
        raise NotImplementedError

    def wait(
        self,
        job: AsyncJob,
        *,
        timeout_seconds: float = 600.0,
        poll_interval: float = 2.0,
    ) -> AsyncJob:
        """Poll until the job reaches a terminal state (or timeout)."""
        import time

        started = time.monotonic()
        while time.monotonic() - started < timeout_seconds:
            job = self.poll_async(job)
            if job.is_terminal:
                return job
            time.sleep(poll_interval)
        job.status = AsyncJobStatus.FAILED
        job.error = f"Async job timed out after {timeout_seconds}s"
        return job


class AsyncImageProvider(ProviderCapabilitiesMixin):
    """Optional interface for async image providers.

    A provider that needs polling implements this *in addition to* (or instead
    of) ``ImageProvider``. Agents that need a result can either call
    ``generate_images`` (sync) or use ``submit_async``/``poll_async``/``wait``.
    """

    name = "base-async-image"

    def submit_async(
        self,
        prompt: str,
        options: Optional[GenerationOptions] = None,
        count: int = 2,
    ) -> AsyncJob:
        raise NotImplementedError

    def poll_async(
        self,
        job: AsyncJob,
        options: Optional[GenerationOptions] = None,
    ) -> AsyncJob:
        raise NotImplementedError


class AsyncVideoProvider(ProviderCapabilitiesMixin):
    """Optional interface for async video providers (image-to-video)."""

    name = "base-async-video"

    def submit_async(
        self,
        image_reference: str,
        prompt: str,
        options: Optional[GenerationOptions] = None,
    ) -> AsyncJob:
        raise NotImplementedError

    def poll_async(
        self,
        job: AsyncJob,
        options: Optional[GenerationOptions] = None,
    ) -> AsyncJob:
        raise NotImplementedError


# =========================================================================
# Provider interfaces (abstract, sync variants)
# =========================================================================


class LLMProvider:
    """Interface for language model providers."""

    name = "base"

    def generate_text(self, prompt: str, options: Optional[dict] = None) -> LLMTextResult:
        raise NotImplementedError

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict,
        options: Optional[dict] = None,
    ) -> dict:
        raise NotImplementedError


class ImageProvider(ProviderCapabilitiesMixin):
    """Interface for image generation providers.

    Supports:
      * text-to-image                         (prompt)
      * multiple candidates                   (count)
      * dimensions / aspect ratio             (options.width/height/aspect_ratio)
      * model selection                       (options.model)
      * generation parameters                 (options.quality/seed/params)
      * async generation / polling            (see AsyncImageProvider)
      * generation ID / status / output refs  (ImageGenerationResult)
      * provider-specific errors              (provider_error on result)
    """

    name = "base"

    def generate_images(
        self,
        prompt: str,
        options: Optional[GenerationOptions] = None,
        count: int = 2,
    ) -> ImageGenerationResult:
        raise NotImplementedError


class VideoProvider(ProviderCapabilitiesMixin):
    """Interface for image-to-video generation providers.

    Supports:
      * image-to-video generation             (image_reference + prompt)
      * animation prompt                      (prompt)
      * duration                              (options.duration_seconds)
      * aspect ratio                          (options.aspect_ratio)
      * model selection                       (options.model)
      * async generation / polling            (see AsyncVideoProvider)
      * generation ID / status / output ref   (VideoGenerationResult)
      * provider-specific errors              (provider_error on result)
    """

    name = "base"

    def generate_video(
        self,
        image_reference: str,
        prompt: str,
        options: Optional[GenerationOptions] = None,
    ) -> VideoGenerationResult:
        raise NotImplementedError


# Backwards-aliases used by earlier code that passed plain dicts for options.
# The mock providers accept either a GenerationOptions or a plain dict.
def coerce_options(options: Optional[dict]) -> GenerationOptions:
    """Accept a plain dict or GenerationOptions and return GenerationOptions."""
    if options is None:
        return GenerationOptions()
    if isinstance(options, GenerationOptions):
        return options
    return GenerationOptions(
        model=options.get("model"),
        width=options.get("width"),
        height=options.get("height"),
        aspect_ratio=options.get("aspect_ratio"),
        duration_seconds=options.get("duration_seconds"),
        prompt=options.get("prompt"),
        quality=options.get("quality"),
        seed=options.get("seed"),
        params=options.get("params", {}),
    )

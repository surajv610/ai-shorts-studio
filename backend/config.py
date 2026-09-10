"""Environment configuration for AI Shorts Studio.

All secrets are read from the environment only and are surfaced as
pydantic SecretStr values so they are never accidentally serialized or
logged. Nothing here ever sends secrets to the frontend.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import SecretStr


class Settings:
    """Reads and validates configuration from environment variables.

    Provider credentials use these variables:
      LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_TIMEOUT
      GEMINI_API_KEY (shared Gemini key for the gemini LLM provider and the
      Google image/video providers)
      IMAGE_PROVIDER, IMAGE_API_KEY, IMAGE_MODEL, IMAGE_BASE_URL
      VIDEO_PROVIDER, VIDEO_API_KEY, VIDEO_MODEL, VIDEO_BASE_URL
      DATABASE_URL
      STORAGE_PATH
      RETRY_ATTEMPTS, RETRY_BACKOFF, RETRY_MAX_DELAY
    """

    def __init__(self, env: Optional[dict] = None):
        self._env = env if env is not None else os.environ

    def _get(self, name: str, default: str = "") -> str:
        return self._env.get(name, default).strip()

    def _get_float(self, name: str, default: float) -> float:
        raw = self._get(name, "")
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    # --- LLM ---
    @property
    def llm_provider(self) -> str:
        return self._get("LLM_PROVIDER", "mock").lower()

    @property
    def llm_api_key(self) -> SecretStr:
        # OpenAI-style key (and the legacy AI_API_KEY alias). Gemini uses
        # ``gemini_api_key`` (GEMINI_API_KEY), with this value as a fallback for
        # backward compatibility.
        return SecretStr(self._get("LLM_API_KEY", self._get("AI_API_KEY")))

    @property
    def gemini_api_key(self) -> SecretStr:
        # Gemini key shared across LLM/image/video Google providers. Honors the
        # legacy GOOGLE_API_KEY / GEMINI_API_KEY aliases.
        return SecretStr(
            self._get("GEMINI_API_KEY", self._get("GOOGLE_API_KEY"))
        )

    @property
    def llm_model(self) -> str:
        """Configured model, or the provider's default fallback.

        The default is a documented fallback only — model availability varies
        by API account and quota, so users can (and should) set LLM_MODEL for
        their current quota/plan.
        """
        model = self._get("LLM_MODEL", self._get("AI_MODEL"))
        if model:
            return model
        if self.llm_provider == "gemini":
            return "gemini-2.5-flash"
        return "gpt-4o-mini"

    @property
    def llm_base_url(self) -> str:
        return self._get("LLM_BASE_URL", self._get("AI_BASE_URL"))

    @property
    def llm_timeout(self) -> float:
        try:
            return float(self._get("LLM_TIMEOUT", self._get("AI_TIMEOUT", "120")))
        except ValueError:
            return 120.0

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "mock":
            return True
        key = self.llm_api_key.get_secret_value()
        if self.llm_provider == "gemini":
            key = key or self.gemini_api_key.get_secret_value()
        return bool(key)

    @property
    def llm_credentialed(self) -> bool:
        """True when a credential is actually present (or the mock is active).

        Reported distinctly from ``llm_configured`` because for the real
        providers both indicate a usable configuration, while the UI wants a
        dedicated top-level signal that no API key is missing.
        """
        if self.llm_provider == "mock":
            return True
        key = self.llm_api_key.get_secret_value()
        if self.llm_provider == "gemini":
            key = key or self.gemini_api_key.get_secret_value()
        return bool(key)

    # --- Image ---
    @property
    def image_provider(self) -> str:
        return self._get("IMAGE_PROVIDER", "mock").lower()

    @property
    def image_api_key(self) -> SecretStr:
        # Google providers accept IMAGE_GEMINI key, with GEMINI_API_KEY fallback.
        return SecretStr(self._get("IMAGE_API_KEY", self._get("GEMINI_API_KEY", self._get("GOOGLE_API_KEY"))))

    @property
    def image_model(self) -> str:
        return self._get("IMAGE_MODEL", "")

    @property
    def image_base_url(self) -> str:
        return self._get("IMAGE_BASE_URL", "")

    @property
    def image_timeout(self) -> float:
        return self._get_float("IMAGE_TIMEOUT", 120.0)

    @property
    def image_configured(self) -> bool:
        return bool(self.image_api_key.get_secret_value()) or self.image_provider == "mock"

    # --- Video ---
    @property
    def video_provider(self) -> str:
        return self._get("VIDEO_PROVIDER", "mock").lower()

    @property
    def video_api_key(self) -> SecretStr:
        return SecretStr(self._get("VIDEO_API_KEY", self._get("GEMINI_API_KEY", self._get("GOOGLE_API_KEY"))))

    @property
    def video_model(self) -> str:
        return self._get("VIDEO_MODEL", "")

    @property
    def video_base_url(self) -> str:
        return self._get("VIDEO_BASE_URL", "")

    @property
    def video_timeout(self) -> float:
        return self._get_float("VIDEO_TIMEOUT", 60.0)

    @property
    def video_poll_timeout(self) -> float:
        return self._get_float("VIDEO_POLL_TIMEOUT", 1800.0)

    @property
    def video_poll_interval(self) -> float:
        return self._get_float("VIDEO_POLL_INTERVAL", 10.0)

    @property
    def video_configured(self) -> bool:
        return bool(self.video_api_key.get_secret_value()) or self.video_provider == "mock"

    # --- Database / storage ---
    @property
    def database_url(self) -> str:
        return self._get("DATABASE_URL", "")

    @property
    def storage_path(self) -> str:
        return self._get("STORAGE_PATH", "")

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url)

    # --- Retry ---
    @property
    def retry_attempts(self) -> int:
        try:
            return max(1, int(self._get("RETRY_ATTEMPTS", "3")))
        except ValueError:
            return 3

    @property
    def retry_backoff(self) -> float:
        try:
            return float(self._get("RETRY_BACKOFF", "1.0"))
        except ValueError:
            return 1.0

    @property
    def retry_max_delay(self) -> float:
        try:
            return float(self._get("RETRY_MAX_DELAY", "30.0"))
        except ValueError:
            return 30.0

    # --- QC ---
    @property
    def qc_min_duration(self) -> float:
        return max(0.1, self._get_float("QC_MIN_DURATION", 1.0))

    @property
    def qc_max_duration(self) -> float:
        return max(self.qc_min_duration, self._get_float("QC_MAX_DURATION", 600.0))

    @property
    def qc_min_width(self) -> int:
        try:
            return max(2, int(self._get("QC_MIN_WIDTH", "320")))
        except ValueError:
            return 320

    @property
    def qc_min_height(self) -> int:
        try:
            return max(2, int(self._get("QC_MIN_HEIGHT", "568")))
        except ValueError:
            return 568

    @property
    def qc_target_width(self) -> int:
        try:
            return max(0, int(self._get("QC_TARGET_WIDTH", "0")))
        except ValueError:
            return 0

    @property
    def qc_target_height(self) -> int:
        try:
            return max(0, int(self._get("QC_TARGET_HEIGHT", "0")))
        except ValueError:
            return 0

    @property
    def qc_min_fps(self) -> float:
        return max(0.1, self._get_float("QC_MIN_FPS", 1.0))

    @property
    def qc_max_fps(self) -> float:
        return max(self.qc_min_fps, self._get_float("QC_MAX_FPS", 240.0))

    @property
    def qc_min_file_size_bytes(self) -> int:
        try:
            return max(0, int(self._get("QC_MIN_FILE_SIZE", "10000")))
        except ValueError:
            return 10000

    @property
    def qc_duration_tolerance(self) -> float:
        """Seconds of slack before the duration mismatch is a warning."""
        return max(0.0, self._get_float("QC_DURATION_TOLERANCE", 2.0))

    @property
    def qc_duration_warn_tolerance(self) -> float:
        """Seconds of slack before the duration mismatch becomes a failure."""
        return max(self.qc_duration_tolerance, self._get_float("QC_DURATION_WARN_TOLERANCE", 10.0))

    @property
    def qc_visual_frames(self) -> int:
        try:
            return max(2, int(self._get("QC_VISUAL_FRAMES", "8")))
        except ValueError:
            return 8

    @property
    def qc_black_luma(self) -> float:
        """Mean luma threshold below which a sampled frame counts as black."""
        return self._get_float("QC_BLACK_LUMA", 10.0)

    @property
    def qc_freeze_delta(self) -> float:
        """Mean per-pixel frame delta below which a video looks frozen."""
        return self._get_float("QC_FREEZE_DELTA", 1.0)

    # --- Metadata ---
    @property
    def metadata_max_title_length(self) -> int:
        """YouTube-safe title length cap (default well under YouTube's 100)."""
        try:
            return max(1, int(self._get("METADATA_MAX_TITLE_LENGTH", "100")))
        except ValueError:
            return 100

    @property
    def metadata_max_description_length(self) -> int:
        try:
            return max(1, int(self._get("METADATA_MAX_DESCRIPTION_LENGTH", "2000")))
        except ValueError:
            return 2000

    @property
    def metadata_max_hashtags(self) -> int:
        try:
            return max(1, int(self._get("METADATA_MAX_HASHTAGS", "12")))
        except ValueError:
            return 12

    @property
    def metadata_max_keywords(self) -> int:
        try:
            return max(1, int(self._get("METADATA_MAX_KEYWORDS", "10")))
        except ValueError:
            return 10

    @property
    def metadata_block_on_qc_failed(self) -> bool:
        """Default: block metadata generation when the current QC is FAILED.

        Set METADATA_BLOCK_ON_QC_FAILED=false to allow it anyway (the caller
        must explicitly surface the QC FAILED warning to the user).
        """
        raw = self._get("METADATA_BLOCK_ON_QC_FAILED", "true").lower()
        return raw in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance for the process."""
    return Settings()


def provider_config_status(provider: str, configured: bool) -> str:
    """Human-readable config status for an optional provider (mock counts)."""
    if provider == "mock":
        return "READY (mock)"
    return "READY" if configured else "NOT_CONFIGURED"

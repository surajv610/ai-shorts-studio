"""System health and configuration reporting.

The health endpoint reports the state of each dependency WITHOUT exposing any
secrets. Provider API keys are never included in the response.

Dependency states are one of:

NOT_CONFIGURED   A real provider is selected but required credentials/config are missing.
CONFIGURED       A supported provider is selected and configured (credentials present)
                 but has not been verified by a live request yet.
READY            The dependency is fully operational (mock providers, storage, ffmpeg).
ERROR            The provider/dependency is unsupported or failed.
"""

from pathlib import Path

from backend.config import get_settings
from backend.ffmpeg import get_ffmpeg_info
from backend.providers.registry import SUPPORTED_LLM_PROVIDERS, SUPPORTED_IMAGE_PROVIDERS, SUPPORTED_VIDEO_PROVIDERS

# Possible dependency states.
NOT_CONFIGURED = "NOT_CONFIGURED"
CONFIGURED = "CONFIGURED"
READY = "READY"
ERROR = "ERROR"


def _status_for_provider(provider: str, configured: bool, supported: set) -> str:
    if provider == "mock":
        return READY
    if provider not in supported:
        return ERROR
    # A supported, non-mock provider that is configured but not yet live-verified
    # is reported as CONFIGURED (distinct from the operational READY state).
    return CONFIGURED if configured else NOT_CONFIGURED


def _storage_status() -> tuple[str, str]:
    from backend import storage

    try:
        storage.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        test = storage.STORAGE_DIR / ".healthcheck"
        test.write_text("ok")
        test.unlink()
        return READY, str(storage.STORAGE_DIR)
    except OSError as e:
        return ERROR, str(e)


def _database_status() -> str:
    settings = get_settings()
    if not settings.database_url:
        return NOT_CONFIGURED
    # The current V1 implementation persists projects to JSON files, so a
    # DATABASE_URL is optional. If one is set we still report READY here
    # because the JSON store remains authoritative for now.
    return READY


def _overall_status(dependencies: dict, storage_status: str) -> str:
    """Derive an overall READY/CONFIGURED/NOT_CONFIGURED/ERROR summary."""
    if ERROR in dependencies.values():
        return ERROR
    # If storage isn't operational we can't run the pipeline.
    if storage_status != READY:
        return storage_status
    # If every provider is operational (mock) -> READY.
    dep_values = set(dependencies.values())
    if dep_values <= {READY} | {NOT_CONFIGURED}:
        # readiness requires at least storage + ffmpeg operational
        if dependencies.get("storage") == READY and dependencies.get("ffmpeg") == READY:
            return READY
        return NOT_CONFIGURED
    # Some provider is CONFIGURED but not yet verified.
    if CONFIGURED in dep_values:
        return CONFIGURED
    return NOT_CONFIGURED


def get_health() -> dict:
    """Build the full health payload. Contains no secrets."""
    settings = get_settings()

    storage_status, storage_detail = _storage_status()

    ffmpeg = get_ffmpeg_info()
    ffmpeg_status = READY if ffmpeg.available else NOT_CONFIGURED

    llm_status = _status_for_provider(
        settings.llm_provider, settings.llm_configured, SUPPORTED_LLM_PROVIDERS
    )
    image_status = _status_for_provider(
        settings.image_provider, settings.image_configured, SUPPORTED_IMAGE_PROVIDERS
    )
    video_status = _status_for_provider(
        settings.video_provider, settings.video_configured, SUPPORTED_VIDEO_PROVIDERS
    )

    dependencies = {
        "database": _database_status(),
        "storage": storage_status,
        "ffmpeg": ffmpeg_status,
        "llm": llm_status,
        "image_provider": image_status,
        "video_provider": video_status,
    }

    overall = _overall_status(dependencies, storage_status)

    return {
        "status": overall,
        "service": "ai-shorts-studio-v1",
        "dependencies": dependencies,
        "config": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_configured": settings.llm_configured,
            "llm_status": llm_status,
            "llm_credentialed": settings.llm_credentialed,
            "image_provider": settings.image_provider,
            "image_model": settings.image_model,
            "image_configured": settings.image_configured,
            "image_status": image_status,
            "video_provider": settings.video_provider,
            "video_model": settings.video_model,
            "video_configured": settings.video_configured,
            "video_status": video_status,
            "database_url_set": bool(settings.database_url),
            "storage_path": storage_detail,
            "ffmpeg_version": ffmpeg.version,
        },
    }

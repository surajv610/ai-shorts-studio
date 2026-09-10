"""Google Veo video-generation provider (Gemini API ``predictLongRunning``).

Implements image-to-video generation using Veo 3.1 (currently
``veo-3.1-generate-preview`` on the Gemini API). Video generation is
asynchronous: submit -> operation id -> poll -> download.

Request shape (current upstream docs, using Vertex-style instances):
    POST {base}/models/{model}:predictLongRunning
    {
      "instances": [{
        "prompt": ...,
        "image": {"mimeType": "image/png", "bytesBase64Encoded": "<b64>"}
      }],
      "parameters": {"aspectRatio": "9:16", "durationSeconds": 8,
                     "resolution": "720p", "sampleCount": 1}
    }

Polling:
    GET {base}/{operation_name}          (operation name from POST response)

Completed operation returns, under ``response.generateVideoResponse.generatedSamples``
a ``video`` object with a temporary ``uri`` (and ``mimeType``). The video is
downloaded and stored in project storage; the internal file reference is
returned instead of the transient provider URL.

The endpoint/model are configurable via environment (VIDEO_MODEL, VIDEO_BASE_URL).
"""

import base64
import uuid
from pathlib import Path
from typing import Optional

from backend.config import Settings, get_settings
from backend.providers.base import (
    AsyncJob,
    AsyncJobStatus,
    AsyncPollingClient,
    AsyncVideoProvider,
    GenerationOptions,
    GenerationStatus,
    ProviderCapabilities,
    ProviderCapabilityError,
    ProviderGenerationError,
    ProviderNotConfiguredError,
    ProviderSafetyError,
    VideoGenerationResult,
    VideoProvider,
    coerce_options,
)
from backend.providers.http import APIClient, map_http_error

DEFAULT_VEO_MODEL = "veo-3.1-generate-preview"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Veo documents 9:16 (portrait) and 16:9 (landscape) only.
SUPPORTED_ASPECT_RATIOS = {"9:16", "16:9"}
# Veo accepts 4 / 6 / 8 seconds; image-to-video is supported at 8 seconds.
ALLOWED_DURATIONS = (4, 6, 8)
IMAGE_DURATION = 8

IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class GoogleVeoProvider(VideoProvider, AsyncVideoProvider, AsyncPollingClient):
    """Google Veo image-to-video generation via the Gemini API."""

    name = "google"

    _capabilities = ProviderCapabilities(
        name="google",
        generates="video",
        capabilities={
            "image_to_video",
            "duration_control",
            "aspect_ratio_9_16",
            "aspect_ratio_16_9",
        },
        supports_async=True,
        description="Google Veo image-to-video generation (predictLongRunning).",
        models=[DEFAULT_VEO_MODEL],
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        output_dir: Optional[str] = None,
        request_func=None,
    ):
        self.settings = settings or get_settings()
        if not self.settings.video_api_key.get_secret_value():
            raise ProviderNotConfiguredError(
                "VIDEO_PROVIDER=google requires VIDEO_API_KEY (or GEMINI_API_KEY)."
            )
        self.api_key = self.settings.video_api_key.get_secret_value()
        self.model = self.settings.video_model or DEFAULT_VEO_MODEL
        self.base_url = self.settings.video_base_url or DEFAULT_GEMINI_BASE_URL
        base = (
            self.settings.storage_path
            or str(Path(__file__).resolve().parent.parent.parent / "storage")
        )
        self._client = APIClient(
            base_url=self.base_url,
            headers={"x-goog-api-key": self.api_key},
            timeout=self.settings.video_timeout,
            retry_attempts=self.settings.retry_attempts,
            retry_backoff=self.settings.retry_backoff,
            retry_max_delay=self.settings.retry_max_delay,
            request_func=request_func,
            error_parser=map_http_error,
        )
        self._output_dir = Path(output_dir or Path(base) / "google")

    # -- option translation ------------------------------------------------

    def _resolve_ratio(self, options: Optional[GenerationOptions]) -> str:
        value = options.aspect_ratio_value() if options else None
        ratio = value or "9:16"
        if ratio not in SUPPORTED_ASPECT_RATIOS:
            raise ProviderCapabilityError(
                f"Google Veo supports only 9:16 and 16:9, got {ratio!r}."
            )
        return ratio

    def _resolve_duration(self, options: Optional[GenerationOptions]) -> int:
        # Image-to-video generation is documented at 8 seconds for Veo 3.1.
        requested = None
        if options is not None and options.duration_seconds:
            requested = int(options.duration_seconds)
        if requested is None:
            return IMAGE_DURATION
        return min(ALLOWED_DURATIONS, key=lambda d: abs(d - requested))

    @staticmethod
    def _read_image(image_reference: str) -> tuple:
        path = Path(image_reference)
        data = path.read_bytes()
        mime = IMAGE_MIME_BY_EXT.get(path.suffix.lower(), "image/png")
        return data, mime

    # -- async interface ---------------------------------------------------

    def submit_async(
        self,
        image_reference: str,
        prompt: str,
        options: Optional[GenerationOptions] = None,
    ) -> AsyncJob:
        options = coerce_options(options)
        ratio = self._resolve_ratio(options)
        duration = self._resolve_duration(options)
        model = options.model or self.model

        image_bytes, mime = self._read_image(image_reference)
        instance = {
            "prompt": prompt,
            "image": {
                "mimeType": mime,
                "bytesBase64Encoded": base64.b64encode(image_bytes).decode("ascii"),
            },
        }
        parameters = {
            "aspectRatio": ratio,
            "durationSeconds": duration,
            "resolution": str(options.params.get("resolution") or "720p"),
            "sampleCount": 1,
        }
        data = self._client.post_json(
            f"/models/{model}:predictLongRunning",
            {"instances": [instance], "parameters": parameters},
        )
        operation = data.get("name") or data.get("operation", {}).get("name")
        if not operation:
            raise ProviderGenerationError(
                f"Veo returned no operation name: {str(data)[:200]}"
            )
        generation_id = str(uuid.uuid4())
        return AsyncJob(
            provider=self.name,
            job_id=str(operation),
            status=AsyncJobStatus.PENDING,
            raw={
                "generation_id": generation_id,
                "prompt": prompt,
                "model": model,
                "aspect_ratio": ratio,
                "duration_seconds": duration,
                "input_image": image_reference,
                "operation": operation,
            },
        )

    def poll_async(self, job: AsyncJob, options=None) -> AsyncJob:
        path = job.job_id
        if not (path.startswith("operations/") or path.startswith("models/")):
            path = f"operations/{path}"
        try:
            data = self._client.get_json(path)
        except ProviderSafetyError as e:
            job.status = AsyncJobStatus.FAILED
            job.error = str(e)
            job.provider_error = {"error": str(e)}
            return job

        if data.get("error"):
            detail = data["error"]
            message = _error_message(detail)
            job.status = AsyncJobStatus.FAILED
            job.error = message or "Veo operation failed"
            job.provider_error = dict(detail) if isinstance(detail, dict) else {"error": detail}
            if _is_safety(message):
                job.provider_error["is_safety"] = True
            return job

        if not data.get("done"):
            job.status = AsyncJobStatus.PROCESSING
            job.raw.setdefault("metadata", {})
            return job

        response = data.get("response") or {}
        video_response = response.get("generateVideoResponse") or {}
        samples = video_response.get("generatedSamples") or []
        first = samples[0] if samples else {}
        video = (first or {}).get("video") or {}

        uri = video.get("uri")
        if not uri:
            filtered = bool(first.get("raiMediaFiltered"))
            job.status = AsyncJobStatus.FAILED
            job.error = (
                "Veo video generation produced no output"
                + (" (content filtered by safety classifier)" if filtered else "")
            )
            job.provider_error = {"generated_samples": samples}
            if filtered:
                job.provider_error["is_safety"] = True
            return job

        job.status = AsyncJobStatus.SUCCEEDED
        job.raw["video_uri"] = uri
        job.raw["mime_type"] = video.get("mimeType") or "video/mp4"
        return job

    def download_result(
        self,
        job: AsyncJob,
        options: Optional[GenerationOptions] = None,
    ) -> VideoGenerationResult:
        """Download a completed job's video into local storage."""
        if job.status != AsyncJobStatus.SUCCEEDED or not job.raw.get("video_uri"):
            raise ProviderGenerationError(
                f"Cannot download: job {job.job_id} is not SUCCEEDED ({job.status})."
            )
        options = coerce_options(options)
        out_dir = Path(options.params.get("output_dir") or self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gen = job.raw.get("generation_id") or str(uuid.uuid4())
        path = out_dir / f"google_{gen[:8]}.mp4"
        bytes_data = self._client.get_content(job.raw["video_uri"])
        path.write_bytes(bytes_data)
        return VideoGenerationResult(
            generation_id=gen,
            provider=self.name,
            model=job.raw.get("model") or self.model,
            prompt=job.raw.get("prompt") or "",
            status=GenerationStatus.SUCCEEDED,
            video=str(path),
            job_id=job.job_id,
            raw={
                "operation": job.job_id,
                "video_uri": job.raw["video_uri"],
                "mime_type": job.raw.get("mime_type"),
                "aspect_ratio": job.raw.get("aspect_ratio"),
                "duration_seconds": job.raw.get("duration_seconds"),
            },
        )

    # -- synchronous convenience ------------------------------------------

    def generate_video(
        self,
        image_reference: str,
        prompt: str,
        options: Optional[GenerationOptions] = None,
    ) -> VideoGenerationResult:
        """Submit, poll to completion, and download the result synchronously."""
        options = coerce_options(options)
        job = self.submit_async(image_reference, prompt, options=options)

        poll_timeout = float(
            options.params.get("poll_timeout") or self.settings.video_poll_timeout
        )
        poll_interval = float(
            options.params.get("poll_interval") or self.settings.video_poll_interval
        )
        job = self.wait(job, timeout_seconds=poll_timeout, poll_interval=poll_interval)

        if job.status == AsyncJobStatus.SUCCEEDED:
            return self.download_result(job, options=options)

        error = job.error or "Veo generation failed"
        if job.provider_error and job.provider_error.get("is_safety"):
            raise ProviderSafetyError(error)
        raise ProviderGenerationError(error)


def _error_message(detail) -> str:
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("status") or "")
    return str(detail)


def _is_safety(msg: str) -> bool:
    lowered = (msg or "").lower()
    markers = ("safety", "blocked", "filtered", "content_filter", "payment")
    return any(m in lowered for m in markers)
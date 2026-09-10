"""Google Gemini image-generation provider (Gemini API ``generateContent``).

Implements the current documented Gemini API image-generation capability (Nano
Banana family, e.g. ``gemini-3.1-flash-image``). It does NOT use the deprecated
Imagen path.

Request shape (current upstream docs):
    POST {base}/models/{model}:generateContent
    {
      "contents": [{"parts": [{"text": ...}]}],
      "generationConfig": {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {"aspectRatio": "9:16", "imageSize": "1K"}
      }
    }

Images are returned base64-encoded in ``candidates[*].content.parts[*].inlineData``.
This adapter downloads/decodes them into the project's image storage and returns
stable internal file references (never a transient provider URL).

The endpoint/model are configurable via environment (IMAGE_MODEL, IMAGE_BASE_URL)
so nothing API-specific leaks into agents.
"""

import base64
import uuid
from pathlib import Path
from typing import List, Optional

from backend.config import Settings, get_settings
from backend.providers.base import (
    AsyncImageProvider,
    AsyncJob,
    GenerationOptions,
    GenerationStatus,
    ImageGenerationResult,
    ImageProvider,
    ProviderCapabilities,
    ProviderCapabilityError,
    ProviderGenerationError,
    ProviderNotConfiguredError,
    ProviderSafetyError,
    coerce_options,
)
from backend.providers.http import APIClient, map_http_error

DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_IMAGE_SIZE = "1K"

# Aspect ratios the image model family documents (9:16 is our target).
SUPPORTED_ASPECT_RATIOS = {
    "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
}

_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


class GoogleImageProvider(ImageProvider, AsyncImageProvider):
    """Google Gemini image generation via the Gemini API (Nano Banana)."""

    name = "google"

    _capabilities = ProviderCapabilities(
        name="google",
        generates="image",
        capabilities={
            "text_to_image",
            "multi_candidate",
            "aspect_ratio_9_16",
            "aspect_ratio_16_9",
            "aspect_ratio_1_1",
        },
        supports_async=False,
        description="Google Gemini Nano Banana image generation (generateContent).",
        models=[DEFAULT_GEMINI_IMAGE_MODEL],
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        output_dir: Optional[str] = None,
        request_func=None,
    ):
        self.settings = settings or get_settings()
        if not self.settings.image_api_key.get_secret_value():
            raise ProviderNotConfiguredError(
                "IMAGE_PROVIDER=google requires IMAGE_API_KEY (or GEMINI_API_KEY)."
            )
        self.api_key = self.settings.image_api_key.get_secret_value()
        self.model = self.settings.image_model or DEFAULT_GEMINI_IMAGE_MODEL
        self.base_url = self.settings.image_base_url or DEFAULT_GEMINI_BASE_URL
        base = (
            self.settings.storage_path
            or str(Path(__file__).resolve().parent.parent.parent / "storage")
        )
        self._client = APIClient(
            base_url=self.base_url,
            headers={"x-goog-api-key": self.api_key},
            timeout=self.settings.image_timeout,
            retry_attempts=self.settings.retry_attempts,
            retry_backoff=self.settings.retry_backoff,
            retry_max_delay=self.settings.retry_max_delay,
            request_func=request_func,
            error_parser=map_http_error,
        )
        self._output_dir = Path(output_dir or Path(base) / "google")

    # -- internal helpers --------------------------------------------------

    def _resolve_ratio(self, options: Optional[GenerationOptions]) -> str:
        value = options.aspect_ratio_value() if options else None
        ratio = value or "9:16"
        if ratio not in SUPPORTED_ASPECT_RATIOS:
            raise ProviderCapabilityError(
                f"Google image provider does not support aspect ratio {ratio!r}. "
                f"Supported: {sorted(SUPPORTED_ASPECT_RATIOS)}"
            )
        return ratio

    def _build_payload(self, prompt: str, ratio: str, options: GenerationOptions) -> dict:
        image_size = str(options.params.get("image_size") or DEFAULT_IMAGE_SIZE)
        generation_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": ratio,
                "imageSize": image_size,
            },
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        return payload

    @staticmethod
    def _extract_images(data: dict) -> List[dict]:
        """Return [{mime_type, data(base64)}] from candidates[].parts[].inlineData."""
        out: List[dict] = []
        candidates = data.get("candidates") or []
        if not candidates:
            return out
        parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                out.append(
                    {
                        "mime_type": inline.get("mimeType") or inline.get("mime_type") or "image/png",
                        "data": inline["data"],
                    }
                )
        return out

    @staticmethod
    def _finish_reason(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        return str((candidates[0] or {}).get("finishReason") or "").upper()

    def generate_images(
        self,
        prompt: str,
        options: Optional[GenerationOptions] = None,
        count: int = 2,
    ) -> ImageGenerationResult:
        options = coerce_options(options)
        ratio = self._resolve_ratio(options)
        model = options.model or self.model
        n = max(1, int(count or 2))

        out_dir = Path(options.params.get("output_dir") or self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        generation_id = str(uuid.uuid4())
        prefix = generation_id[:8]
        images: List[str] = []
        request_ids: List[str] = []

        for i in range(n):
            payload = self._build_payload(prompt, ratio, options)
            data = self._client.post_json(
                f"/models/{model}:generateContent", payload
            )
            finish = self._finish_reason(data)
            if finish in ("SAFETY", "BLOCKLIST", "RECITATION"):
                raise ProviderSafetyError(
                    f"Gemini image generation blocked for candidate {i + 1}: {finish}"
                )
            found = self._extract_images(data)
            if not found:
                block = (data.get("promptFeedback") or {}).get("blockReason")
                raise ProviderGenerationError(
                    f"Gemini returned no image data (finish={finish}, block={block})"
                )
            # The model returns one image per call; use the first inline part.
            img = found[0]
            raw_bytes = base64.b64decode(img["data"])
            ext = _MIME_EXT.get(img["mime_type"], ".png")
            path = out_dir / f"google_{prefix}_{i + 1}{ext}"
            path.write_bytes(raw_bytes)
            images.append(str(path))
            request_ids.append(str(data.get("response", {}).get("requestId") or "") or "")

        # Provider request identifier, where available (often empty for this API).
        request_id = request_ids[0] if request_ids else ""

        return ImageGenerationResult(
            generation_id=generation_id,
            provider=self.name,
            model=model,
            prompt=prompt,
            status=GenerationStatus.SUCCEEDED,
            images=images,
            raw={
                "request_id": request_id or None,
                "aspect_ratio": ratio,
                "count": n,
                "model": model,
            },
        )

    # -- async surface -----------------------------------------------------
    # Image generation is synchronous; expose a conformant immediately-ready job.

    def submit_async(self, prompt, options=None, count=2) -> AsyncJob:
        res = self.generate_images(prompt, options=options, count=count)
        return AsyncJob(
            provider=self.name,
            job_id=res.generation_id,
            status="SUCCEEDED",
            raw={"images": res.images},
        )

    def poll_async(self, job: AsyncJob, options=None) -> AsyncJob:
        return job
"""Image providers: a mock implementation (no external API or paid calls).

The mock produces two visually distinguishable PNG images per scene so the
image-selection step can be exercised end-to-end without spending credits.

Declared capabilities: ``text_to_image``, ``multi_candidate``,
``aspect_ratio_9_16`` and (any requested) ``aspect_ratio_*``. The mock honors
width/height/aspect-ratio options and exposes a synchronous result.
"""

import struct
import zlib
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.config import Settings, get_settings
from backend.providers.base import (
    AsyncJob,
    AsyncImageProvider,
    GenerationOptions,
    GenerationStatus,
    ImageGenerationResult,
    ImageProvider,
    ProviderCapabilities,
    ProviderCapabilityError,
    coerce_options,
)

DEFAULT_MOCK_WIDTH = 320
DEFAULT_MOCK_HEIGHT = 568


def _png_bytes(width: int, height: int, rgb: tuple) -> bytes:
    """Encode a solid-color RGB image as a minimal valid PNG (no deps)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        c += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    idat = zlib.compress(raw)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


class MockImageProvider(ImageProvider, AsyncImageProvider):
    """Mock image provider: writes two distinct solid-color PNGs.

    Works with a plain dict or a :class:`GenerationOptions` options object so
    the existing sync call sites keep working unchanged.
    """

    name = "mock"

    _capabilities = ProviderCapabilities(
        name="mock",
        generates="image",
        capabilities={"text_to_image", "multi_candidate", "aspect_ratio_9_16"},
        supports_async=False,
        description="Local PNG generation for development (no external API).",
        models=["mock-image-model"],
    )

    def __init__(self, settings: Optional[Settings] = None, output_dir: Optional[str] = None):
        self.settings = settings or get_settings()
        self.model = self.settings.image_model or "mock-image-model"
        # Default output under the configured storage path or a local mock dir.
        base = (
            self.settings.storage_path
            or str(Path(__file__).resolve().parent.parent.parent / "storage" / "mock")
        )
        self._output_dir = Path(output_dir or base) / "images"

    def _resolve_size(self, options: Optional[GenerationOptions]) -> tuple:
        ratio = options.aspect_ratio_key()
        if options.width and options.height:
            return options.width, options.height
        if ratio:
            a, b = ratio.split("_")
            scale = 64
            w = int(a) * scale
            h = int(b) * scale
            return max(1, w), max(1, h)
        return DEFAULT_MOCK_WIDTH, DEFAULT_MOCK_HEIGHT

    def generate_images(
        self,
        prompt: str,
        options: Optional[GenerationOptions] = None,
        count: int = 2,
    ) -> ImageGenerationResult:
        options = coerce_options(options)
        out_dir = Path(options.params.get("output_dir") or self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generation_id = str(uuid4())
        width, height = self._resolve_size(options)
        # Sanity guard: cout is capped to the number of distinct palettes (2),
        # but we still return `count` refs by cycling palettes for larger asks.
        images = []
        # Two clearly different palettes so candidates are distinguishable.
        palettes = [(24, 90, 70), (180, 120, 40)]  # green vs amber
        n = max(1, count)
        for i in range(n):
            color = palettes[i % len(palettes)]
            path = out_dir / f"mock_{generation_id[:8]}_{i + 1}.png"
            path.write_bytes(_png_bytes(width, height, color))
            images.append(str(path))
        return ImageGenerationResult(
            generation_id=generation_id,
            provider=self.name,
            model=options.model or self.model,
            prompt=prompt,
            status=GenerationStatus.SUCCEEDED,
            images=images,
            raw={
                "width": width,
                "height": height,
                "aspect_ratio": options.aspect_ratio,
                "count": n,
            },
        )

    # -- async contract ---------------------------------------------------
    # The mock is synchronous, but it exposes a conformant async surface so
    # callers that always use the generic interface still work. It simply
    # returns a SUCCEEDED job immediately.

    def submit_async(self, prompt, options=None, count=2) -> AsyncJob:
        options = coerce_options(options)
        if not self.supports("text_to_image"):
            raise ProviderCapabilityError(
                f"{self.name} does not support text_to_image"
            )
        res = self.generate_images(prompt, options=options, count=count)
        return AsyncJob(
            provider=self.name,
            job_id=res.generation_id,
            status="SUCCEEDED",
            raw={"images": res.images},
        )

    def poll_async(self, job: AsyncJob, options=None) -> AsyncJob:
        return job

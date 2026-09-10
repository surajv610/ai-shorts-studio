"""Video providers: a mock implementation using FFmpeg to make real clips.

The mock produces a short valid MP4 clip per scene so video generation and
assembly can be exercised without spending credits. If FFmpeg is unavailable,
it falls back to writing a deterministic stub file (clearly marked).

Declared capabilities: ``image_to_video``, ``duration_control`` and
``aspect_ratio_9_16``. The mock honors duration and aspect-ratio options and
exposes a synchronous result plus a conformant async surface.
"""

import subprocess
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.ffmpeg import get_ffmpeg_info
from backend.config import Settings, get_settings
from backend.providers.base import (
    AsyncJob,
    AsyncJobStatus,
    AsyncVideoProvider,
    GenerationOptions,
    GenerationStatus,
    ProviderCapabilities,
    ProviderCapabilityError,
    ProviderGenerationError,
    VideoGenerationResult,
    VideoProvider,
    coerce_options,
)

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 568


def _resolve_size(options: Optional[GenerationOptions]) -> tuple:
    if options.width and options.height:
        return options.width, options.height
    ratio = options.aspect_ratio_key()
    if ratio:
        a, b = ratio.split("_")
        scale = 64
        return max(1, int(a) * scale), max(1, int(b) * scale)
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


class MockVideoProvider(VideoProvider, AsyncVideoProvider):
    """Mock image-to-video provider that generates clips locally."""

    name = "mock"

    _capabilities = ProviderCapabilities(
        name="mock",
        generates="video",
        capabilities={"image_to_video", "duration_control", "aspect_ratio_9_16"},
        supports_async=False,
        description="Local clip generation via FFmpeg (no external API).",
        models=["mock-video-model"],
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        output_dir: Optional[str] = None,
        *,
        poll_steps: int = 2,
        fail_after: Optional[int] = None,
        fail_message: str = "mock video generation failure",
    ):
        self.settings = settings or get_settings()
        self.model = self.settings.video_model or "mock-video-model"
        self.poll_steps = max(1, int(poll_steps))
        self.fail_after = fail_after
        self.fail_message = fail_message
        base = (
            self.settings.storage_path
            or str(Path(__file__).resolve().parent.parent.parent / "storage" / "mock")
        )
        self._output_dir = Path(output_dir or base) / "videos"

    def generate_video(
        self,
        image_reference: str,
        prompt: str,
        options: Optional[GenerationOptions] = None,
    ) -> VideoGenerationResult:
        options = coerce_options(options)
        out_dir = Path(options.params.get("output_dir") or self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generation_id = str(uuid4())
        duration = max(1, int(options.duration_seconds or 3))
        width, height = _resolve_size(options)
        path = out_dir / f"mock_{generation_id[:8]}.mp4"

        ffmpeg = get_ffmpeg_info()
        if ffmpeg.available and ffmpeg.path:
            self._render_mp4(
                ffmpeg.path, str(path), duration, width=width, height=height
            )
            status = GenerationStatus.SUCCEEDED
            error = None
        else:
            # Fallback when FFmpeg is missing: deterministic stub video file.
            path.write_text(
                f"MOCK VIDEO (no ffmpeg): source={image_reference}, prompt={prompt}\n"
            )
            status = GenerationStatus.SUCCEEDED
            error = "FFmpeg unavailable; wrote stub marker file instead of a real clip."

        return VideoGenerationResult(
            generation_id=generation_id,
            provider=self.name,
            model=options.model or self.model,
            prompt=prompt,
            status=status,
            error=error,
            video=str(path),
            raw={
                "source": image_reference,
                "duration_seconds": duration,
                "width": width,
                "height": height,
                "aspect_ratio": options.aspect_ratio,
            },
        )

    # -- async contract ---------------------------------------------------
    # The mock simulates the real async lifecycle: submit -> PENDING, then each
    # poll advances PENDING -> PROCESSING -> SUCCEEDED (or FAILED when
    # ``fail_after`` polls are reached). The actual MP4 is rendered eagerly at
    # submit time so the test suite stays deterministic and offline.

    def submit_async(self, image_reference, prompt, options=None) -> AsyncJob:
        options = coerce_options(options)
        if not self.supports("image_to_video"):
            raise ProviderCapabilityError(
                f"{self.name} does not support image_to_video"
            )
        res = self.generate_video(image_reference, prompt, options=options)
        return AsyncJob(
            provider=self.name,
            job_id=f"mock-job-{res.generation_id[:8]}",
            status=AsyncJobStatus.PENDING,
            raw={
                "generation_id": res.generation_id,
                "video": res.video,
                "prompt": prompt,
                "model": options.model or self.model,
                "duration_seconds": res.raw.get("duration_seconds"),
                "_mock_polls": 0,
            },
        )

    def poll_async(self, job: AsyncJob, options=None) -> AsyncJob:
        raw = job.raw or {}
        raw["_mock_polls"] = int(raw.get("_mock_polls", 0)) + 1
        if self.fail_after is not None and raw["_mock_polls"] >= self.fail_after:
            job.status = AsyncJobStatus.FAILED
            job.error = self.fail_message
            job.provider_error = {"is_safety": "safety" in self.fail_message.lower()}
            return job
        if raw["_mock_polls"] < self.poll_steps:
            job.status = AsyncJobStatus.PROCESSING
        else:
            job.status = AsyncJobStatus.SUCCEEDED
        return job

    def download_result(
        self,
        job: AsyncJob,
        options: Optional[GenerationOptions] = None,
    ) -> VideoGenerationResult:
        """Return the already-local mock MP4 (nothing to download)."""
        if job.status != AsyncJobStatus.SUCCEEDED:
            raise ProviderGenerationError(
                f"Cannot download: job {job.job_id} is not SUCCEEDED ({job.status})."
            )
        raw = job.raw or {}
        return VideoGenerationResult(
            generation_id=raw.get("generation_id") or job.job_id,
            provider=self.name,
            model=raw.get("model") or self.model,
            prompt=raw.get("prompt") or "",
            status=GenerationStatus.SUCCEEDED,
            video=raw["video"],
            job_id=job.job_id,
            raw={
                "duration_seconds": raw.get("duration_seconds"),
                "aspect_ratio": (coerce_options(options).aspect_ratio),
            },
        )

    @staticmethod
    def _render_mp4(
        ffmpeg_path: str, out_path: str, duration: int, width: int, height: int
    ) -> None:
        """Render a short solid-color MP4 with a moving gradient to look like a clip."""
        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate=24:duration={duration}",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed to render mock video: {proc.stderr[:300]}"
            )

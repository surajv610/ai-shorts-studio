"""FFmpeg detection and capability probing.

The backend uses FFmpeg to assemble final MP4s and (optionally) to generate
mock video clips. This module lets the rest of the app know whether FFmpeg is
installed and usable.
"""

import shutil
import subprocess
from typing import Optional


class FFmpegInfo:
    """Information about the detected FFmpeg installation."""

    def __init__(
        self,
        available: bool,
        path: Optional[str] = None,
        version: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.available = available
        self.path = path
        self.version = version
        self.error = error

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "path": self.path,
            "version": self.version,
            "error": self.error,
        }

    def __bool__(self) -> bool:
        return self.available


def detect_ffmpeg() -> FFmpegInfo:
    """Locate ffmpeg on PATH and return its version if usable."""
    path = shutil.which("ffmpeg")
    if not path:
        return FFmpegInfo(available=False, error="ffmpeg not found on PATH")

    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return FFmpegInfo(available=False, path=path, error=str(e))

    if proc.returncode != 0:
        return FFmpegInfo(
            available=False,
            path=path,
            error=proc.stderr.strip()[:200] or "ffmpeg -version failed",
        )

    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    version = first_line.strip()[:120] if first_line else None
    return FFmpegInfo(available=True, path=path, version=version)


def is_ffmpeg_available() -> bool:
    return detect_ffmpeg().available


_detected_cache: Optional[FFmpegInfo] = None


def get_ffmpeg_info(refresh: bool = False) -> FFmpegInfo:
    """Cached FFmpeg detection (refresh=True re-probes)."""
    global _detected_cache
    if _detected_cache is None or refresh:
        _detected_cache = detect_ffmpeg()
    return _detected_cache

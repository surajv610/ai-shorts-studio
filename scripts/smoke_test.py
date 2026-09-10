#!/usr/bin/env python3
"""Manual smoke test for the real (paid) provider adapters.

WARNING
-------
Running this script calls real AI APIs and CONSUMES CREDITS (tokens + image
credits + video credits). It is intentionally NOT part of the automated test
suite. Run it only when you want to verify a live integration.

It exercises whatever providers are configured via environment:
    LLM_PROVIDER=openai     -> 1 chat completion (small prompt)
    IMAGE_PROVIDER=google   -> 1 Gemini Nano-Banana image
    VIDEO_PROVIDER=google   -> 1 short (4s) 9:16 Veo image-to-video clip (uses
                               the image from the image step, or `INPUT_IMAGE`
                               if set), via the async submit->poll->download path

Providers left at mock are skipped (mock = no cost).

NOTE FOR VIDEO: keep it quick — one short 9:16 clip only. The smoke test drives
the same async submit/poll/download flow the app uses, and reports the local
MP4 path when done. To view it in the app UI, run a project through the app to
the video-generation screen (clips are stored under projects/<id>/videos and
served at /assets/projects/<id>/videos/...), or open the printed file directly.

Usage:
    source .env   # or export the *_PROVIDER / *_API_KEY vars yourself
    python3 scripts/smoke_test.py
    echo $?       # 0 = all configured providers succeeded
"""

import os
import sys
import tempfile
import time
from pathlib import Path


def _configured_deps() -> list:
    deps = []
    if os.environ.get("LLM_PROVIDER", "mock").lower() == "openai":
        deps.append("llm")
    if os.environ.get("IMAGE_PROVIDER", "mock").lower() == "google":
        deps.append("image")
    if os.environ.get("VIDEO_PROVIDER", "mock").lower() == "google":
        deps.append("video")
    if not deps:
        print("Nothing to smoke-test: all providers are set to mock.")
        print("Set LLM_PROVIDER/IMAGE_PROVIDER/VIDEO_PROVIDER to real adapters to proceed.")
        return []
    return deps


def main() -> int:
    deps = _configured_deps()
    if not deps:
        return 2

    print("=" * 60)
    print("SMOKE TEST — REAL (PAID) PROVIDER CALLS")
    print("This will consume AI credits (tokens/image/video). Proceeding...")
    print("=" * 60)

    # Import after the warning so a misconfigured app still prints it first.
    from backend.config import get_settings

    settings = get_settings()
    results = []
    with tempfile.TemporaryDirectory(prefix="aiss-smoke-") as tmp:
        for dep in deps:
            try:
                if dep == "llm":
                    results.append(("llm", _smoke_llm(settings)))
                elif dep == "image":
                    results.append(("image", _smoke_image(settings, tmp)))
                elif dep == "video":
                    # Prefer the image we just generated, else INPUT_IMAGE.
                    if "image" in deps and results and results[-1][0] == "image":
                        input_image = results[-1][1]["image"]
                    else:
                        input_image = os.environ.get("INPUT_IMAGE")
                        if not input_image:
                            raise RuntimeError(
                                "VIDEO_PROVIDER=google needs INPUT_IMAGE when the "
                                "image provider is not being exercised in the same run."
                            )
                    results.append(("video", _smoke_video(settings, tmp, input_image)))
            except Exception as e:
                results.append((dep, {"error": f"{type(e).__name__}: {e}"}))
                traceback.print_exc()

    print()
    print("=" * 60)
    ok = True
    for dep, res in results:
        if "error" in res:
            ok = False
            print(f"[FAIL] {dep}: {res['error']}")
        else:
            print(f"[ OK ] {dep}")
            for key, value in res.items():
                print(f"        {key}: {value}")
    print("=" * 60)
    print("RESULT:", "ALL CHECKS PASSED" if ok else "ONE OR MORE CHECKS FAILED")
    return 0 if ok else 1


def _smoke_llm(settings) -> dict:
    from backend.providers.registry import get_llm_provider

    provider = get_llm_provider(settings)
    res = provider.generate_text("Reply with the single word: OK")
    return {"model": provider.model, "provider": provider.name, "reply": res.text[:200]}


def _smoke_image(settings, tmp) -> dict:
    from backend.providers.base import GenerationOptions
    from backend.providers.registry import get_image_provider

    provider = get_image_provider(settings)
    res = provider.generate_images(
        "A single blooming flower on dark soil, 9:16, photorealistic.",
        options=GenerationOptions(aspect_ratio="9:16", params={"output_dir": tmp}),
        count=1,
    )
    if not res.images or not Path(res.images[0]).exists():
        raise RuntimeError(f"Image generation returned no readable file: {res.images}")
    size = Path(res.images[0]).stat().st_size
    return {"model": res.model, "provider": res.provider, "image": res.images[0], "bytes": size}


def _smoke_video(settings, tmp, input_image) -> dict:
    from backend.providers.base import GenerationOptions
    from backend.providers.registry import get_video_provider

    in_path = Path(input_image)
    if not in_path.exists():
        raise RuntimeError(f"Input image does not exist: {input_image}")
    provider = get_video_provider(settings)
    options = GenerationOptions(
        aspect_ratio="9:16",
        duration_seconds=4,
        params={
            "output_dir": tmp,
            "poll_interval": settings.video_poll_interval,
            "poll_timeout": settings.video_poll_timeout,
        },
    )
    # Same async submit -> poll -> download path the Video Agent uses.
    job = provider.submit_async(str(in_path), "slow push-in, gentle camera drift", options=options)
    attempts = 0
    while not job.is_terminal and attempts < 400:
        job = provider.poll_async(job, options=options)
        time.sleep(settings.video_poll_interval)
        attempts += 1
    if not job.is_terminal or job.status.value != "SUCCEEDED":
        raise RuntimeError(
            f"Video generation did not succeed: {getattr(job, 'status', 'unknown')} "
            f"- {getattr(job, 'error', 'no error')}"
        )
    res = provider.download_result(job, options=options)
    if not Path(res.video).exists():
        raise RuntimeError(f"Video generation returned no readable file: {res.video}")
    return {
        "model": res.model,
        "provider": res.provider,
        "job_id": res.job_id,
        "video": res.video,
        "bytes": Path(res.video).stat().st_size,
        "note": "view in app under /assets/ after storing under projects/<id>/videos",
    }


if __name__ == "__main__":
    sys.exit(main())
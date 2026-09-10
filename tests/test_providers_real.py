"""Tests for the real provider adapters (OpenAI LLM, Google image, Google Veo).

All HTTP traffic is replaced with ``request_func`` injection -- no real network
or paid API calls happen in this suite.
"""

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import pytest

from backend.config import Settings
from backend.health import get_health
from backend.providers.base import (
    AsyncJob,
    AsyncJobStatus,
    GenerationOptions,
    GenerationStatus,
    ProviderAuthError,
    ProviderCapabilityError,
    ProviderGenerationError,
    ProviderInvalidRequestError,
    ProviderNotConfiguredError,
    ProviderQuotaError,
    ProviderSafetyError,
    ProviderUnavailableError,
    VideoGenerationResult,
)
from backend.providers.google_image import GoogleImageProvider
from backend.providers.google_veo import GoogleVeoProvider
from backend.providers.image import _png_bytes
from backend.providers.llm import OpenAILLMProvider, map_llm_error
from backend.providers.registry import (
    get_image_provider,
    get_video_provider,
    provider_capabilities,
    provider_supports,
)
from backend.providers.http import map_http_error


def _resp(status_code, payload=None, content=None):
    if content is not None:
        return httpx.Response(
            status_code, content=content, request=httpx.Request("GET", "http://x")
        )
    return httpx.Response(
        status_code, json=payload or {}, request=httpx.Request("POST", "http://x")
    )


def _google_settings(tmp_path, **overrides):
    env = {
        "LLM_PROVIDER": "mock",
        "IMAGE_PROVIDER": "google",
        "IMAGE_API_KEY": "test-gemini-key",
        "VIDEO_PROVIDER": "google",
        "VIDEO_API_KEY": "test-gemini-key",
        "STORAGE_PATH": str(tmp_path),
    }
    env.update(overrides)
    return Settings(env=env)


# =========================================================================
# OpenAI LLM adapter
# =========================================================================


def test_openai_llm_success_via_injected_request():
    calls = []

    def fake(method, url, **kwargs):
        calls.append((url, kwargs))
        return _resp(200, {"choices": [{"message": {"content": "hello world"}}]})

    p = OpenAILLMProvider(
        Settings(env={"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"}),
        request_func=fake,
    )
    res = p.generate_text("say hi")
    assert res.text == "hello world"
    assert res.provider == "openai"
    assert calls[0][0].endswith("/chat/completions")
    body = calls[0][1]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"][0]["content"] == "say hi"
    assert calls[0][1]["headers"].get("Authorization") == "Bearer sk-test"


def test_openai_llm_structured_via_injected_request():
    def fake(method, url, **kwargs):
        return _resp(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})

    p = OpenAILLMProvider(
        Settings(env={"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"}),
        request_func=fake,
    )
    out = p.generate_structured(
        "sys", "user", {"name": "out", "schema": {"type": "object"}}
    )
    assert out == {"ok": True}


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, {"error": {"message": "bad key"}}, ProviderAuthError),
        (403, {"error": {"message": "forbidden"}}, ProviderAuthError),
        (429, {"error": {"message": "rate limit"}}, ProviderQuotaError),
        (503, {"error": {"message": "overloaded"}}, ProviderUnavailableError),
        (400, {"error": {"message": "word was blocked by content_filter"}}, ProviderSafetyError),
        (400, {"error": {"message": "invalid param"}}, ProviderInvalidRequestError),
    ],
)
def test_map_llm_error_cases(status, body, expected):
    err = map_llm_error(status, body, "POST", "/chat/completions")
    assert isinstance(err, expected)


def test_openai_llm_http_error_raised():
    def fake(method, url, **kwargs):
        return _resp(401, {"error": {"message": "unauthorized"}})

    p = OpenAILLMProvider(
        Settings(env={"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"}),
        request_func=fake,
    )
    with pytest.raises(ProviderAuthError):
        p.generate_text("hi")


# =========================================================================
# Google Gemini image provider
# =========================================================================


def _image_body() -> dict:
    raw = b"\x89PNG testbytes"
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(raw).decode("ascii"),
                            }
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }


def test_google_image_two_candidates(tmp_path):
    calls = []

    def fake(method, url, **kwargs):
        calls.append(kwargs)
        return _resp(200, _image_body())

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    res = p.generate_images("a flower", count=2)

    assert res.provider == "google"
    assert res.status == GenerationStatus.SUCCEEDED
    assert len(res.images) == 2
    assert len(calls) == 2
    for path in res.images:
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"\x89PNG testbytes"
    # canonical 9:16 kept normalized internally
    assert res.raw["aspect_ratio"] == "9:16"


def test_google_image_payload_and_auth_header(tmp_path):
    seen = {}

    def fake(method, url, **kwargs):
        seen.update(kwargs)
        return _resp(200, _image_body())

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    p.generate_images("x", count=1)

    headers = seen["headers"]
    assert headers.get("x-goog-api-key") == "test-gemini-key"
    assert "Authorization" not in headers  # Google path uses x-goog-api-key, no bearer
    body = seen["json"]
    gc = body["generationConfig"]
    assert gc["responseModalities"] == ["TEXT", "IMAGE"]
    assert gc["imageConfig"]["aspectRatio"] == "9:16"
    assert gc["imageConfig"]["imageSize"] == "1K"
    assert "responseFormat" not in gc
    assert body["contents"][0]["parts"][0]["text"] == "x"


def test_google_image_landscape_ratio(tmp_path):
    seen = {}

    def fake(method, url, **kwargs):
        seen.update(kwargs)
        return _resp(200, _image_body())

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    p.generate_images(
        "wide", options=GenerationOptions(aspect_ratio="16:9"), count=1
    )
    gc = seen["json"]["generationConfig"]
    assert gc["imageConfig"]["aspectRatio"] == "16:9"
    assert gc["imageConfig"]["imageSize"] == "1K"
    assert "responseFormat" not in gc


def test_google_image_no_legacy_response_format_payload(tmp_path):
    seen = {}

    def fake(method, url, **kwargs):
        seen.update(kwargs)
        return _resp(200, _image_body())

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    p.generate_images(
        "x", options=GenerationOptions(aspect_ratio="9:16"), count=1
    )
    gc = seen["json"]["generationConfig"]
    assert gc.get("responseFormat") is None
    assert "response_format" not in gc
    assert gc["imageConfig"]["aspectRatio"] == "9:16"
    assert "aspectRatio" not in gc


def test_google_image_image_size_passthrough(tmp_path):
    seen = {}

    def fake(method, url, **kwargs):
        seen.update(kwargs)
        return _resp(200, _image_body())

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    p.generate_images(
        "x",
        options=GenerationOptions(params={"image_size": "2K"}),
        count=1,
    )
    gc = seen["json"]["generationConfig"]
    assert gc["imageConfig"]["aspectRatio"] == "9:16"
    assert gc["imageConfig"]["imageSize"] == "2K"


def test_google_image_requires_key():
    s = Settings(env={"IMAGE_PROVIDER": "google", "IMAGE_API_KEY": "", "STORAGE_PATH": "/tmp"})
    with pytest.raises(ProviderNotConfiguredError):
        GoogleImageProvider(s)


def test_google_image_http_auth_error(tmp_path):
    def fake(method, url, **kwargs):
        return _resp(401, {"error": {"message": "API key not valid"}})

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    with pytest.raises(ProviderAuthError):
        p.generate_images("x", count=1)


def test_google_image_safety_finish_reason(tmp_path):
    def fake(method, url, **kwargs):
        return _resp(
            200,
            {
                "candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    with pytest.raises(ProviderSafetyError):
        p.generate_images("x", count=1)


def test_google_image_no_inline_data(tmp_path):
    def fake(method, url, **kwargs):
        return _resp(200, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    with pytest.raises(ProviderGenerationError):
        p.generate_images("x", count=1)


def test_google_image_unsupported_ratio(tmp_path):
    def fake(method, url, **kwargs):
        return _resp(200, _image_body())

    p = GoogleImageProvider(_google_settings(tmp_path), request_func=fake)
    with pytest.raises(ProviderCapabilityError):
        p.generate_images(
            "x", options=GenerationOptions(aspect_ratio="1:99"), count=1
        )


def test_map_http_error_client_codes():
    assert isinstance(map_http_error(401, {"error": {"message": "no"}}, "POST", "/"), ProviderAuthError)
    assert isinstance(map_http_error(403, {}, "GET", "/"), ProviderAuthError)
    assert isinstance(map_http_error(429, {}, "GET", "/"), ProviderQuotaError)
    assert isinstance(map_http_error(500, {}, "GET", "/"), ProviderUnavailableError)
    assert isinstance(map_http_error(503, {}, "GET", "/"), ProviderUnavailableError)
    assert isinstance(map_http_error(400, {"error": {"message": "prompt was blocked for safety"}}, "POST", "/"), ProviderSafetyError)
    assert isinstance(map_http_error(404, {}, "GET", "/"), ProviderInvalidRequestError)


# =========================================================================
# Google Veo video provider
# =========================================================================


def _make_input_png(tmp_path) -> str:
    path = tmp_path / "input.png"
    path.write_bytes(_png_bytes(32, 32, (10, 20, 30)))
    return str(path)


def test_google_veo_submit_builds_payload(tmp_path):
    seen = {}
    input_image = _make_input_png(tmp_path)

    def fake(method, url, **kwargs):
        seen.update(kwargs)
        return _resp(200, {"name": "models/veo-3.1-generate-preview/operations/op123"})

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    job = p.submit_async(input_image, "grow slowly")

    assert job.status == AsyncJobStatus.PENDING
    assert job.job_id == "models/veo-3.1-generate-preview/operations/op123"
    assert seen["headers"].get("x-goog-api-key") == "test-gemini-key"
    body = seen["json"]
    assert body["instances"][0]["prompt"] == "grow slowly"
    assert body["instances"][0]["image"]["mimeType"] == "image/png"
    assert body["instances"][0]["image"]["bytesBase64Encoded"]
    assert base64.b64decode(body["instances"][0]["image"]["bytesBase64Encoded"]).startswith(b"\x89PNG")
    assert body["parameters"]["aspectRatio"] == "9:16"
    assert body["parameters"]["durationSeconds"] == 8  # image-to-video clamps to 8s
    assert body["parameters"]["sampleCount"] == 1


def test_google_veo_poll_processing(tmp_path):
    input_image = _make_input_png(tmp_path)

    def fake(method, url, **kwargs):
        if method == "POST":
            return _resp(200, {"name": "operations/op1"})
        return _resp(200, {"done": False, "metadata": {"state": "RUNNING"}})

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    job = p.submit_async(input_image, "poke")
    job = p.poll_async(job)
    assert job.status == AsyncJobStatus.PROCESSING
    assert not job.is_terminal


def test_google_veo_poll_and_download(tmp_path):
    input_image = _make_input_png(tmp_path)
    video_bytes = b"\x00\x00\x00\x18ftypmp42 ... fake mp4"

    def fake(method, url, **kwargs):
        if method == "POST":
            return _resp(200, {"name": "operations/op1"})
        if "op1" in url:
            return _resp(
                200,
                {
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [
                                {"video": {"uri": "https://storage.googleapis.com/clip.mp4", "mimeType": "video/mp4"}}
                            ]
                        }
                    },
                },
            )
        return _resp(200, content=video_bytes)

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    job = p.submit_async(input_image, "grow")
    final = p.poll_async(job)
    assert final.status == AsyncJobStatus.SUCCEEDED
    assert final.raw["video_uri"] == "https://storage.googleapis.com/clip.mp4"

    res: VideoGenerationResult = p.download_result(final)
    assert res.status == GenerationStatus.SUCCEEDED
    assert res.job_id == "operations/op1"
    assert Path(res.video).exists()
    assert Path(res.video).read_bytes() == video_bytes


def test_google_veo_generate_video_end_to_end(tmp_path):
    input_image = _make_input_png(tmp_path)
    video_bytes = b"mp4-bytes"

    def fake(method, url, **kwargs):
        if method == "POST":
            return _resp(200, {"name": "operations/e2e"})
        if "e2e" in url:
            return _resp(
                200,
                {
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [
                                {"video": {"uri": "https://storage.googleapis.com/x.mp4"}}
                            ]
                        }
                    },
                },
            )
        return _resp(200, content=video_bytes)

    s = _google_settings(tmp_path)
    p = GoogleVeoProvider(s, request_func=fake)
    res = p.generate_video(
        input_image,
        "grow",
        options=GenerationOptions(
            aspect_ratio="9:16",
            params={"poll_interval": 0.01, "poll_timeout": 10},
        ),
    )
    assert res.video
    assert Path(res.video).exists()
    assert res.provider == "google"
    assert res.raw["duration_seconds"] == 8


def test_google_veo_poll_failure_message(tmp_path):
    input_image = _make_input_png(tmp_path)

    def fake(method, url, **kwargs):
        if method == "POST":
            return _resp(200, {"name": "operations/op1"})
        return _resp(200, {"done": True, "error": {"code": 400, "message": "kaboom"}})

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    job = p.submit_async(input_image, "poke")
    final = p.poll_async(job)
    assert final.status == AsyncJobStatus.FAILED
    assert "kaboom" in final.error


def test_google_veo_safety_filtered(tmp_path):
    input_image = _make_input_png(tmp_path)

    def fake(method, url, **kwargs):
        if method == "POST":
            return _resp(200, {"name": "operations/op1"})
        return _resp(
            200,
            {
                "done": True,
                "response": {
                    "generateVideoResponse": {
                        "generatedSamples": [{"raiMediaFiltered": True, "video": {}}]
                    }
                },
            },
        )

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    job = p.submit_async(input_image, "poke")
    with pytest.raises(ProviderSafetyError):
        p.generate_video(input_image, "poke", options=GenerationOptions(params={"poll_interval": 0.01, "poll_timeout": 10}))
    # the poll returns FAILED with a safety flag on the job itself
    final = p.poll_async(job)
    assert final.status == AsyncJobStatus.FAILED
    assert final.provider_error and final.provider_error.get("is_safety") is True


def test_google_veo_poll_timeout(tmp_path):
    input_image = _make_input_png(tmp_path)

    def fake(method, url, **kwargs):
        if method == "POST":
            return _resp(200, {"name": "operations/op1"})
        return _resp(200, {"done": False})

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    with pytest.raises(ProviderGenerationError) as exc:
        p.generate_video(
            input_image,
            "grow",
            options=GenerationOptions(params={"poll_interval": 0.005, "poll_timeout": 0.01}),
        )
    assert "timed out" in str(exc.value)


def test_google_veo_requires_key():
    s = Settings(env={"VIDEO_PROVIDER": "google", "VIDEO_API_KEY": "", "STORAGE_PATH": "/tmp"})
    with pytest.raises(ProviderNotConfiguredError):
        GoogleVeoProvider(s)


def test_google_veo_http_auth_on_submit(tmp_path):
    input_image = _make_input_png(tmp_path)

    def fake(method, url, **kwargs):
        return _resp(401, {"error": {"message": "bad key"}})

    p = GoogleVeoProvider(_google_settings(tmp_path), request_func=fake)
    with pytest.raises(ProviderAuthError):
        p.submit_async(input_image, "poke")


# =========================================================================
# Registry / config / capability integration
# =========================================================================


def test_registry_returns_google_image_provider(tmp_path):
    p = get_image_provider(_google_settings(tmp_path))
    assert isinstance(p, GoogleImageProvider)
    assert p.name == "google"


def test_registry_google_image_requires_key():
    s = Settings(env={"IMAGE_PROVIDER": "google", "IMAGE_API_KEY": "", "STORAGE_PATH": "/tmp"})
    with pytest.raises(ProviderNotConfiguredError):
        get_image_provider(s)


def test_registry_returns_google_video_provider(tmp_path):
    p = get_video_provider(_google_settings(tmp_path))
    assert isinstance(p, GoogleVeoProvider)
    assert p.name == "google"


def test_registry_google_video_requires_key():
    s = Settings(env={"VIDEO_PROVIDER": "google", "VIDEO_API_KEY": "", "STORAGE_PATH": "/tmp"})
    with pytest.raises(ProviderNotConfiguredError):
        get_video_provider(s)


def test_config_gemini_api_key_fallback():
    s = Settings(env={"IMAGE_PROVIDER": "google", "GEMINI_API_KEY": "shared-key", "STORAGE_PATH": "/tmp"})
    assert s.image_api_key.get_secret_value() == "shared-key"
    assert s.image_configured is True
    v = Settings(env={"VIDEO_PROVIDER": "google", "GOOGLE_API_KEY": "legacy", "STORAGE_PATH": "/tmp"})
    assert v.video_api_key.get_secret_value() == "legacy"


def test_google_capabilities(tmp_path):
    s = _google_settings(tmp_path)
    assert provider_supports("image", "text_to_image", s)
    assert provider_supports("image", "aspect_ratio_9_16", s)
    assert not provider_supports("image", "image_to_video", s)
    assert provider_supports("video", "image_to_video", s)
    assert provider_supports("video", "duration_control", s)
    assert not provider_supports("video", "text_to_image", s)
    caps_image = provider_capabilities("image", s)
    assert caps_image.name == "google"
    assert caps_image.generates == "image"
    caps_video = provider_capabilities("video", s)
    assert caps_video.name == "google"
    assert caps_video.supports_async is True


def test_health_google_configured(monkeypatch, tmp_path):
    s = _google_settings(tmp_path)
    monkeypatch.setattr("backend.health.get_settings", lambda: s)
    h = get_health()
    assert h["dependencies"]["image_provider"] == "CONFIGURED"
    assert h["dependencies"]["video_provider"] == "CONFIGURED"


def test_health_google_missing_key(monkeypatch, tmp_path):
    s = Settings(
        env={
            "LLM_PROVIDER": "mock",
            "IMAGE_PROVIDER": "google",
            "IMAGE_API_KEY": "",
            "VIDEO_PROVIDER": "google",
            "VIDEO_API_KEY": "",
            "STORAGE_PATH": str(tmp_path),
        }
    )
    monkeypatch.setattr("backend.health.get_settings", lambda: s)
    h = get_health()
    assert h["dependencies"]["image_provider"] == "NOT_CONFIGURED"
    assert h["dependencies"]["video_provider"] == "NOT_CONFIGURED"


def test_google_image_env_model_override(tmp_path):
    seen = {}

    def fake(method, url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return _resp(200, _image_body())

    s = _google_settings(tmp_path, IMAGE_MODEL="gemini-x-flash-image")
    p = GoogleImageProvider(s, request_func=fake)
    p.generate_images("x", count=1)
    assert "/models/gemini-x-flash-image:generateContent" in seen["url"]
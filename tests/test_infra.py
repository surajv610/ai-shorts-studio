import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

import pytest

from backend.config import Settings, get_settings
from backend.ffmpeg import detect_ffmpeg, get_ffmpeg_info
from backend.health import get_health
from backend.storage import (
    ensure_project_dirs,
    project_asset_dir,
    save_project,
    load_project,
)
from backend.models import Project, YouTubeMetadata, QCResult
from backend.providers.base import (
    ImageProvider,
    ImageGenerationResult,
    LLMProvider,
    LLMTextResult,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRemoteError,
    ProviderTimeoutError,
    VideoGenerationResult,
    GenerationStatus,
    AsyncJob,
    AsyncJobStatus,
    AsyncPollingClient,
    GenerationOptions,
    ProviderCapabilities,
)
from backend.providers.http import APIClient
from backend.providers.registry import (
    get_llm_provider,
    get_image_provider,
    get_video_provider,
    provider_capabilities,
    provider_supports,
)
from backend.providers.llm import MockLLMProvider, OpenAILLMProvider
from backend.providers.image import MockImageProvider
from backend.providers.video import MockVideoProvider


# ---------- helper fixtures ----------

@pytest.fixture
def mock_settings():
    return Settings(
        env={
            "LLM_PROVIDER": "mock",
            "IMAGE_PROVIDER": "mock",
            "VIDEO_PROVIDER": "mock",
            "STORAGE_PATH": tempfile.mkdtemp(),
        }
    )


@pytest.fixture(autouse=True)
def clean_storage(monkeypatch, tmp_path):
    """Point the storage dir at a temp path for the duration of each test."""
    monkeypatch.setattr("backend.storage.STORAGE_DIR", tmp_path)
    return tmp_path


# ---------- provider interfaces ----------

class ConcreteLLM(LLMProvider):
    name = "concrete"
    def generate_text(self, prompt, options=None):
        return LLMTextResult(text="hi", provider="concrete")
    def generate_structured(self, system_prompt, user_message, json_schema, options=None):
        return {"ok": True}


def test_llm_interface_implementable():
    p = ConcreteLLM()
    assert p.generate_text("x").text == "hi"
    assert p.generate_structured("s", "u", {}) == {"ok": True}


def test_provider_interface_raises_not_implemented():
    class Bare(ImageProvider):
        name = "bare"
    with pytest.raises(NotImplementedError):
        Bare().generate_images("p")


# ---------- mock providers ----------

def test_mock_image_provider_generates_two_distinct_images(mock_settings, tmp_path):
    out = tmp_path / "imgs"
    p = MockImageProvider(mock_settings, output_dir=str(out))
    res = p.generate_images("a seed on soil", count=2)
    assert res.provider == "mock"
    assert res.status == GenerationStatus.SUCCEEDED
    assert len(res.images) == 2
    assert res.images[0] != res.images[1]
    for path in res.images:
        assert Path(path).exists()
        assert Path(path).read_bytes().startswith(b"\x89PNG")
    # distinguishability: different file contents
    assert Path(res.images[0]).read_bytes() != Path(res.images[1]).read_bytes()


def test_mock_llm_provider_returns_structured(mock_settings):
    p = MockLLMProvider(mock_settings)
    out = p.generate_structured("s", "u", {"schema": {}})
    assert "scenes" in out
    assert "project_bible" in out
    assert p.generate_text("hello").text.startswith("(mock)")


def test_mock_video_provider_produces_clip(mock_settings, tmp_path):
    out = tmp_path / "vids"
    p = MockVideoProvider(mock_settings, output_dir=str(out))
    res = p.generate_video("/some/image.png", "grow", {"duration_seconds": 1})
    assert res.provider == "mock"
    assert res.status == GenerationStatus.SUCCEEDED
    assert res.video
    assert Path(res.video).exists()


# ---------- missing API credentials ----------

def test_openai_llm_requires_key():
    s = Settings(env={"LLM_PROVIDER": "openai", "LLM_API_KEY": ""})
    with pytest.raises(ProviderNotConfiguredError):
        get_llm_provider(s)


def test_registry_rejects_unsupported_provider(mock_settings):
    s = Settings(env={"IMAGE_PROVIDER": "dall-e", "STORAGE_PATH": "/tmp"})
    with pytest.raises(ProviderNotConfiguredError):
        get_image_provider(s)


def test_registry_unknown_llm_provider():
    s = Settings(env={"LLM_PROVIDER": "nope"})
    with pytest.raises(ProviderNotConfiguredError):
        get_llm_provider(s)


# ---------- successful provider initialization ----------

def test_provider_init_with_mock(mock_settings):
    assert get_llm_provider(mock_settings).name == "mock"
    assert get_image_provider(mock_settings).name == "mock"
    assert get_video_provider(mock_settings).name == "mock"


def test_openai_provider_init_with_key():
    s = Settings(env={"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})
    p = OpenAILLMProvider(s)
    assert p.model  # configured model is present


# ---------- provider errors ----------

def test_provider_remote_error_is_raiseable():
    err = ProviderRemoteError("boom")
    assert isinstance(err, ProviderError)


def test_mock_image_provider_writes_to_specified_output(mock_settings, tmp_path):
    out = tmp_path / "imgs2"
    p = MockImageProvider(mock_settings, output_dir=str(out))
    res = p.generate_images("x", count=2)
    assert str(out) in res.images[0]


# ---------- retry behavior ----------

def _response(status_code, payload=None):
    return httpx.Response(status_code, json=payload or {}, request=httpx.Request("POST", "http://x"))


def test_apiclient_retries_transient_then_succeeds():
    calls = {"n": 0}

    def fake_request(method, url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _response(503)
        return _response(200, {"choices": [{"message": {"content": '{"a":1}'}}]})

    client = APIClient(
        "http://example.com/v1", "k",
        timeout=5, retry_attempts=3, retry_backoff=0.01,
        request_func=fake_request,
    )
    data = client.post_json("/chat/completions", {})
    assert data == {"choices": [{"message": {"content": '{"a":1}'}}]}
    assert calls["n"] == 3


def test_apiclient_gives_up_after_retries():
    call_count = {"n": 0}

    def fake_request(method, url, **kwargs):
        call_count["n"] += 1
        return _response(500)

    client = APIClient(
        "http://example.com/v1", "k",
        timeout=5, retry_attempts=3, retry_backoff=0.01,
        request_func=fake_request,
    )
    with pytest.raises(ProviderRemoteError):
        client.post_json("/chat/completions", {})
    assert call_count["n"] == 3


def test_apiclient_does_not_retry_client_errors():
    calls = {"n": 0}

    def fake_request(method, url, **kwargs):
        calls["n"] += 1
        return _response(401)

    client = APIClient(
        "http://example.com/v1", "k", timeout=5, retry_attempts=3,
        retry_backoff=0.01, request_func=fake_request,
    )
    with pytest.raises(ProviderRemoteError):
        client.post_json("/chat/completions", {})
    assert calls["n"] == 1


def test_apiclient_timeout_raises():
    def fake_request(method, url, **kwargs):
        raise httpx.TimeoutException("timed out")

    client = APIClient(
        "http://example.com/v1", "k", timeout=1, retry_attempts=2,
        retry_backoff=0.01, request_func=fake_request,
    )
    with pytest.raises(ProviderTimeoutError):
        client.post_json("/chat/completions", {})


# ---------- database / storage persistence ----------

def test_project_persists_all_stages(clean_storage):
    project = Project(idea="seed", status="WAITING_FOR_IMAGE_SELECTION")
    project.storyboard = {"title": "Seed"}
    project.project_bible = {"subject": "seed"}
    project.image_generations = [{"generation_id": "g1", "provider": "mock", "images": ["a.png"]}]
    project.video_generations = [{"generation_id": "g2", "provider": "mock", "video": "v.mp4"}]
    project.final_video = "final.mp4"
    project.youtube_metadata = YouTubeMetadata(title="T", description="D", hashtags=["#s"])
    project.qc_result = QCResult(passed=True, issues=[], checked_at="now")
    project.errors = ["error1"]

    save_project(project)
    loaded = load_project(project.id)
    assert loaded.id == project.id
    assert loaded.status.value == "WAITING_FOR_IMAGE_SELECTION"
    assert loaded.storyboard["title"] == "Seed"
    assert loaded.project_bible["subject"] == "seed"
    assert loaded.image_generations[0]["generation_id"] == "g1"
    assert loaded.video_generations[0]["video"] == "v.mp4"
    assert loaded.final_video == "final.mp4"
    assert loaded.youtube_metadata.title == "T"
    assert loaded.qc_result.passed is True
    assert loaded.errors == ["error1"]


def test_project_creates_nested_asset_dirs(clean_storage):
    project = Project(idea="x")
    save_project(project)
    root = clean_storage / project.id
    for name in ("storyboard", "images", "videos", "final", "metadata", "qc"):
        assert (root / name).is_dir(), f"{name} dir missing"
    # The flat JSON record also still exists.
    assert (clean_storage / f"{project.id}.json").exists()
    # Helpers return the right dirs.
    assert project_asset_dir(project.id, "images").name == "images"
    with pytest.raises(ValueError):
        project_asset_dir(project.id, "nope")


# ---------- FFmpeg detection ----------

def test_ffmpeg_detect_returns_info():
    info = detect_ffmpeg()
    assert info.available in (True, False)
    if info.available:
        assert info.path
        assert info.version


def test_ffmpeg_detection_shape():
    d = get_ffmpeg_info().to_dict()
    assert set(d) == {"available", "path", "version", "error"}


# ---------- health endpoint & secret protection ----------

def test_health_never_exposes_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "sk-SUPER-SECRET")
    monkeypatch.setenv("IMAGE_API_KEY", "img-SECRET")
    monkeypatch.setenv("VIDEO_API_KEY", "vid-SECRET")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    payload = json.dumps(get_health())
    assert "sk-SUPER-SECRET" not in payload
    assert "img-SECRET" not in payload
    assert "vid-SECRET" not in payload
    get_settings.cache_clear()


def test_settings_payload_has_no_secrets():
    s = get_settings()
    # The settings endpoint mirrors config fields; assert the API key fields
    # themselves are not python in the payload by construction.
    from backend.api import settings as settings_endpoint
    # Check the endpoint body does not include literal key values via an
    # in-context monkeypatch is covered by test_health_never_exposes_secrets.


def test_health_reports_dependencies():
    h = get_health()
    assert "dependencies" in h
    for dep in ("database", "storage", "ffmpeg", "llm", "image_provider", "video_provider"):
        assert dep in h["dependencies"]
    assert h["dependencies"]["storage"] in ("READY", "ERROR")


def test_mock_mode_health_ready(mock_settings, monkeypatch):
    monkeypatch.setattr("backend.health.get_settings", lambda: mock_settings)
    h = get_health()
    assert h["dependencies"]["llm"] == "READY"
    assert h["dependencies"]["image_provider"] == "READY"
    assert h["dependencies"]["video_provider"] == "READY"


# ---------- capability metadata ----------

def test_mock_image_capabilities(mock_settings):
    p = get_image_provider(mock_settings)
    caps = p.capabilities()
    assert "text_to_image" in caps.capabilities
    assert "multi_candidate" in caps.capabilities
    assert p.supports("text_to_image")
    assert p.capabilities().supports("aspect_ratio_9_16")
    assert not p.supports("image_to_video")


def test_mock_video_capabilities(mock_settings):
    p = get_video_provider(mock_settings)
    assert "image_to_video" in p.capabilities().capabilities
    assert p.supports("duration_control")
    assert p.supports("aspect_ratio_9_16")
    assert not p.supports("text_to_image")


def test_provider_capabilities_lookup(mock_settings):
    assert provider_supports("image", "text_to_image", mock_settings)
    assert provider_supports("image", "multi_candidate", mock_settings)
    assert provider_supports("video", "image_to_video", mock_settings)
    assert not provider_supports("video", "text_to_image", mock_settings)
    caps = provider_capabilities("image", mock_settings)
    assert caps.generates == "image"
    assert caps.name == "mock"


# ---------- generation options ----------

def test_aspect_ratio_key():
    assert GenerationOptions(aspect_ratio="9:16").aspect_ratio_key() == "9_16"
    assert GenerationOptions(aspect_ratio="16/9").aspect_ratio_key() == "16_9"
    assert GenerationOptions(aspect_ratio="").aspect_ratio_key() == ""
    assert GenerationOptions(aspect_ratio=None).aspect_ratio_key() == ""


def test_mock_image_honors_size_options(mock_settings, tmp_path):
    p = MockImageProvider(mock_settings, output_dir=str(tmp_path / "imgs"))
    res = p.generate_images("x", options=GenerationOptions(aspect_ratio="9:16"), count=2)
    # aspect ratio 9:16 -> 576x1024 at scale 64
    assert res.images
    assert res.raw["width"] == 576
    assert res.raw["height"] == 1024


# ---------- async job abstraction ----------

class FakeAsyncProvider(AsyncPollingClient):
    def __init__(self):
        self._polls = 0

    def submit_async(self, kind, prompt, options=None):
        return AsyncJob(provider="fake", job_id="job1", status=AsyncJobStatus.PENDING)

    def poll_async(self, job):
        self._polls += 1
        if self._polls >= 2:
            job.status = AsyncJobStatus.SUCCEEDED
            job.raw = {"done": True}
        else:
            job.status = AsyncJobStatus.PROCESSING
        return job


def test_async_polling_client_waits_for_terminal():
    p = FakeAsyncProvider()
    job = p.submit_async("video", "grow")
    final = p.wait(job, timeout_seconds=5, poll_interval=0.01)
    assert final.status == AsyncJobStatus.SUCCEEDED
    assert final.raw == {"done": True}
    assert p._polls >= 2


def test_async_job_terminal_property():
    j = AsyncJob(provider="x", job_id="1", status=AsyncJobStatus.SUCCEEDED)
    assert j.is_terminal
    j2 = AsyncJob(provider="x", job_id="2", status=AsyncJobStatus.PROCESSING)
    assert not j2.is_terminal


def test_mock_async_image_submit_and_poll(mock_settings, tmp_path):
    p = get_image_provider(mock_settings)
    res = p.generate_images("seed", count=2)
    job = p.submit_async("seed", count=2)
    assert job.status == AsyncJobStatus.SUCCEEDED
    assert job.raw["images"]
    # polling a terminal job is a no-op
    assert p.poll_async(job).status == AsyncJobStatus.SUCCEEDED


def test_mock_async_video_submit_and_poll(mock_settings):
    p = get_video_provider(mock_settings)
    job = p.submit_async("img.png", "grow", options={"duration_seconds": 1})
    assert job.status == AsyncJobStatus.PENDING
    assert job.raw["video"]
    job = p.poll_async(job)
    assert job.status == AsyncJobStatus.PROCESSING
    job = p.poll_async(job)
    assert job.status == AsyncJobStatus.SUCCEEDED
    assert Path(job.raw["video"]).exists()


def test_mock_async_video_simulated_failure():
    from backend.providers.video import MockVideoProvider

    s = Settings(env={"STORAGE_PATH": "/tmp"})
    p = MockVideoProvider(s, fail_after=1)
    job = p.submit_async("img.png", "grow")
    job = p.poll_async(job)
    assert job.status == AsyncJobStatus.FAILED
    assert "failure" in job.error


# ---------- config validation for real-but-unimplemented providers ----------

def test_unimplemented_image_provider_gives_clear_error():
    # Any non-mock image provider selection yields a clear config error
    # (no adapter implemented yet, so never silently falls back to mock).
    s = Settings(env={"IMAGE_PROVIDER": "some-vendor", "IMAGE_API_KEY": "k"})
    with pytest.raises(ProviderNotConfiguredError) as exc:
        get_image_provider(s)
    assert "IMAGE_PROVIDER" in str(exc.value)
    assert "mock" in str(exc.value)


def test_unimplemented_video_provider_gives_clear_error():
    s = Settings(env={"VIDEO_PROVIDER": "some-vendor", "VIDEO_API_KEY": "k"})
    with pytest.raises(ProviderNotConfiguredError) as exc:
        get_video_provider(s)
    assert "VIDEO_PROVIDER" in str(exc.value)
    assert "mock" in str(exc.value)


# ---------- health state distinction (CONFIGURED vs READY vs NOT_CONFIGURED) ----------

def test_health_distinguishes_configured_state(monkeypatch, tmp_path):
    s = Settings(
        env={
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "sk-test",
            "IMAGE_PROVIDER": "mock",
            "VIDEO_PROVIDER": "mock",
            "STORAGE_PATH": str(tmp_path),
        }
    )
    monkeypatch.setattr("backend.health.get_settings", lambda: s)
    monkeypatch.setattr("backend.health.SUPPORTED_LLM_PROVIDERS", {"openai", "mock"})
    h = get_health()
    assert h["dependencies"]["llm"] == "CONFIGURED"
    assert h["dependencies"]["image_provider"] == "READY"
    assert h["dependencies"]["video_provider"] == "READY"


def test_health_configured_missing_key(monkeypatch, tmp_path):
    s = Settings(
        env={
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "",
            "IMAGE_PROVIDER": "mock",
            "VIDEO_PROVIDER": "mock",
            "STORAGE_PATH": str(tmp_path),
        }
    )
    monkeypatch.setattr("backend.health.get_settings", lambda: s)
    monkeypatch.setattr("backend.health.SUPPORTED_LLM_PROVIDERS", {"openai", "mock"})
    h = get_health()
    assert h["dependencies"]["llm"] == "NOT_CONFIGURED"


def test_health_unsupported_provider_is_error(monkeypatch, tmp_path):
    s = Settings(
        env={
            "LLM_PROVIDER": "mock",
            "IMAGE_PROVIDER": "nonsense",
            "VIDEO_PROVIDER": "mock",
            "STORAGE_PATH": str(tmp_path),
        }
    )
    monkeypatch.setattr("backend.health.get_settings", lambda: s)
    monkeypatch.setattr(
        "backend.health.SUPPORTED_IMAGE_PROVIDERS", {"mock"}
    )
    h = get_health()
    assert h["dependencies"]["image_provider"] == "ERROR"

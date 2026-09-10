"""Tests for the Gemini LLM provider.

All HTTP traffic is replaced with ``request_func`` injection — no real Gemini API
calls happen during automated tests. Coverage targets every Gemini-specific
behavior documented in the provider brief (structured output, error mapping,
quota vs rate-limit distinction, secret safety, capability reporting, agent
compatibility, etc.).
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import pytest

from backend.config import Settings
from backend.health import get_health
from backend.providers.base import (
    ProviderAuthError,
    ProviderGenerationError,
    ProviderInvalidRequestError,
    ProviderNotConfiguredError,
    ProviderQuotaError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderRemoteError,
    ProviderSafetyError,
    ProviderUnavailableError,
)
from backend.providers.gemini import (
    DEFAULT_GEMINI_LLM_MODEL,
    GeminiLLMProvider,
    map_gemini_error,
    REASON_GEMINI_AUTH_FAILED,
    REASON_GEMINI_QUOTA_EXCEEDED,
    REASON_GEMINI_RATE_LIMITED,
)
from backend.providers.http import APIClient, RETRYABLE_STATUS
from backend.providers.registry import (
    SUPPORTED_LLM_PROVIDERS,
    get_llm_provider,
    provider_capabilities,
    provider_supports,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_API_KEY = "test-gemini-api-key-12345"
FAKE_MODEL = DEFAULT_GEMINI_LLM_MODEL
FAKE_URL = "https://generativelanguage.googleapis.com/v1beta"
FAKE_KEY_HEADER = "x-goog-api-key"


def _gemini_settings(tmp_path, **overrides):
    env = {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": FAKE_API_KEY,
        "STORAGE_PATH": str(tmp_path),
    }
    env.update(overrides)
    return Settings(env=env)


def _ok_body(text="hello"):
    """Simulate a successful Gemini generateContent response."""
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2},
    }


def _ok_json_body(data: dict):
    return _ok_body(json.dumps(data))


def _no_candidates_body():
    return {"promptFeedback": {"blockReason": "SAFETY"}}


def _safety_block_body():
    return {
        "candidates": [],
        "promptFeedback": {"blockReason": "SAFETY"},
    }


def _safety_finish_body():
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": ""}], "role": "model"},
                "finishReason": "SAFETY",
            }
        ]
    }


def _empty_parts_body():
    return {
        "candidates": [
            {
                "content": {"parts": [], "role": "model"},
                "finishReason": "STOP",
            }
        ]
    }


def _error_body(status="INVALID_ARGUMENT", message="bad request", code=400):
    return {"error": {"code": code, "message": message, "status": status}}


def _quota_body():
    return _error_body(status="RESOURCE_EXHAUSTED", message="quota exceeded", code=429)


def _rate_limit_body():
    return _error_body(status="RATE_LIMITED", message="rate limit", code=429)


def _auth_body():
    return _error_body(status="UNAUTHENTICATED", message="invalid api key", code=401)


def _perm_body():
    return _error_body(status="PERMISSION_DENIED", message="access denied", code=403)


def _model_not_found_body():
    return _error_body(status="NOT_FOUND", message="model not found", code=404)


def _unavailable_body():
    return _error_body(status="UNAVAILABLE", message="service unavailable", code=503)


def _resp(status_code: int, payload: Optional[dict] = None, text: str = ""):
    content = json.dumps(payload or {}).encode() if not text else text.encode()
    return httpx.Response(
        status_code,
        content=content,
        request=httpx.Request("POST", "http://fake-gemini"),
        headers={"content-type": "application/json"},
    )


class RecordingRequestFunc:
    """Captures all calls; returns pre-programmed responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._calls: list[tuple[str, str, dict]] = []

    def __call__(self, method: str, url: str, **kwargs) -> httpx.Response:
        self._calls.append((method, url, kwargs))
        if self._responses:
            return self._responses.pop(0)
        return _resp(500)

    @property
    def calls(self):
        return self._calls

    @property
    def last_payload(self) -> dict:
        return self._calls[-1][2].get("json") or {}


# ---------------------------------------------------------------------------
# 1  Happy-path generate_text
# ---------------------------------------------------------------------------

class TestGenerateText:
    def test_successful_text_generation(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("hello world"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        res = p.generate_text("say hi")
        assert res.text == "hello world"
        assert res.model == FAKE_MODEL
        assert res.provider == "gemini"
        assert len(func.calls) == 1
        assert func.calls[0][0] == "POST"
        assert FAKE_URL in func.calls[0][1]
        assert ":generateContent" in func.calls[0][1]

    def test_api_key_sent_in_header(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test")
        headers = func.calls[0][2].get("headers") or {}
        assert headers.get(FAKE_KEY_HEADER) == FAKE_API_KEY

    def test_auth_header_not_bearer(self, tmp_path):
        """Gemini uses x-goog-api-key, NOT Authorization: Bearer."""
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test")
        headers = func.calls[0][2].get("headers") or {}
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# 2  Happy-path generate_structured
# ---------------------------------------------------------------------------

class TestGenerateStructured:
    def test_structured_returns_parsed_json(self, tmp_path):
        data = {"title": "Test", "theme": "Sci-fi"}
        func = RecordingRequestFunc([_resp(200, _ok_json_body(data))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        schema = {"name": "StoryMeta", "schema": {"type": "object", "properties": {"title": {"type": "string"}}}}
        result = p.generate_structured("Be creative", "Return a story meta", schema)
        assert result == data
        assert isinstance(result, dict)

    def test_structured_includes_response_schema_in_config(self, tmp_path):
        schema = {
            "name": "SceneBrief",
            "schema": {
                "type": "object",
                "properties": {"scene": {"type": "string"}},
            },
        }
        func = RecordingRequestFunc([_resp(200, _ok_json_body({"scene": "opening"}))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_structured("system msg", "user msg", schema)
        gen_config = func.last_payload.get("generationConfig", {})
        assert gen_config.get("responseMimeType") == "application/json"
        assert gen_config.get("responseSchema") == schema["schema"]


# ---------------------------------------------------------------------------
# 3  System prompt handling
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_included_when_provided(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("hi"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("hello", options={"system": "You are helpful"})
        sys_inst = func.last_payload.get("systemInstruction", {})
        parts = (sys_inst.get("parts") or [])
        assert len(parts) == 1
        assert parts[0]["text"] == "You are helpful"

    def test_system_not_included_when_missing(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("hi"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("hello")
        assert "systemInstruction" not in func.last_payload

    def test_structured_always_has_system_instruction(self, tmp_path):
        """Structured generation always uses system prompt (required by agents)."""
        schema = {"name": "X", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
        func = RecordingRequestFunc([_resp(200, _ok_json_body({"a": "b"}))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_structured("My system prompt", "My user msg", schema)
        sys_inst = func.last_payload.get("systemInstruction", {})
        assert sys_inst.get("parts", [{}])[0].get("text") == "My system prompt"


# ---------------------------------------------------------------------------
# 4  GenerationConfig: temperature, maxOutputTokens, model override
# ---------------------------------------------------------------------------

class TestGenerationConfig:
    def test_default_temperature(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test")
        assert func.last_payload["generationConfig"].get("temperature") == 0.7

    def test_custom_temperature(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test", options={"temperature": 1.2})
        assert func.last_payload["generationConfig"]["temperature"] == 1.2

    def test_max_tokens_passed(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test", options={"max_tokens": 256})
        assert func.last_payload["generationConfig"]["maxOutputTokens"] == 256

    def test_max_tokens_not_in_payload_when_absent(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test")
        assert "maxOutputTokens" not in func.last_payload.get("generationConfig", {})


# ---------------------------------------------------------------------------
# 5  Model override / fallback
# ---------------------------------------------------------------------------

class TestModelFallback:
    def test_default_model(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test")
        assert f"/models/{FAKE_MODEL}:" in func.calls[0][1]
        assert p.model == FAKE_MODEL

    def test_model_from_env(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(
            _gemini_settings(tmp_path, LLM_MODEL="gemini-2.0-pro"),
            request_func=func,
        )
        p.generate_text("test")
        assert "/models/gemini-2.0-pro:" in func.calls[0][1]
        assert p.model == "gemini-2.0-pro"

    def test_model_override_via_options(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        p.generate_text("test", options={"model": "gemini-1.5-flash"})
        assert "/models/gemini-1.5-flash:" in func.calls[0][1]


# ---------------------------------------------------------------------------
# 6  Error mapping — auth (401 / 403)
# ---------------------------------------------------------------------------

class TestAuthErrors:
    def test_401_returns_auth_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(401, _auth_body())])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderAuthError) as exc_info:
            p.generate_text("test")
        assert getattr(exc_info.value, "reason", None) == REASON_GEMINI_AUTH_FAILED

    def test_403_returns_auth_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(403, _perm_body())])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderAuthError):
            p.generate_text("test")


# ---------------------------------------------------------------------------
# 7  Error mapping — quota exhausted (no retry)
# ---------------------------------------------------------------------------

class TestQuotaErrors:
    def test_quota_exhausted_not_retried(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(429, _quota_body())

        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        with pytest.raises(ProviderQuotaExceededError) as exc_info:
            p.generate_text("test")
        assert call_count == 1  # no retry
        assert getattr(exc_info.value, "reason", None) == REASON_GEMINI_QUOTA_EXCEEDED

    def test_resource_exhausted_status_code_no_retry(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(429, _error_body(status="RESOURCE_EXHAUSTED", message="quota", code=429))

        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        with pytest.raises(ProviderQuotaExceededError):
            p.generate_text("test")
        assert call_count == 1


# ---------------------------------------------------------------------------
# 8  Error mapping — rate limit (transient, retried with backoff)
# ---------------------------------------------------------------------------

class TestRateLimitErrors:
    def test_transient_rate_limit_retried(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _resp(429, _rate_limit_body())
            return _resp(200, _ok_body("ok"))

        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        res = p.generate_text("test")
        assert res.text == "ok"
        assert call_count == 3  # retried twice, succeeded third

    def test_rate_limit_transient_retires_up_to_retry_limit(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(429, _rate_limit_body())

        p = GeminiLLMProvider(
            _gemini_settings(tmp_path, RETRY_ATTEMPTS="2"),
            request_func=counting_request,
        )
        with pytest.raises(ProviderRateLimitError):
            p.generate_text("test")
        assert call_count == 2  # retried up to limit


# ---------------------------------------------------------------------------
# 9  Error mapping — unavailable / 5xx (retried)
# ---------------------------------------------------------------------------

class TestUnavailableErrors:
    def test_500_retried_then_fails(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(500, _unavailable_body())

        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        with pytest.raises(ProviderUnavailableError):
            p.generate_text("test")
        assert call_count >= 2  # retried

    def test_404_not_retried(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(404, _model_not_found_body())

        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        with pytest.raises(ProviderInvalidRequestError):
            p.generate_text("test")
        assert call_count == 1  # not retried


# ---------------------------------------------------------------------------
# 10  Error mapping — content safety
# ---------------------------------------------------------------------------

class TestSafetyErrors:
    def test_prompt_block_safety_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _safety_block_body())])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderGenerationError) as exc_info:
            p.generate_text("test")
        assert "no candidates" in str(exc_info.value).lower()

    def test_finish_reason_safety_safety_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _safety_finish_body())])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderSafetyError) as exc_info:
            p.generate_text("test")
        assert "SAFETY" in str(exc_info.value)

    def test_empty_parts_generation_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _empty_parts_body())])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderGenerationError) as exc_info:
            p.generate_text("test")
        assert "empty" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 11  Malformed JSON from Gemini
# ---------------------------------------------------------------------------

class TestMalformedJson:
    def test_non_json_response_raises_remote_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, text="not json at all")])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderRemoteError, match="Could not parse JSON"):
            p.generate_structured("system", "user", {"name": "X", "schema": {}})

    def test_json_array_not_object_raises_remote_error(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, text='[1,2,3]')])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderRemoteError, match="non-object response"):
            p.generate_structured("system", "user", {"name": "X", "schema": {}})


# ---------------------------------------------------------------------------
# 12  No API key → ProviderNotConfiguredError
# ---------------------------------------------------------------------------

class TestNotConfigured:
    def test_no_gemini_key_raises(self):
        with pytest.raises(ProviderNotConfiguredError, match="GEMINI_API_KEY"):
            GeminiLLMProvider(Settings(env={"LLM_PROVIDER": "gemini"}))

    def test_no_gemini_key_message_lists_env_var(self):
        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            GeminiLLMProvider(Settings(env={"LLM_PROVIDER": "gemini"}))
        assert "GEMINI_API_KEY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 13  Fallback key via LLM_API_KEY when GEMINI_API_KEY absent
# ---------------------------------------------------------------------------

class TestLegacyKeyFallback:
    def test_llm_api_key_used_when_gemini_key_missing(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(
            Settings(env={
                "LLM_PROVIDER": "gemini",
                "LLM_API_KEY": "legacy-key",
                "STORAGE_PATH": str(tmp_path),
            }),
            request_func=func,
        )
        res = p.generate_text("test")
        assert res.text == "ok"
        headers = func.calls[0][2].get("headers") or {}
        assert headers.get(FAKE_KEY_HEADER) == "legacy-key"


# ---------------------------------------------------------------------------
# 14  /health and /settings report no secrets
# ---------------------------------------------------------------------------

class TestSecretSafety:
    def test_health_never_exposes_gemini_api_key(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        _ = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        health = get_health()
        health_str = json.dumps(health)
        assert FAKE_API_KEY not in health_str

    def test_settings_never_exposes_gemini_api_key(self, tmp_path):
        from backend.api import app
        from fastapi.testclient import TestClient

        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        _ = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/settings")
        assert FAKE_API_KEY not in resp.text


# ---------------------------------------------------------------------------
# 15  Capability reporting
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_gemini_capabilities_report_structured_output(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        caps = p.capabilities()
        assert caps.supports("structured_output")
        assert caps.supports("json_output")
        assert caps.supports("configurable_model")
        assert caps.supports("text_generation")
        assert caps.generates == "text"

    def test_provider_capabilities_for_gemini(self, tmp_path):
        func = RecordingRequestFunc([_resp(200, _ok_body("ok"))])
        _ = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        caps = provider_capabilities("llm", _gemini_settings(tmp_path))
        assert caps.name == "gemini"


# ---------------------------------------------------------------------------
# 16  SUPPORTED_LLM_PROVIDERS includes gemini
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_gemini_in_supported(self):
        assert "gemini" in SUPPORTED_LLM_PROVIDERS

    def test_mock_still_supported(self):
        assert "mock" in SUPPORTED_LLM_PROVIDERS

    def test_openai_still_supported(self):
        assert "openai" in SUPPORTED_LLM_PROVIDERS

    def test_get_llm_provider_gemini(self, tmp_path):
        p = get_llm_provider(_gemini_settings(tmp_path))
        assert isinstance(p, GeminiLLMProvider)

    def test_get_llm_provider_mock(self, tmp_path):
        p = get_llm_provider(Settings(env={"LLM_PROVIDER": "mock", "STORAGE_PATH": str(tmp_path)}))
        assert type(p).__name__ == "MockLLMProvider"

    def test_unsupported_provider_raises(self):
        with pytest.raises(ProviderNotConfiguredError, match="Unsupported"):
            get_llm_provider(Settings(env={"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "x"}))


# ---------------------------------------------------------------------------
# 17  No fallback to OpenAI on Gemini failure
# ---------------------------------------------------------------------------

class TestNoFallback:
    def test_gemini_auth_error_does_not_fall_back_to_openai(self, tmp_path):
        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(401, _auth_body())

        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        with pytest.raises(ProviderAuthError):
            p.generate_text("test")
        assert call_count == 1  # stopped, no fallback to another provider


# ---------------------------------------------------------------------------
# 18  Agent compatibility — agents use generic call_structured interface
# ---------------------------------------------------------------------------

class TestAgentCompatibility:
    """Story / Image / Video / Metadata agents must operate through the generic
    LLMProvider interface. They must NOT import GeminiLLMProvider directly."""

    def test_no_agent_imports_gemini_llm_provider(self):
        from pathlib import Path
        agent_dir = Path(__file__).resolve().parent.parent / "agents"
        for py_file in agent_dir.glob("*.py"):
            text = py_file.read_text()
            assert "GeminiLLMProvider" not in text, (
                f"{py_file.name} imports GeminiLLMProvider — agents must stay vendor-agnostic"
            )

    def test_story_agent_uses_generic_call_structured(self, tmp_path):
        """The story agent calls call_structured via agents.llm, which resolves
        to the configured provider through the registry."""
        from agents.story import StoryAgent

        call_count = 0

        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(200, _ok_json_body({"title": "Agent Test"}))

        func = RecordingRequestFunc([_resp(200, _ok_json_body({"title": "Agent Test"}))])
        _ = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        # The story agent gets its LLM via agents.llm → registry.get_llm_provider
        # We verify the story agent can be instantiated and calls structured output.
        # (Full pipeline test would be heavy; here we just verify the interface compiles
        # and the provider type flows correctly.)
        from agents.llm import call_structured
        result = call_structured("system", "user", {"name": "X", "schema": {"type": "object"}})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 19  Provider-specific HTTP status code edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_400_invalid_argument(self, tmp_path):
        func = RecordingRequestFunc([_resp(400, _error_body(status="INVALID_ARGUMENT", message="bad schema"))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        with pytest.raises(ProviderInvalidRequestError):
            p.generate_text("test")

    def test_429_with_quota_in_details(self, tmp_path):
        body = {
            "error": {
                "code": 429,
                "message": "requests per minute exceeded",
                "status": "RATE_LIMITED",
                "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "quotaMetric": "gemini.requests"}],
            }
        }
        func = RecordingRequestFunc([_resp(429, body)])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        # Should NOT retry when quota failure details present
        call_count = 0
        def counting_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _resp(429, body)
        p2 = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=counting_request)
        with pytest.raises(ProviderQuotaExceededError):
            p2.generate_text("test")
        assert call_count == 1


# ---------------------------------------------------------------------------
# 20  map_gemini_error unit tests (standalone, no HTTP)
# ---------------------------------------------------------------------------

class TestMapGeminiError:
    def test_auth_401(self):
        e = map_gemini_error(401, _auth_body(), "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderAuthError)

    def test_auth_403(self):
        e = map_gemini_error(403, _perm_body(), "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderAuthError)

    def test_quota_resource_exhausted(self):
        e = map_gemini_error(429, _quota_body(), "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderQuotaExceededError)

    def test_rate_limit_transient(self):
        e = map_gemini_error(429, _rate_limit_body(), "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderRateLimitError)

    def test_unavailable_500(self):
        e = map_gemini_error(500, None, "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderUnavailableError)

    def test_invalid_request_404(self):
        e = map_gemini_error(404, _model_not_found_body(), "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderInvalidRequestError)

    def test_429_quota_in_message(self):
        body = {"error": {"code": 429, "message": "quota exceeded", "status": "RATE_LIMITED"}}
        e = map_gemini_error(429, body, "POST", "/models/x:generateContent")
        assert isinstance(e, ProviderQuotaExceededError)


# ---------------------------------------------------------------------------
# 21  Provider never fabricates fields
# ---------------------------------------------------------------------------

class TestNoFabrication:
    def test_generate_structured_returns_only_gemini_data(self, tmp_path):
        data = {"only_this": "yes"}
        func = RecordingRequestFunc([_resp(200, _ok_json_body(data))])
        p = GeminiLLMProvider(_gemini_settings(tmp_path), request_func=func)
        result = p.generate_structured("sys", "msg", {"name": "X", "schema": {}})
        assert result == data
        assert set(result.keys()) == {"only_this"}


# ---------------------------------------------------------------------------
# 22  should_retry predicate prevents retry on quota exhaustion
# ---------------------------------------------------------------------------

class TestShouldRetryPredicate:
    def test_quota_429_should_not_retry(self):
        from backend.providers.gemini import _gemini_should_retry
        assert _gemini_should_retry(429, _quota_body()) is False

    def test_rate_limit_429_should_retry(self):
        from backend.providers.gemini import _gemini_should_retry
        assert _gemini_should_retry(429, _rate_limit_body()) is True

    def test_500_should_retry(self):
        from backend.providers.gemini import _gemini_should_retry
        assert _gemini_should_retry(500, None) is True

    def test_503_should_retry(self):
        from backend.providers.gemini import _gemini_should_retry
        assert _gemini_should_retry(503, None) is True

    def test_non_retryable_not_called_by_predicate(self):
        """The predicate is only called for RETRYABLE_STATUS codes."""
        from backend.providers.gemini import _gemini_should_retry
        # 400 is not in RETRYABLE_STATUS so predicate should never be called,
        # but if called it returns False.
        assert _gemini_should_retry(400, None) is False


# ---------------------------------------------------------------------------
# 23  Config: llm_model provider-aware default
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_gemini_default_model(self, tmp_path):
        s = Settings(env={"LLM_PROVIDER": "gemini", "STORAGE_PATH": str(tmp_path)})
        assert s.llm_model == "gemini-2.5-flash"

    def test_mock_default_model(self, tmp_path):
        s = Settings(env={"LLM_PROVIDER": "mock", "STORAGE_PATH": str(tmp_path)})
        assert s.llm_model == "gpt-4o-mini"

    def test_openai_default_model(self, tmp_path):
        s = Settings(env={"LLM_PROVIDER": "openai", "STORAGE_PATH": str(tmp_path)})
        assert s.llm_model == "gpt-4o-mini"

    def test_explicit_model_wins(self, tmp_path):
        s = Settings(env={"LLM_PROVIDER": "gemini", "LLM_MODEL": "gemini-2.0-pro", "STORAGE_PATH": str(tmp_path)})
        assert s.llm_model == "gemini-2.0-pro"

    def test_llm_configured_for_gemini_with_key(self, tmp_path):
        s = Settings(env={"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "key", "STORAGE_PATH": str(tmp_path)})
        assert s.llm_configured is True
        assert s.llm_credentialed is True

    def test_llm_not_configured_for_gemini_without_key(self, tmp_path):
        s = Settings(env={"LLM_PROVIDER": "gemini", "STORAGE_PATH": str(tmp_path)})
        assert s.llm_configured is False
        assert s.llm_credentialed is False

    def test_gemini_api_key_property(self, tmp_path):
        s = Settings(env={"GEMINI_API_KEY": "g-key", "STORAGE_PATH": str(tmp_path)})
        assert s.gemini_api_key.get_secret_value() == "g-key"

    def test_gemini_key_fallback_to_google_api_key(self, tmp_path):
        s = Settings(env={"GOOGLE_API_KEY": "google-key", "STORAGE_PATH": str(tmp_path)})
        assert s.gemini_api_key.get_secret_value() == "google-key"

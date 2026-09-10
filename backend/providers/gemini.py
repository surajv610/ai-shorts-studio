"""Google Gemini LLM provider (Gemini API ``generateContent``).

Implements the existing :class:`LLMProvider` interface for text and structured
(JSON-schema-constrained) generation. All vendor-specific behavior lives in this
provider; Story / Image / Video / Metadata agents never know which vendor they
are talking to.

Key behaviors:

* Structured output uses ``responseMimeType: application/json`` plus
  ``responseSchema`` (taken directly from the existing agent schemas — nothing
  is duplicated here).
* Errors are mapped into the shared ``ProviderError`` hierarchy. Quota
  exhaustion and temporary rate limiting are distinguished
  (``ProviderQuotaExceededError`` vs ``ProviderRateLimitError``) and surfaced as
  ``GEMINI_QUOTA_EXCEEDED`` / ``GEMINI_RATE_LIMITED``.
* Retrying follows the shared exponential-backoff client. Only safe transient
  failures are retried (network/timeout, 5xx, rate limit); invalid credentials,
  invalid model, malformed requests, content rejection and quota exhaustion are
  NEVER retried.
* The provider NEVER falls back to OpenAI or any other vendor. On failure it
  stops and reports the problem.
* Secrets are sent as an ``x-goog-api-key`` header and are never logged or
  included in error messages.
"""

import json
from typing import Callable, Optional

from backend.config import Settings, get_settings
from backend.providers.base import (
    LLMProvider,
    LLMTextResult,
    ProviderAuthError,
    ProviderCapabilities,
    ProviderError,
    ProviderGenerationError,
    ProviderInvalidRequestError,
    ProviderNotConfiguredError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderRemoteError,
    ProviderSafetyError,
    ProviderUnavailableError,
)
from backend.providers.http import APIClient, RETRYABLE_STATUS

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

# Public, secret-free reasons surfaced to callers/UI.
REASON_GEMINI_QUOTA_EXCEEDED = "GEMINI_QUOTA_EXCEEDED"
REASON_GEMINI_RATE_LIMITED = "GEMINI_RATE_LIMITED"
REASON_GEMINI_AUTH_FAILED = "GEMINI_AUTH_FAILED"

_SAFETY_MARKERS = (
    "safety", "blocked", "block_reason", "blockreason", "finish_reason",
    "finishreason", "prohibited", "recitation", "blocklist", "candidate:",
    "prompt_feedback", "promptfeedback",
)

_QUOTA_STATUS_CODES = {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"}


def _err_of(body: Optional[dict]) -> dict:
    return body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}


def _error_message(body: Optional[dict]) -> str:
    err = _err_of(body)
    msg = str(err.get("message") or "")
    # Prefer the Google API status code when there is no human message.
    if not msg:
        msg = f"Gemini API error: {err.get('status') or err.get('code') or 'unknown'}"
    return msg


def _is_quota_exhausted(body: Optional[dict]) -> bool:
    """True when a Gemini error body is quota exhaustion (not a transient limit)."""
    err = _err_of(body)
    status = str(err.get("status") or "").upper()
    if status in _QUOTA_STATUS_CODES:
        return True
    message = str(err.get("message") or "").lower()
    if "quota" in message or "resource exhausted" in message:
        return True
    details = err.get("details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            # Check for Google RPC QuotaFailure details or any quota-related field
            detail_type = str(detail.get("@type") or "").lower()
            hay = " ".join(
                str(v).lower() for v in detail.values() if isinstance(v, (str, int, float))
            )
            if (
                "quota" in hay
                or "resource_exhausted" in hay
                or "quotafailure" in detail_type
                or "quota" in detail_type
            ):
                return True
    return False


def map_gemini_error(
    status: int,
    body: Optional[dict],
    method: str,
    path: str,
) -> ProviderError:
    """Map a Gemini API error (HTTP status + JSON body) to a ProviderError.

    Does not retry quota exhaustion. Never includes secrets in the message.
    """
    message = _error_message(body)
    lowered = (message + " " + str(body or "")).lower()
    err = _err_of(body)
    status_code = str(err.get("status") or "").upper()

    if _mentions_safety(lowered):
        return ProviderSafetyError(message)

    if status in (401, 403) or status_code in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        exc = ProviderAuthError(message or f"{method} {path}: Gemini authentication failed")
        exc.reason = REASON_GEMINI_AUTH_FAILED
        return exc

    if _is_quota_exhausted(body) or status_code in _QUOTA_STATUS_CODES:
        exc = ProviderQuotaExceededError(
            "Gemini quota exceeded - the generation was stopped. "
            "Check the API account's current quota/billing; this app will never "
            "silently switch to another provider."
            + (f" ({message})" if message and message.lower() != "unknown" else "")
        )
        exc.category = "quota_exhausted"
        exc.reason = REASON_GEMINI_QUOTA_EXCEEDED
        return exc

    if status == 429:
        exc = ProviderRateLimitError(
            f"Gemini rate limit hit - retried safely and still failing. {message}"
        )
        exc.category = "rate_limit"
        exc.reason = REASON_GEMINI_RATE_LIMITED
        return exc

    if status_code in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED"):
        return ProviderUnavailableError(message or f"{method} {path}: Gemini unavailable")

    if status >= 500:
        return ProviderUnavailableError(message or f"{method} {path}: Gemini upstream error")

    return ProviderInvalidRequestError(message or f"{method} {path} returned HTTP {status}")


def _mentions_safety(text: str) -> bool:
    return any(marker in text for marker in _SAFETY_MARKERS)


def _gemini_should_retry(status: int, body: Optional[dict]) -> bool:
    """Retry transient failures only. NEVER retry quota exhaustion."""
    if status == 429 and _is_quota_exhausted(body):
        return False
    if status == 429:
        return True  # transient rate limit: safe to retry with backoff
    return status in RETRYABLE_STATUS


class GeminiLLMProvider(LLMProvider):
    """Google Gemini text/structured generation via the Gemini API."""

    name = "gemini"

    _capabilities = ProviderCapabilities(
        name="gemini",
        generates="text",
        capabilities={
            "text_generation",
            "structured_output",
            "json_output",
            "configurable_model",
        },
        supports_async=False,
        description=(
            "Google Gemini language model via the Gemini API generateContent "
            "endpoint (text + JSON-schema-constrained structured output)."
        ),
        models=[DEFAULT_GEMINI_LLM_MODEL],
    )

    def capabilities(self):
        return self._capabilities

    def supports(self, capability: str) -> bool:
        return self._capabilities.supports(capability)

    def __init__(
        self,
        settings: Optional[Settings] = None,
        request_func: Optional[Callable] = None,
    ):
        self.settings = settings or get_settings()
        self.api_key = (
            self.settings.gemini_api_key.get_secret_value()
            or self.settings.llm_api_key.get_secret_value()
        )
        if not self.api_key:
            raise ProviderNotConfiguredError(
                "LLM_PROVIDER=gemini but no API key is set. "
                "Set GEMINI_API_KEY (or legacy GOOGLE_API_KEY / LLM_API_KEY) "
                "and optionally LLM_MODEL, or use LLM_PROVIDER=mock for development."
            )
        self.model = self.settings.llm_model or DEFAULT_GEMINI_LLM_MODEL
        self.base_url = self.settings.llm_base_url or DEFAULT_GEMINI_BASE_URL
        self._client = APIClient(
            base_url=self.base_url,
            headers={"x-goog-api-key": self.api_key},
            timeout=self.settings.llm_timeout,
            retry_attempts=self.settings.retry_attempts,
            retry_backoff=self.settings.retry_backoff,
            retry_max_delay=self.settings.retry_max_delay,
            request_func=request_func,
            error_parser=map_gemini_error,
            should_retry=_gemini_should_retry,
        )

    # -- request building -------------------------------------------------

    def _generation_config(self, options: dict, response_schema: Optional[dict]) -> dict:
        config: dict = {
            "temperature": options.get("temperature", DEFAULT_TEMPERATURE),
        }
        max_tokens = options.get("max_tokens")
        if max_tokens:
            config["maxOutputTokens"] = int(max_tokens)
        if response_schema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = response_schema
        return config

    def _build_payload(
        self,
        options: dict,
        *,
        system: Optional[str] = None,
        messages: list,
        response_schema: Optional[dict] = None,
    ) -> dict:
        payload: dict = {
            "contents": messages,
            "generationConfig": self._generation_config(options, response_schema),
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _call(self, model: str, payload: dict) -> dict:
        try:
            data = self._client.post_json(f"/models/{model}:generateContent", payload)
        except (ProviderError, ProviderRemoteError):
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise ProviderGenerationError(f"Gemini request failed: {e}")
        if not isinstance(data, dict):
            raise ProviderRemoteError(
                f"Gemini returned non-object response ({type(data).__name__}) "
                f"for /models/{model}:generateContent"
            )
        return data

    # -- response parsing -------------------------------------------------

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise ProviderGenerationError(
                f"Gemini returned no candidates (block_reason={block})"
            )
        candidate = candidates[0]
        finish = str((candidate or {}).get("finishReason") or "").upper()
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"):
            raise ProviderSafetyError(f"Gemini blocked the response: {finish}")
        parts = ((candidate or {}).get("content") or {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        text = "".join(texts).strip()
        if not text:
            raise ProviderGenerationError(
                f"Gemini returned empty content (finish_reason={finish})"
            )
        return text

    # -- public interface -------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        options: Optional[dict] = None,
    ) -> LLMTextResult:
        options = options or {}
        model = options.get("model") or self.model
        payload = self._build_payload(
            options,
            system=options.get("system"),
            messages=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        data = self._call(model, payload)
        return LLMTextResult(
            text=self._extract_text(data),
            model=model,
            provider=self.name,
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict,
        options: Optional[dict] = None,
    ) -> dict:
        options = options or {}
        model = options.get("model") or self.model
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        payload = self._build_payload(
            options,
            system=system_prompt,
            messages=[{"role": "user", "parts": [{"text": user_message}]}],
            response_schema=schema,
        )
        data = self._call(model, payload)
        text = self._extract_text(data)
        try:
            parsed = json.loads(text)
        except ValueError as e:
            raise ProviderRemoteError(
                f"Gemini returned malformed JSON for '{json_schema.get('name', 'structured')}': {e}"
            )
        if not isinstance(parsed, dict):
            raise ProviderRemoteError(
                f"Gemini returned non-object JSON for '{json_schema.get('name', 'structured')}'"
            )
        return parsed
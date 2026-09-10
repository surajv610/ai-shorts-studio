"""Reusable HTTP API client with timeout, retries, and exponential backoff.

This is provider-agnostic. Any provider that talks to a REST API can reuse it.
Secrets (API keys) are passed in headers/body by the caller and are never
logged. Tests can inject a request function to avoid real network calls.
"""

import random
import time
from typing import Any, Callable, Optional

import httpx

from backend.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderQuotaError,
    ProviderRemoteError,
    ProviderSafetyError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

# Status codes that are safe to retry (transient failures).
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Error mapping helpers
# ---------------------------------------------------------------------------

_SAFETY_MARKERS = (
    "safety",
    "blocked",
    "block_reason",
    "blockreason",
    "finish_reason",
    "finishreason",
    "content_filter",
    "prompt_feedback",
    "promptfeedback",
)


def map_http_error(
    status: int,
    body: Optional[dict],
    method: str,
    path: str,
) -> ProviderError:
    """Map an HTTP error status + JSON body to the specific ProviderError subclass.

    Distinguishes authentication failure, invalid request, quota/rate limit,
    content/safety rejection, provider unavailable, and generic failure.
    """
    message = _describe(body) or f"{method} {path} returned HTTP {status}"

    if status in (401, 403):
        return ProviderAuthError(message)
    if status == 429:
        return ProviderQuotaError(message)
    if status >= 500:
        return ProviderUnavailableError(message)
    # 400/404/410/422 and other client errors.
    if _mentions_safety(message, body):
        return ProviderSafetyError(message)
    return ProviderInvalidRequestError(message)


def _describe(body: Optional[dict]) -> str:
    """Extract a readable error message from a provider error body."""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("status") or err.get("code")
        if msg:
            return str(msg)
        detail = err.get("details")
        if isinstance(detail, list) and detail and isinstance(detail[0], dict):
            return str(detail[0].get("message") or detail[0])
    if isinstance(body.get("message"), str):
        return body["message"]
    return ""


def _mentions_safety(message: str, body: Optional[dict]) -> bool:
    lowered = message.lower()
    if any(marker in lowered for marker in _SAFETY_MARKERS):
        return True
    text = str(body or "").lower()
    return any(marker in text for marker in _SAFETY_MARKERS)


class APIClient:
    """Synchronous HTTP client with retry + exponential backoff.

    Usage:
        client = APIClient(base_url, api_key, timeout=30)
        data = client.post_json("/chat/completions", payload)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 120.0,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
        retry_max_delay: float = 30.0,
        headers: Optional[dict] = None,
        request_func: Optional[Callable[..., httpx.Response]] = None,
        error_parser: Optional[Callable[[int, Optional[dict], str, str], "ProviderError"]] = None,
        should_retry: Optional[Callable[[int, Optional[dict]], bool]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff
        self._retry_max_delay = retry_max_delay
        self._extra_headers = dict(headers or {})
        # Optional injection points for tests.
        self._request_func = request_func
        # Optional callable used to map non-retryable HTTP errors to specific
        # ProviderError subclasses. Signature: (status_code, json_body, method, path).
        self._error_parser = error_parser
        # Optional predicate ``(status_code, json_body) -> bool`` deciding whether
        # to retry a response. Defaults to the generic RETRYABLE_STATUS set. Lets
        # providers refuse to retry non-transient failures (e.g. quota exhaustion).
        self._should_retry = should_retry

    def _headers(self, json: bool) -> dict:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if json:
            headers["Content-Type"] = "application/json"
        headers.update(self._extra_headers)
        return headers

    def _perform(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        # Allow absolute URLs (e.g. downloading a temp video/audio object URL
        # returned by a provider) in addition to paths relative to base_url.
        if path.startswith(("http://", "https://")):
            url = path
        else:
            url = self._base_url + path
        if self._request_func is not None:
            return self._request_func(method, url, **kwargs)

        with httpx.Client(timeout=self._timeout) as client:
            return client.request(method, url, **kwargs)

    def request(self, method: str, path: str, json: Optional[dict] = None, **kwargs: Any) -> httpx.Response:
        """Send a request with retry + exponential backoff."""
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self._retry_attempts:
            attempt += 1
            try:
                resp = self._perform(
                    method,
                    path,
                    headers=self._headers(json is not None),
                    json=json,
                    **kwargs,
                )
            except httpx.TimeoutException as e:
                last_error = ProviderTimeoutError(
                    f"{method} {path} timed out: {self._timeout}s"
                )
                if attempt >= self._retry_attempts:
                    raise last_error
                self._sleep(attempt)
                continue
            except httpx.HTTPError as e:
                last_error = ProviderRemoteError(f"{method} {path} failed: {e}")
                if attempt >= self._retry_attempts:
                    raise last_error
                self._sleep(attempt)
                continue

            if resp.status_code in RETRYABLE_STATUS and attempt < self._retry_attempts:
                should_retry = True
                if self._should_retry is not None:
                    should_retry = self._should_retry(
                        resp.status_code, self._safe_json(resp)
                    )
                if should_retry:
                    self._sleep(attempt)
                    continue

            if resp.status_code >= 400:
                # Non-retryable client error (401/403/404/422) -> fail fast.
                if self._error_parser is not None:
                    parsed = self._error_parser(
                        resp.status_code,
                        self._safe_json(resp),
                        method,
                        path,
                    )
                    if isinstance(parsed, ProviderError):
                        raise parsed
                raise ProviderRemoteError(
                    f"{method} {path} returned HTTP {resp.status_code}"
                )

            return resp

        raise ProviderRemoteError(f"{method} {path} failed after retries: {last_error}")

    def _sleep(self, attempt: int) -> None:
        base = self._retry_backoff * (2 ** (attempt - 1))
        delay = min(base, self._retry_max_delay)
        # Add jitter to avoid synchronized retry storms.
        jitter = random.uniform(0, 0.25 * delay)
        time.sleep(delay + jitter)

    def post_json(self, path: str, payload: dict) -> dict:
        """POST a JSON body and return parsed JSON."""
        resp = self.request("POST", path, json=payload)
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderRemoteError(f"Could not parse JSON response from {path}: {e}")

    def get_json(self, path: str) -> dict:
        resp = self.request("GET", path)
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderRemoteError(f"Could not parse JSON response from {path}: {e}")

    def get_content(self, path: str) -> bytes:
        """GET a URL/path and return raw bytes (used for downloading results)."""
        resp = self.request("GET", path)
        if resp.status_code != 200 or not resp.content:
            raise ProviderRemoteError(f"Could not download content from {path}")
        return resp.content

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Optional[dict]:
        """Best-effort parse of a response body as JSON (never raises)."""
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
        return None

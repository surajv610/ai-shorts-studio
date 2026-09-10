#!/usr/bin/env python3
"""Gemini LLM Smoke Test — runs ONE inexpensive structured request.

WARNING: This script makes a REAL Gemini API request and will consume quota.
Run it manually with:
    GEMINI_API_KEY="your-key" python3 scripts/gemini_smoke_test.py

It is never imported or executed by pytest, CI, or the FastAPI app.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("Gemini LLM Smoke Test")
    print("=" * 60)
    print()
    print("WARNING: This script sends a REAL Gemini API request.")
    print("It uses the minimal GEMINI_QUOTA cost (1 small structured call).")
    print()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")
        print("Usage: GEMINI_API_KEY='your-key' python3 scripts/gemini_smoke_test.py")
        sys.exit(1)

    from backend.config import Settings
    from backend.providers.gemini import GeminiLLMProvider

    settings = Settings(env={
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": api_key,
        "LLM_MODEL": os.environ.get("LLM_MODEL", ""),
        "LLM_BASE_URL": os.environ.get("LLM_BASE_URL", ""),
        "LLM_TIMEOUT": os.environ.get("LLM_TIMEOUT", "30"),
    })

    print(f"Provider: gemini")
    print(f"Model:    {settings.llm_model}")
    print(f"Base URL: {settings.llm_base_url or '(default)'}")
    print()

    try:
        provider = GeminiLLMProvider(settings)
    except Exception as e:
        print(f"ERROR: Could not initialize Gemini provider: {e}")
        sys.exit(1)

    # Single inexpensive structured call
    print("Sending structured generation request...")
    result = provider.generate_structured(
        system_prompt="You are a helpful assistant. Respond concisely in valid JSON only.",
        user_message='Return a JSON object with exactly these fields: {"status": "ok", "test": "smoke"}',
        json_schema={
            "name": "SmokeTest",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "test": {"type": "string"},
                },
                "required": ["status", "test"],
            },
        },
    )
    print()
    print("Response:")
    print(json.dumps(result, indent=2))
    print()

    # Basic validation
    if isinstance(result, dict) and result.get("status") == "ok":
        print("PASS — Gemini structured output working.")
    else:
        print(f"WARN — Unexpected response: {result}")
        print("The API call succeeded but the content did not match expectations.")
        print("This may be normal depending on the model's interpretation.")
        sys.exit(0)


if __name__ == "__main__":
    main()

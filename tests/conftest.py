"""Test configuration shared across the suite.

Sets the environment variables that `config.settings` requires BEFORE any
`src/` module is imported. Pytest loads `conftest.py` during collection,
which happens before test modules are imported, so this guarantees the
Pydantic Settings singleton sees a valid configuration.

Real credentials are NEVER used in tests. The values below are obviously
fake — if any test ever tries to reach an external service with them, it
will fail loudly (which is what we want).
"""

from __future__ import annotations

import os
from pathlib import Path

# IMPORTANT: these `setdefault` calls run at module import time, before any
# `from src.* import ...` in test modules. Do NOT move them below imports.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # quieter test output

os.environ.setdefault("IMAP_HOST", "imap.test.local")
os.environ.setdefault("IMAP_USERNAME", "test@example.com")
os.environ.setdefault("IMAP_PASSWORD", "test-app-password")

os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
# Fallback provider key — needed by LLMClient fallback-path tests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-openai-key")

os.environ.setdefault("TWILIO_ACCOUNT_SID", "AC" + "x" * 32)
os.environ.setdefault("TWILIO_AUTH_TOKEN", "x" * 32)
os.environ.setdefault("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
os.environ.setdefault("WHATSAPP_TO", "whatsapp:+5511999999999")

import pytest  # noqa: E402  — imports must come after env setup above

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to `tests/fixtures/`."""
    return FIXTURES_DIR

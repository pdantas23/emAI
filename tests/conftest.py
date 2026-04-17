"""Test configuration shared across the suite.

Sets the environment variables that `config.settings` requires BEFORE any
`src/` module is imported. After the admin-provisioned refactor, Settings
only needs DATABASE_URL and ENCRYPTION_KEY at the bootstrap level. API keys
are injected via constructor (dependency injection) in tests.

Real credentials are NEVER used in tests.
"""

from __future__ import annotations

import os
from pathlib import Path

# IMPORTANT: these `setdefault` calls run at module import time, before any
# `from src.* import ...` in test modules.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# Bootstrap settings only need DATABASE_URL (tests use SQLite in-memory)
# and ENCRYPTION_KEY (for crypto tests).
os.environ.setdefault("DATABASE_URL", "sqlite:///")
os.environ.setdefault("ENCRYPTION_KEY", "a" * 64)  # 32 bytes of 0xAA

# Admin password for UI tests
os.environ.setdefault("ADMIN_PASSWORD", "test-admin")

import pytest  # noqa: E402

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to `tests/fixtures/`."""
    return FIXTURES_DIR

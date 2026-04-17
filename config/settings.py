"""Bootstrap configuration — the minimal env needed before the DB is reachable.

After the admin-provisioned refactor, API keys and per-user settings live in
the database (`user_credentials` + `user_settings`). This module retains
ONLY the two values needed to reach the database and decrypt what's inside:

    DATABASE_URL      — Supabase Postgres connection string
    ENCRYPTION_KEY    — 64-hex-char AES-256 key (see src/storage/crypto.py)

Plus non-secret operational knobs (APP_ENV, LOG_LEVEL, TIMEZONE) that apply
globally to the process, not per user.

Everything else (Anthropic keys, Twilio creds, IMAP passwords, WhatsApp
routing) is loaded from the DB at runtime via `config.runtime_settings`.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `config/` directory this file lives in.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class AppEnv(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# --------------------------------------------------------------------------- #
# Bootstrap settings — the bare minimum to reach the DB.
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Bootstrap config. Import the `settings` singleton at the bottom."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Operational ----
    app_env: AppEnv = Field(AppEnv.development)
    log_level: LogLevel = Field(LogLevel.INFO)
    timezone: str = Field("America/Sao_Paulo")

    # ---- Database (Supabase Postgres or SQLite for dev/tests) ----
    database_url: str = Field("sqlite:///./logs/emai_state.sqlite")

    # ---- Encryption key for credentials at rest ----
    encryption_key: str = Field(
        "",
        description="64-char hex string (AES-256). Only env var besides DB URL.",
    )

    # ---- Admin password for the Streamlit UI ----
    admin_password: str = Field("admin", description="Simple password for Admin tab")

    # ---- Convenience ----

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.production


# --------------------------------------------------------------------------- #
# Singleton — cached so we parse env exactly once per process.
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings: Settings = get_settings()

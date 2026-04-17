"""Centralized application configuration.

All environment variables are loaded and validated here through Pydantic Settings.
The rest of the codebase MUST import the typed `settings` singleton from this module
instead of reading `os.environ` directly. This keeps secrets in one place, fails fast
on misconfiguration, and gives editors/mypy full autocomplete on every config value.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `config/` directory this file lives in.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Enums (typed choices instead of stringly-typed config)
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


class LLMProvider(str, Enum):
    anthropic = "anthropic"
    openai = "openai"


# --------------------------------------------------------------------------- #
# Section settings — grouped by concern, composed into the root Settings below.
# --------------------------------------------------------------------------- #


class IMAPSettings(BaseSettings):
    """IMAP credentials and fetch behavior."""

    model_config = SettingsConfigDict(env_prefix="IMAP_", extra="ignore")

    host: str = Field(..., description="IMAP server hostname, e.g. imap.gmail.com")
    port: int = Field(993, ge=1, le=65535)
    username: str = Field(..., description="Email account username")
    password: SecretStr = Field(..., description="App password (NOT account password)")
    folder: str = Field("INBOX")
    use_ssl: bool = Field(True)
    fetch_limit: int = Field(50, ge=1, le=500, description="Max unread emails per run")


class LLMSettings(BaseSettings):
    """LLM provider, model, and runtime parameters."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: LLMProvider = Field(LLMProvider.anthropic)
    model: str = Field("claude-sonnet-4-6")
    classifier_model: str = Field(
        "claude-haiku-4-5-20251001",
        description="Cheap model for the pre-summary relevance gate",
    )
    max_tokens: int = Field(2048, ge=128, le=8192)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    timeout_seconds: int = Field(60, ge=5, le=300)


class WhatsAppSettings(BaseSettings):
    """Twilio WhatsApp credentials and routing."""

    model_config = SettingsConfigDict(env_prefix="TWILIO_", extra="ignore")

    account_sid: SecretStr = Field(...)
    auth_token: SecretStr = Field(...)
    whatsapp_from: str = Field(
        ..., description="Twilio sender, e.g. whatsapp:+14155238886"
    )

    @field_validator("whatsapp_from")
    @classmethod
    def _must_be_whatsapp_uri(cls, v: str) -> str:
        if not v.startswith("whatsapp:+"):
            raise ValueError("Must start with 'whatsapp:+' (e.g. whatsapp:+5511999999999)")
        return v


class FilterSettings(BaseSettings):
    """Pre-LLM filter thresholds."""

    model_config = SettingsConfigDict(extra="ignore")

    spam_score_threshold: float = Field(5.0, ge=0.0)
    attachment_max_mb: int = Field(10, ge=1, le=100)
    enable_ocr: bool = Field(False)


# --------------------------------------------------------------------------- #
# Root settings — single source of truth.
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Root configuration object. Import the `settings` singleton at the bottom."""
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__" 
    )

    # ---- App ----
    app_env: AppEnv = Field(AppEnv.development)
    log_level: LogLevel = Field(LogLevel.INFO)
    timezone: str = Field("America/Sao_Paulo")

    # ---- API keys (kept at root because each provider has a distinct prefix) ----
    anthropic_api_key: SecretStr | None = Field(None)
    openai_api_key: SecretStr | None = Field(None)

    # ---- WhatsApp routing (recipient lives at the root, sender in WhatsAppSettings) ----
    whatsapp_to: str = Field(..., description="Recipient, e.g. whatsapp:+5511999999999")

    # ---- Storage ----
    database_url: str = Field("sqlite:///./logs/emai_state.sqlite")

    # ---- Scheduling ----
    run_interval_minutes: int = Field(30, ge=1, le=1440)

    # ---- Composed sections ----
    imap: IMAPSettings = Field(default_factory=IMAPSettings)  # type: ignore[arg-type]
    llm: LLMSettings = Field(default_factory=LLMSettings)
    whatsapp: WhatsAppSettings = Field(default_factory=WhatsAppSettings)  # type: ignore[arg-type]
    filters: FilterSettings = Field(default_factory=FilterSettings)

    # ---- Validators ----

    @field_validator("whatsapp_to")
    @classmethod
    def _validate_recipient(cls, v: str) -> str:
        if not v.startswith("whatsapp:+"):
            raise ValueError("WHATSAPP_TO must start with 'whatsapp:+' (E.164 format)")
        return v

    @model_validator(mode="after")
    def _ensure_active_provider_has_key(self) -> Settings:
        """Selected LLM provider must have a corresponding API key set."""
        if self.llm.provider is LLMProvider.anthropic and not self.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing."
            )
        if self.llm.provider is LLMProvider.openai and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is missing.")
        return self

    # ---- Convenience ----

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.production


# --------------------------------------------------------------------------- #
# Singleton accessor — cached so we parse env exactly once per process.
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance. Use this in non-trivial code paths
    where you want to defer instantiation (e.g. tests that override env vars)."""
    return Settings()  # type: ignore[call-arg]


# Eager singleton for ergonomic imports: `from config.settings import settings`.
# This is what most of the codebase will use.
settings: Settings = get_settings()

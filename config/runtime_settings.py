"""Per-user runtime configuration loaded from the database.

This replaces the old `settings` singleton for everything except bootstrap
config (DATABASE_URL, ENCRYPTION_KEY). Instead of reading API keys from
`.env`, the orchestrator receives a `UserRuntimeConfig` built from the
`user_credentials` + `user_settings` tables at the start of each run.

The lifecycle is:
    1. Admin provisions credentials via Streamlit → encrypted in DB.
    2. User configures email/WhatsApp/interval via Streamlit → DB.
    3. Engine startup: load_user_config(user_id) → UserRuntimeConfig.
    4. build_orchestrator(config) wires every collaborator from this object.
    5. After the run, the config object (and any decrypted secrets in it)
       is garbage-collected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserRuntimeConfig:
    """Everything the orchestrator and its collaborators need for one user.

    Frozen so secrets can't be accidentally mutated during a run.
    The decrypted API keys live in this object's memory only — they are
    never logged, never serialized, and garbage-collected when the run ends.
    """

    # ---- Identity ------------------------------------------------------------
    user_id: str

    # ---- LLM -----------------------------------------------------------------
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # ---- Evolution API (WhatsApp gateway) ------------------------------------
    evolution_url: str | None = None
    evolution_api_key: str | None = None
    evolution_instance: str | None = None

    # ---- Supabase (per-user project, if different from admin DB) -------------
    supabase_url: str | None = None
    supabase_key: str | None = None

    # ---- IMAP ----------------------------------------------------------------
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_username: str = ""           # = user_settings.email
    imap_password: str = ""           # = decrypted gmail_app_password

    # ---- WhatsApp routing ----------------------------------------------------
    whatsapp_to: str = ""

    # ---- Scheduling ----------------------------------------------------------
    run_interval_minutes: int = 30

    # ---- Database (from bootstrap settings) ----------------------------------
    database_url: str = ""

    # ---- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of missing-credential error strings (empty = all good).

        This is the runtime equivalent of the old `preflight_check()`. The
        orchestrator refuses to run if this returns a non-empty list.
        """
        errors: list[str] = []

        if not self.anthropic_api_key and not self.openai_api_key:
            errors.append("No LLM API key configured (need anthropic_key or openai_key)")

        if not self.evolution_url:
            errors.append("evolution_url is missing")
        if not self.evolution_api_key:
            errors.append("evolution_api_key is missing")
        if not self.evolution_instance:
            errors.append("evolution_instance is missing")

        if not self.imap_username:
            errors.append("email (IMAP username) is missing")
        if not self.imap_password:
            errors.append("gmail_app_password is missing")

        if not self.whatsapp_to:
            errors.append("whatsapp_to (recipient) is missing")

        if not self.database_url:
            errors.append("database_url is missing")

        return errors

    @property
    def is_ready(self) -> bool:
        """True when all mandatory credentials are present."""
        return len(self.validate()) == 0

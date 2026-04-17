"""CRUD for the `user_credentials` table (Admin-provisioned API keys).

Every write encrypts sensitive fields via `src.storage.crypto` before they
hit Postgres. Every read decrypts on the fly — the plaintext lives in RAM
only for the duration of the caller's scope.

This module is used by:
  - The Streamlit Admin tab (write path)
  - `load_user_config()` in main.py (read path, once per pipeline run)
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, select

from src.storage.audit import AuditStore
from src.storage.crypto import decrypt, encrypt
from src.utils.logger import log


# --------------------------------------------------------------------------- #
# SQLModel table
# --------------------------------------------------------------------------- #


class UserCredential(SQLModel, table=True):
    """One row per user — infrastructure keys provisioned by Admin."""

    __tablename__ = "user_credentials"

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(unique=True, index=True)

    # Encrypted BYTEA columns — stored as raw bytes.
    anthropic_key: bytes | None = Field(default=None)
    openai_key: bytes | None = Field(default=None)

    # Evolution API (WhatsApp gateway)
    evolution_url: str | None = Field(default=None)
    evolution_api_key: bytes | None = Field(default=None)
    evolution_instance: str | None = Field(default=None, max_length=100)

    # Supabase project credentials
    supabase_url: str | None = Field(default=None)
    supabase_key: bytes | None = Field(default=None)

    # Gmail / IMAP
    gmail_app_password: bytes | None = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_by: str | None = Field(default=None)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class CredentialStore:
    """Read/write encrypted credentials for a given user_id."""

    def __init__(self, engine: "Engine") -> None:  # noqa: F821
        self._engine = engine
        self._audit = AuditStore(engine)

    def upsert(
        self,
        *,
        user_id: str,
        admin_name: str = "admin",
        anthropic_key: str | None = None,
        openai_key: str | None = None,
        evolution_url: str | None = None,
        evolution_api_key: str | None = None,
        evolution_instance: str | None = None,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        gmail_app_password: str | None = None,
    ) -> UserCredential:
        """Insert or update credentials for a user.

        Only non-None values are updated — passing `anthropic_key=None` leaves
        the existing value untouched (it does NOT erase it).
        """
        # Track which fields are being set (names only — never values).
        changed_fields: list[str] = []

        with Session(self._engine) as session:
            row = session.exec(
                select(UserCredential).where(UserCredential.user_id == user_id)
            ).first()

            is_new = row is None
            if is_new:
                row = UserCredential(user_id=user_id)
                session.add(row)

            # Encrypt and set only the fields that were explicitly provided.
            if anthropic_key is not None:
                row.anthropic_key = encrypt(anthropic_key)
                changed_fields.append("anthropic_key")
            if openai_key is not None:
                row.openai_key = encrypt(openai_key)
                changed_fields.append("openai_key")
            if evolution_url is not None:
                row.evolution_url = evolution_url
                changed_fields.append("evolution_url")
            if evolution_api_key is not None:
                row.evolution_api_key = encrypt(evolution_api_key)
                changed_fields.append("evolution_api_key")
            if evolution_instance is not None:
                row.evolution_instance = evolution_instance
                changed_fields.append("evolution_instance")
            if supabase_url is not None:
                row.supabase_url = supabase_url
                changed_fields.append("supabase_url")
            if supabase_key is not None:
                row.supabase_key = encrypt(supabase_key)
                changed_fields.append("supabase_key")
            if gmail_app_password is not None:
                row.gmail_app_password = encrypt(gmail_app_password)
                changed_fields.append("gmail_app_password")

            row.updated_by = admin_name
            row.updated_at = datetime.now(UTC)

            session.commit()
            session.refresh(row)
            log.info("Upserted credentials for user_id={} by {}", user_id, admin_name)

        # Audit: record what happened (field names only, never values).
        self._audit.record(
            actor=admin_name,
            action="credentials.create" if is_new else "credentials.update",
            target_user_id=user_id,
            metadata={"fields_changed": changed_fields},
        )

        return row

    def get_decrypted(self, user_id: str) -> dict[str, str | None]:
        """Return all credentials for a user, decrypted.

        Returns a plain dict so the caller can build a `UserRuntimeConfig`
        without importing SQLModel types. Missing/NULL fields come back as None.
        """
        with Session(self._engine) as session:
            row = session.exec(
                select(UserCredential).where(UserCredential.user_id == user_id)
            ).first()

        if row is None:
            return {}

        def _d(blob: bytes | None) -> str | None:
            return decrypt(blob) if blob else None

        return {
            "user_id": row.user_id,
            "anthropic_key": _d(row.anthropic_key),
            "openai_key": _d(row.openai_key),
            "evolution_url": row.evolution_url,
            "evolution_api_key": _d(row.evolution_api_key),
            "evolution_instance": row.evolution_instance,
            "supabase_url": row.supabase_url,
            "supabase_key": _d(row.supabase_key),
            "gmail_app_password": _d(row.gmail_app_password),
        }

    def list_users(self) -> list[str]:
        """Return all provisioned user_ids."""
        with Session(self._engine) as session:
            rows = session.exec(select(UserCredential.user_id)).all()
            return list(rows)

    def has_user(self, user_id: str) -> bool:
        with Session(self._engine) as session:
            row = session.exec(
                select(UserCredential.id).where(UserCredential.user_id == user_id)
            ).first()
            return row is not None

    def delete(self, user_id: str, *, admin_name: str = "admin") -> bool:
        """Delete a user's credentials. Returns True if a row was removed."""
        with Session(self._engine) as session:
            row = session.exec(
                select(UserCredential).where(UserCredential.user_id == user_id)
            ).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            log.info("Deleted credentials for user_id={}", user_id)

        self._audit.record(
            actor=admin_name,
            action="credentials.delete",
            target_user_id=user_id,
        )
        return True

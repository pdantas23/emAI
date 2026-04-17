"""CRUD for the `user_settings` table (user-facing configuration).

This table holds the non-sensitive, user-editable settings: which email
account to monitor, where to send WhatsApp reports, and how often to run.

Used by:
  - The Streamlit User tab (write path)
  - `load_user_config()` in main.py (read path)
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, Session, SQLModel, select

from src.utils.logger import log


# --------------------------------------------------------------------------- #
# SQLModel table
# --------------------------------------------------------------------------- #


class UserSetting(SQLModel, table=True):
    """Per-user runtime configuration editable by the user themselves."""

    __tablename__ = "user_settings"

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(unique=True, index=True)

    email: str = Field(max_length=320)
    imap_host: str = Field(default="imap.gmail.com", max_length=255)
    imap_port: int = Field(default=993)
    whatsapp_to: str = Field(max_length=30)
    run_interval_minutes: int = Field(default=30)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class UserSettingsStore:
    """Read/write user-facing settings."""

    def __init__(self, engine: "Engine") -> None:  # noqa: F821
        self._engine = engine

    def upsert(
        self,
        *,
        user_id: str,
        email: str,
        whatsapp_to: str,
        imap_host: str = "imap.gmail.com",
        imap_port: int = 993,
        run_interval_minutes: int = 30,
    ) -> UserSetting:
        """Insert or update settings for a user."""
        with Session(self._engine) as session:
            row = session.exec(
                select(UserSetting).where(UserSetting.user_id == user_id)
            ).first()

            if row is None:
                row = UserSetting(user_id=user_id, email=email, whatsapp_to=whatsapp_to)
                session.add(row)
            else:
                row.email = email
                row.whatsapp_to = whatsapp_to

            row.imap_host = imap_host
            row.imap_port = imap_port
            row.run_interval_minutes = run_interval_minutes
            row.updated_at = datetime.now(UTC)

            session.commit()
            session.refresh(row)
            log.info("Upserted settings for user_id={}", user_id)
            return row

    def get(self, user_id: str) -> dict[str, str | int | None]:
        """Return settings for a user as a plain dict. Empty dict if not found."""
        with Session(self._engine) as session:
            row = session.exec(
                select(UserSetting).where(UserSetting.user_id == user_id)
            ).first()

        if row is None:
            return {}

        return {
            "user_id": row.user_id,
            "email": row.email,
            "imap_host": row.imap_host,
            "imap_port": row.imap_port,
            "whatsapp_to": row.whatsapp_to,
            "run_interval_minutes": row.run_interval_minutes,
        }

    def delete(self, user_id: str) -> bool:
        with Session(self._engine) as session:
            row = session.exec(
                select(UserSetting).where(UserSetting.user_id == user_id)
            ).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

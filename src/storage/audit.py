"""Append-only audit log for credential and settings changes.

Every mutation through `CredentialStore` or `UserSettingsStore` records
*what* changed (field names only — never values) and *who* did it.

This module is intentionally simple: one table, one writer, no updates,
no deletes. The audit trail is immutable by design — even the app role
only has INSERT + SELECT grants.
"""

from __future__ import annotations

from datetime import UTC, datetime

import json as _json

from sqlmodel import Field, Session, SQLModel, select

from src.utils.logger import log


# --------------------------------------------------------------------------- #
# SQLModel table
# --------------------------------------------------------------------------- #


class AuditLog(SQLModel, table=True):
    """One row per admin action — append-only."""

    __tablename__ = "system_audit_logs"

    id: int | None = Field(default=None, primary_key=True)
    actor: str
    action: str
    target_user_id: str
    # Stored as TEXT (JSON-serialized) for SQLite compatibility.
    # Postgres migration uses JSONB; the app layer handles serialization.
    metadata_json: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class AuditStore:
    """Append-only writer for the system_audit_logs table."""

    def __init__(self, engine: "Engine") -> None:  # noqa: F821
        self._engine = engine

    def record(
        self,
        *,
        actor: str,
        action: str,
        target_user_id: str,
        metadata: dict | None = None,
    ) -> None:
        """Insert a single audit entry. Never raises — audit failures are
        logged but must not break the primary operation."""
        try:
            with Session(self._engine) as session:
                entry = AuditLog(
                    actor=actor,
                    action=action,
                    target_user_id=target_user_id,
                    metadata_json=_json.dumps(metadata) if metadata else None,
                    created_at=datetime.now(UTC),
                )
                session.add(entry)
                session.commit()
            log.debug(
                "[AUDIT] {} by {} on user={} meta={}",
                action, actor, target_user_id, metadata,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[AUDIT] failed to write audit log: {} — action={} actor={} target={}",
                exc, action, actor, target_user_id,
            )

    def list_recent(self, *, limit: int = 50) -> list[AuditLog]:
        """Return the most recent audit entries, newest first."""
        with Session(self._engine) as session:
            return list(
                session.exec(
                    select(AuditLog)
                    .order_by(AuditLog.created_at.desc())  # type: ignore[attr-defined]
                    .limit(limit)
                )
            )

    def list_for_user(self, target_user_id: str, *, limit: int = 50) -> list[AuditLog]:
        """Return audit entries for a specific target user."""
        with Session(self._engine) as session:
            return list(
                session.exec(
                    select(AuditLog)
                    .where(AuditLog.target_user_id == target_user_id)
                    .order_by(AuditLog.created_at.desc())  # type: ignore[attr-defined]
                    .limit(limit)
                )
            )

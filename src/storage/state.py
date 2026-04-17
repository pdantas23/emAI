"""State persistence for the email pipeline (Supabase Postgres / SQLite).

This module is the orchestrator's MEMORY across runs. Every email that
reaches a terminal state — delivered to WhatsApp, or skipped as irrelevant
— gets one row in the `processed_emails` table. Before processing any
email, the orchestrator asks `has_been_processed(message_id)` and skips
ones we've already handled.

The dedup contract is what makes the whole system idempotent:

    1. fetch_unread() returns emails NOT marked \\Seen on IMAP.
    2. has_been_processed() filters out anything we already finished.
    3. mark_as_processed() is called ONLY on terminal success/skip.
    4. mark_as_seen() is called AFTER mark_as_processed() commits.

Crash anywhere in steps 2-4 leaves the email available for retry on the
next run, and dedup prevents double-summarization.

**Failure policy: FAIL-STOP.** If the database is unreachable at startup,
`StateStore()` raises `StorageError` and the pipeline halts. Same for any
write failure during operation — we propagate the exception rather than
swallowing it. Rationale: a silent storage failure would let the pipeline
process the same email twice on the next run, costing both Sonnet tokens
AND user trust in the WhatsApp report. Better to refuse to run.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from config.settings import settings
from src.storage.models import DeliveryStatus, ProcessedEmail
from src.utils.logger import log


# --------------------------------------------------------------------------- #
# Public exception
# --------------------------------------------------------------------------- #


class StorageError(Exception):
    """Raised on any database failure — connection, query, or write.

    The pipeline treats this as fatal: the orchestrator should NOT continue
    processing emails when the state store is unavailable, otherwise the
    next run would re-process emails we already delivered.
    """


# --------------------------------------------------------------------------- #
# Engine factory — kept module-level so tests can swap with an in-memory DB
# --------------------------------------------------------------------------- #


def _build_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine with sensible per-dialect defaults.

    SQLite needs `check_same_thread=False` to play nicely with any future
    threaded use (apscheduler runs jobs on a worker thread). Postgres has
    no such quirks; the default settings are right for Supabase.
    """
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args, echo=False)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class StateStore:
    """Single entry point for everything that touches `processed_emails`.

    Construct ONCE at orchestrator startup. The constructor performs a
    `SELECT 1` to verify the database is reachable — if it isn't, we raise
    `StorageError` immediately (fail-stop) so the pipeline doesn't proceed
    in a half-broken state.

    The `engine` parameter is the testing / dependency-injection seam:
    pass an in-memory SQLite engine in tests; in production let it default
    to `settings.database_url`.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine: Engine = engine or _build_engine(settings.database_url)
        self._verify_connection()
        self._ensure_tables()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def has_been_processed(self, message_id: str) -> bool:
        """True iff a terminal-state row exists for this Message-ID.

        Cheap by design: an indexed equality lookup on `message_id`. The
        orchestrator calls this ONCE per fetched email — keep it fast.
        """
        try:
            with Session(self._engine) as session:
                row = session.exec(
                    select(ProcessedEmail.id).where(
                        ProcessedEmail.message_id == message_id
                    )
                ).first()
        except Exception as exc:
            raise StorageError(
                f"Failed to query processed_emails for message_id={message_id}: {exc}"
            ) from exc
        return row is not None

    def mark_as_processed(
        self,
        *,
        message_id: str,
        uid: str,
        relevance: bool,
        priority: str,
        delivery_status: DeliveryStatus = DeliveryStatus.delivered,
        twilio_sids: list[str] | None = None,
    ) -> ProcessedEmail:
        """Record a terminal-state row and commit.

        The signature is deliberately restricted to non-sensitive metadata:
        Message-ID / UID for dedup, the classifier booleans (relevance,
        priority), delivery status, and the Twilio SIDs. Callers MUST NOT
        pass email bodies, subjects, sender identities, classifier reasons,
        or LLM summaries — and the store exposes no column to hold them.
        This is the enforcement half of emAI's privacy contract; the
        orchestrator is expected to drop those fields on the floor before
        getting here.

        Call this ONLY when the email has reached a terminal outcome:
          - DELIVERED: summary built and successfully sent to WhatsApp.
          - SKIPPED_IRRELEVANT: classifier said `relevance=False`.

        Do NOT call this on transient delivery failures — leaving no row
        means the next pipeline run picks the email up again for retry.

        Raises:
            StorageError: any DB error (unique violation, connection loss,
                schema drift, ...). Pipeline is expected to halt.
        """
        record = ProcessedEmail(
            message_id=message_id,
            uid=uid,
            relevance=relevance,
            priority=priority,
            delivery_status=delivery_status.value,
            twilio_sids=",".join(twilio_sids) if twilio_sids else None,
        )

        try:
            with Session(self._engine) as session:
                session.add(record)
                session.commit()
                session.refresh(record)
        except Exception as exc:
            raise StorageError(
                f"Failed to persist processed_emails row for "
                f"message_id={message_id}: {exc}"
            ) from exc

        log.info(
            "Persisted processed_email id={} message_id={} status={} priority={} "
            "relevance={} sids={}",
            record.id,
            record.message_id,
            record.delivery_status,
            record.priority,
            record.relevance,
            record.twilio_sids or "-",
        )
        return record

    def get(self, message_id: str) -> ProcessedEmail | None:
        """Fetch the full row for a Message-ID (or None). Useful for audits."""
        try:
            with Session(self._engine) as session:
                return session.exec(
                    select(ProcessedEmail).where(
                        ProcessedEmail.message_id == message_id
                    )
                ).first()
        except Exception as exc:
            raise StorageError(
                f"Failed to fetch processed_emails row for "
                f"message_id={message_id}: {exc}"
            ) from exc

    def list_recent(self, limit: int = 50) -> list[ProcessedEmail]:
        """Return the most recently processed rows, newest first."""
        try:
            with Session(self._engine) as session:
                return list(
                    session.exec(
                        select(ProcessedEmail)
                        .order_by(ProcessedEmail.processed_at.desc())  # type: ignore[attr-defined]
                        .limit(limit)
                    )
                )
        except Exception as exc:
            raise StorageError(f"Failed to list recent processed_emails: {exc}") from exc

    def close(self) -> None:
        """Dispose the connection pool. Safe to call multiple times."""
        self._engine.dispose()

    # ------------------------------------------------------------------ #
    # Internal — startup checks
    # ------------------------------------------------------------------ #

    def _verify_connection(self) -> None:
        """Run a trivial `SELECT 1` to prove the DB is reachable.

        This is the FAIL-STOP gate. If it raises, the pipeline never gets
        a chance to write or read state — much better than discovering the
        DB is down only after we've shipped a WhatsApp message and need to
        record it.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise StorageError(
                f"Database unreachable at startup ({self._engine.url}): {exc}"
            ) from exc
        log.debug("State store connected to {}", self._engine.url)

    def _ensure_tables(self) -> None:
        """Create tables that don't exist yet. Idempotent.

        For Supabase production you'd typically manage schema via Alembic
        migrations and skip this; for MVP / dev / tests, auto-create keeps
        the friction at zero. `create_all` is a no-op when tables exist.
        """
        try:
            SQLModel.metadata.create_all(self._engine)
        except Exception as exc:
            raise StorageError(
                f"Failed to create/verify schema on {self._engine.url}: {exc}"
            ) from exc

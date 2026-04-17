"""SQLModel table definitions for emAI's persistence layer.

Currently a single table — `processed_emails` — that captures everything
we know about a triaged email, including the classifier verdict, the
summary the LLM produced (when relevant), and the delivery outcome.

Why a flat table instead of a normalized schema?
  - One row per email is the smallest unit of work in the orchestrator.
  - Querying "did we already process this Message-ID?" is the hottest
    operation; one indexed lookup beats a join every time.
  - The audit story is "show me what happened to email X", which a flat
    row answers without joins.

For Supabase production the connection URL is provided via DATABASE_URL.
For dev/tests it falls back to SQLite (`logs/emai_state.sqlite`). The
DDL is dialect-agnostic — every column type below is supported by both
SQLite and Postgres without changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from sqlmodel import Field, SQLModel


# --------------------------------------------------------------------------- #
# Enums (string-valued so they survive a database round-trip cleanly)
# --------------------------------------------------------------------------- #


class DeliveryStatus(str, Enum):
    """Terminal state of an email's journey through the pipeline.

    We deliberately do NOT have a `pending` value: a row is only inserted
    once the email reaches a terminal state. Failed deliveries are LOGGED
    but not persisted — that's how we get free retries on the next run
    (the orchestrator's `has_been_processed` check returns False, so the
    email is re-fetched and re-attempted).
    """

    delivered = "delivered"             # Summary built and successfully sent to WhatsApp.
    skipped_irrelevant = "skipped_irrelevant"  # Classifier said relevance=False; no summary, no send.


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #


class ProcessedEmail(SQLModel, table=True):
    """One row per email the pipeline has finished with.

    `message_id` is the dedup key. The IMAP UID can change across servers
    (or after a folder move) but RFC-822 Message-ID is stable, so we key
    everything off that.

    **Privacy contract.** This table stores ONLY metadata — no email body,
    no subject, no sender identity, no classifier justification, no LLM
    summary. Sensitive content is processed in memory (classified,
    summarized, rendered for WhatsApp) and then dropped on the floor
    before we touch the database. The fields below are the full list of
    what we're willing to retain; anything else must stay transient.
    """

    __tablename__ = "processed_emails"

    # ---- Primary key & dedup index ----
    id: int | None = Field(default=None, primary_key=True)

    # ---- Multi-user partitioning ----
    user_id: str | None = Field(default=None, index=True, description="Owner user_id")

    message_id: str = Field(
        index=True,
        unique=True,
        max_length=998,  # RFC-2822 line length cap; real values are far shorter.
        description="RFC-822 Message-ID — stable dedup key across IMAP servers",
    )

    uid: str = Field(
        index=True,
        max_length=64,
        description="IMAP server UID at processing time (may change across folder moves)",
    )

    # ---- Classifier verdict (non-sensitive booleans/enums only) ----
    relevance: bool
    priority: str = Field(max_length=10, description="low | medium | high")

    # ---- Delivery audit ----
    delivery_status: str = Field(
        max_length=24,
        description="One of DeliveryStatus values (stored as text for portability)",
    )

    # Comma-separated message IDs from the messaging gateway (Evolution API).
    # IDs never contain commas, so CSV is unambiguous and human-readable.
    message_ids: str | None = Field(
        default=None,
        max_length=2048,
        description="Comma-separated Messaging gateway IDs from successful sends",
    )

    # ---- Bookkeeping ----
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
        description="UTC timestamp when this row was written",
    )

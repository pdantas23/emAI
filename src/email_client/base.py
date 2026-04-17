"""Provider-agnostic interface and data models for email ingestion.

Concrete implementations live in `gmail_imap.py`, `outlook_imap.py`, etc.
The orchestrator only ever depends on `EmailClient` and `RawEmail` defined here —
swapping providers is therefore a one-line change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class Attachment(BaseModel):
    """An email attachment. `content` is None when the payload was skipped
    because it exceeded the configured size limit (metadata is still kept)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str
    content_type: str
    size_bytes: int
    content: bytes | None = Field(
        None, description="None if skipped due to attachment_max_mb threshold"
    )

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def was_skipped(self) -> bool:
        return self.content is None and self.size_bytes > 0


class RawEmail(BaseModel):
    """Provider-agnostic email returned by `EmailClient.fetch_unread()`.

    `body_text` and `body_html` are the RAW MIME parts straight from the server.
    Cleaning HTML and choosing the best body for LLM input is the parser's job.
    """

    # ---- Identity (used for idempotency / dedup) ----
    uid: str = Field(..., description="IMAP server UID — needed to mark as seen later")
    message_id: str = Field(..., description="RFC-822 Message-ID — stable across servers")
    folder: str

    # ---- Participants ----
    sender_name: str | None = None
    sender_email: str
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)

    # ---- Content ----
    subject: str = ""
    date: datetime
    body_text: str = ""
    body_html: str = ""

    # ---- Metadata ----
    headers: dict[str, str] = Field(default_factory=dict)
    attachments: list[Attachment] = Field(default_factory=list)
    spam_flag: bool = Field(
        False, description="True if the IMAP server already flagged this as spam"
    )

    # ---- Convenience ----

    @property
    def has_html(self) -> bool:
        return bool(self.body_html)

    @property
    def has_attachments(self) -> bool:
        return len(self.attachments) > 0


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class EmailClient(ABC):
    """Abstract email reader. Implementations MUST be usable as context managers
    so that connections are guaranteed to be released even on exception."""

    # ---- Lifecycle ----

    @abstractmethod
    def connect(self) -> None:
        """Open the IMAP connection. Idempotent (no-op if already connected)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the IMAP connection. Must not raise — log and swallow errors."""

    # ---- Operations ----

    @abstractmethod
    def fetch_unread(self, limit: int | None = None) -> Iterator[RawEmail]:
        """Yield unread emails from the configured folder.

        IMPORTANT: implementations MUST NOT mark emails as `\\Seen` during fetch.
        That's the orchestrator's responsibility, AFTER successful processing
        and delivery. This guarantees idempotency: a crash mid-pipeline won't
        cause emails to be silently skipped on the next run.
        """

    @abstractmethod
    def mark_as_seen(self, uid: str) -> None:
        """Flag a message as read on the server. Called by the orchestrator
        only after the message has been fully processed AND delivered."""

    # ---- Context manager protocol (concrete; subclasses don't override) ----

    def __enter__(self) -> EmailClient:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()

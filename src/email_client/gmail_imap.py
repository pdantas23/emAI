"""Gmail IMAP client built on top of `imap-tools`.

Although named "Gmail", this implementation works for any IMAP server
(Outlook, Fastmail, custom). Gmail-specific notes:
- Authentication MUST use an App Password, not the account password.
  Generate one at https://myaccount.google.com/apppasswords
- Two-Factor Authentication is required to enable App Passwords.

Idempotency contract (see `EmailClient.fetch_unread` docstring):
- We pass `mark_seen=False` to imap-tools, so emails REMAIN unread on the
  server until the orchestrator explicitly calls `mark_as_seen(uid)` after
  the WhatsApp report has been delivered. A crash mid-pipeline is safe —
  the same emails will be picked up on the next run.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from imap_tools import AND, MailBox, MailboxLoginError, MailMessage

from config.settings import settings
from src.email_client.base import Attachment, EmailClient, RawEmail
from src.utils.logger import log


class GmailIMAPClient(EmailClient):
    """IMAP client implementation. Reads connection settings from `settings.imap`."""

    def __init__(self) -> None:
        cfg = settings.imap
        self._host: str = cfg.host
        self._port: int = cfg.port
        self._username: str = cfg.username
        self._password: str = cfg.password.get_secret_value()
        self._folder: str = cfg.folder
        self._fetch_limit: int = cfg.fetch_limit
        self._max_attachment_bytes: int = settings.filters.attachment_max_mb * 1024 * 1024
        self._mailbox: MailBox | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        if self._mailbox is not None:
            return  # idempotent
        log.info(
            "Connecting to IMAP {}:{} as {}", self._host, self._port, self._username
        )
        try:
            self._mailbox = MailBox(self._host, port=self._port).login(
                self._username, self._password, initial_folder=self._folder
            )
        except MailboxLoginError as exc:
            log.error("IMAP login failed for {}: {}", self._username, exc)
            raise
        log.info("IMAP connected; folder='{}'", self._folder)

    def disconnect(self) -> None:
        if self._mailbox is None:
            return
        try:
            self._mailbox.logout()
            log.info("IMAP disconnected")
        except Exception as exc:  # noqa: BLE001 — disconnect must never raise
            log.warning("IMAP logout error (ignored): {}", exc)
        finally:
            self._mailbox = None

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #

    def fetch_unread(self, limit: int | None = None) -> Iterator[RawEmail]:
        if self._mailbox is None:
            raise RuntimeError(
                "EmailClient not connected. Use as a context manager or call connect() first."
            )

        effective_limit = min(limit or self._fetch_limit, self._fetch_limit)
        log.info("Fetching up to {} unread email(s) from '{}'", effective_limit, self._folder)

        count = 0
        # mark_seen=False is the cornerstone of our idempotency guarantee.
        for msg in self._mailbox.fetch(
            criteria=AND(seen=False),
            limit=effective_limit,
            mark_seen=False,
            bulk=True,
            reverse=True,
        ):
            try:
                yield self._convert(msg)
                count += 1
            except Exception as exc:  # noqa: BLE001 — one bad email shouldn't kill the run
                log.error("Failed to convert message uid={}: {}", msg.uid, exc)
                continue
        log.info("Fetched {} unread email(s)", count)

    def mark_as_seen(self, uid: str) -> None:
        if self._mailbox is None:
            raise RuntimeError("EmailClient not connected.")
        self._mailbox.flag(uid, "\\Seen", True)
        log.debug("Marked uid={} as \\Seen", uid)

    # ------------------------------------------------------------------ #
    # Internal helpers — translate imap-tools types into our domain model
    # ------------------------------------------------------------------ #

    def _convert(self, msg: MailMessage) -> RawEmail:
        """Map an `imap-tools` MailMessage onto our provider-agnostic `RawEmail`."""
        sender_name, sender_email = self._extract_sender(msg)

        return RawEmail(
            uid=str(msg.uid),
            message_id=self._extract_message_id(msg),
            folder=self._folder,
            sender_name=sender_name,
            sender_email=sender_email,
            to=list(msg.to),
            cc=list(msg.cc),
            subject=msg.subject or "",
            date=msg.date or datetime.now(timezone.utc),
            body_text=msg.text or "",
            body_html=msg.html or "",
            headers=self._flatten_headers(msg.headers),
            attachments=[self._convert_attachment(a) for a in msg.attachments],
            spam_flag=self._infer_spam_flag(msg.headers),
        )

    @staticmethod
    def _extract_sender(msg: MailMessage) -> tuple[str | None, str]:
        if msg.from_values:
            name = (msg.from_values.name or "").strip() or None
            email = (msg.from_values.email or "").strip().lower()
            return name, email
        # Fallback: bare email string with no display name.
        return None, (msg.from_ or "").strip().lower()

    @staticmethod
    def _extract_message_id(msg: MailMessage) -> str:
        """Pull `Message-ID` out of headers, stripping the surrounding angle brackets.
        Falls back to a UID-based synthetic ID if missing (rare but legal)."""
        raw = msg.headers.get("message-id", ())
        if raw:
            value = raw[0] if isinstance(raw, tuple) else str(raw)
            return value.strip().strip("<>")
        return f"no-id-{msg.uid}"

    @staticmethod
    def _flatten_headers(headers: dict[str, Any]) -> dict[str, str]:
        """imap-tools returns headers as `dict[str, tuple[str, ...]]`.
        We flatten to `dict[str, str]` (joining multi-values with `, `)."""
        out: dict[str, str] = {}
        for key, value in headers.items():
            if isinstance(value, tuple):
                out[key] = ", ".join(str(v) for v in value)
            else:
                out[key] = str(value)
        return out

    def _convert_attachment(self, att: Any) -> Attachment:
        size = len(att.payload) if att.payload else 0
        skip = size > self._max_attachment_bytes
        if skip:
            log.warning(
                "Attachment '{}' ({:.1f} MB) exceeds {} MB limit — skipping payload",
                att.filename,
                size / (1024 * 1024),
                settings.filters.attachment_max_mb,
            )
        return Attachment(
            filename=att.filename or "unnamed",
            content_type=att.content_type or "application/octet-stream",
            size_bytes=size,
            content=None if skip else att.payload,
        )

    @staticmethod
    def _infer_spam_flag(headers: dict[str, Any]) -> bool:
        """Best-effort detection of server-side spam flags.
        Honors common headers set by Gmail, SpamAssassin, etc."""
        for key in ("x-spam-flag", "x-spam-status"):
            raw = headers.get(key)
            if not raw:
                continue
            value = raw[0] if isinstance(raw, tuple) else str(raw)
            if "yes" in value.lower():
                return True
        return False

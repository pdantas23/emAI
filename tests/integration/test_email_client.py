"""Integration tests for `src/email_client/`.

Strategy: real `.eml` fixtures are parsed with the stdlib `email` module
into a `FakeMailMessage` that mirrors the attributes our `gmail_imap.py`
expects from `imap_tools.MailMessage`. The IMAP wire protocol itself is
mocked — what we care about is:

  1. The parser correctly cleans HTML / extracts text (PURE — no I/O).
  2. `_convert()` correctly maps MailMessage → RawEmail.
  3. The fetch loop honors the idempotency contract (`mark_seen=False`).
  4. Errors on a single message do not derail the whole run.

We do NOT hit a real IMAP server in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from email import message_from_bytes
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.email_client.base import Attachment, RawEmail
from src.email_client.gmail_imap import GmailIMAPClient
from src.email_client.parser import (
    clean_html_to_text,
    extract_text_from_email,
    parse_sender,
)


# =========================================================================== #
# Helpers — fake imap-tools structures populated from real .eml files
# =========================================================================== #


@dataclass
class FakeAddress:
    """Mirrors `imap_tools.message.EmailAddress`."""

    name: str
    email: str
    full: str = ""


@dataclass
class FakeAttachment:
    """Mirrors `imap_tools.message.MailAttachment`."""

    filename: str
    content_type: str
    payload: bytes


@dataclass
class FakeMailMessage:
    """Mirrors the subset of `imap_tools.MailMessage` attributes we use."""

    uid: str
    from_: str
    from_values: FakeAddress | None
    to: tuple[str, ...]
    cc: tuple[str, ...]
    subject: str
    date: datetime | None
    text: str
    html: str
    headers: dict[str, tuple[str, ...]]
    attachments: list[FakeAttachment] = field(default_factory=list)


def _load_eml(path: Path, uid: str = "1001") -> FakeMailMessage:
    """Parse an `.eml` file into the FakeMailMessage shape our code expects."""
    msg = message_from_bytes(path.read_bytes())

    text_part = ""
    html_part = ""
    attachments: list[FakeAttachment] = []

    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        disposition = (part.get("Content-Disposition") or "").lower()
        payload_bytes = part.get_payload(decode=True) or b""

        if "attachment" in disposition:
            attachments.append(
                FakeAttachment(
                    filename=part.get_filename() or "unnamed",
                    content_type=ctype,
                    payload=payload_bytes,
                )
            )
            continue

        charset = part.get_content_charset() or "utf-8"
        decoded = payload_bytes.decode(charset, errors="replace")
        if ctype == "text/plain" and not text_part:
            text_part = decoded
        elif ctype == "text/html" and not html_part:
            html_part = decoded

    raw_from = msg.get("From", "")
    name, addr = parseaddr(raw_from)
    from_values = (
        FakeAddress(name=name, email=addr.lower(), full=raw_from) if addr else None
    )

    headers: dict[str, tuple[str, ...]] = {}
    for k, v in msg.items():
        key = k.lower()
        headers[key] = headers.get(key, ()) + (str(v),)

    date_str = msg.get("Date")
    date = parsedate_to_datetime(date_str) if date_str else None

    def _split_addrs(value: str | None) -> tuple[str, ...]:
        return tuple(s.strip() for s in (value or "").split(",") if s.strip())

    return FakeMailMessage(
        uid=uid,
        from_=addr.lower() if addr else "",
        from_values=from_values,
        to=_split_addrs(msg.get("To")),
        cc=_split_addrs(msg.get("Cc")),
        subject=msg.get("Subject") or "",
        date=date,
        text=text_part,
        html=html_part,
        headers=headers,
        attachments=attachments,
    )


# =========================================================================== #
# parser.py — pure functions, no I/O
# =========================================================================== #


class TestCleanHtmlToText:
    def test_empty_input_returns_empty(self) -> None:
        assert clean_html_to_text("") == ""
        assert clean_html_to_text("   \n  ") == ""

    def test_basic_html_returns_text(self) -> None:
        out = clean_html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in out
        assert "world" in out

    def test_strips_script_and_style(self) -> None:
        html = "<style>body{}</style><script>evil()</script><p>Visible</p>"
        out = clean_html_to_text(html)
        assert "evil" not in out
        assert "body{}" not in out
        assert "Visible" in out

    def test_removes_1x1_tracking_pixel(self) -> None:
        html = '<p>Body</p><img src="https://track.com/p.gif" width="1" height="1">'
        out = clean_html_to_text(html)
        assert "track.com" not in out
        assert "Body" in out

    def test_keeps_links_for_llm_context(self) -> None:
        html = '<p>Click <a href="https://example.com/promo">here</a></p>'
        out = clean_html_to_text(html)
        assert "https://example.com/promo" in out

    def test_collapses_multiple_blank_lines(self) -> None:
        html = "<p>One</p>\n\n\n\n\n<p>Two</p>"
        out = clean_html_to_text(html)
        # Should never have more than one blank line in a row
        assert "\n\n\n" not in out

    def test_unparseable_html_falls_back_gracefully(self) -> None:
        # BeautifulSoup is very tolerant; this should not raise
        out = clean_html_to_text("<<not really html>>")
        assert isinstance(out, str)


class TestExtractTextFromEmail:
    def _make_email(self, *, text: str = "", html: str = "") -> RawEmail:
        return RawEmail(
            uid="1",
            message_id="x@x",
            folder="INBOX",
            sender_email="a@b.c",
            date=datetime.now(),
            body_text=text,
            body_html=html,
        )

    def test_prefers_plain_text_when_available(self) -> None:
        email = self._make_email(text="Plain version", html="<p>HTML version</p>")
        assert extract_text_from_email(email) == "Plain version"

    def test_falls_back_to_html_when_no_plain(self) -> None:
        email = self._make_email(html="<p>Only HTML</p>")
        out = extract_text_from_email(email)
        assert "Only HTML" in out

    def test_returns_empty_when_no_body_at_all(self) -> None:
        email = self._make_email()
        assert extract_text_from_email(email) == ""

    def test_strip_signature_removes_after_delimiter(self) -> None:
        email = self._make_email(text="Body content here\n-- \nJoao Silva\nDirector")
        out = extract_text_from_email(email, strip_signature=True)
        assert "Body content here" in out
        assert "Joao Silva" not in out

    def test_strip_signature_default_keeps_signature(self) -> None:
        email = self._make_email(text="Body\n-- \nSignature line")
        out = extract_text_from_email(email)  # strip_signature=False
        assert "Signature line" in out


class TestParseSender:
    def test_with_display_name(self) -> None:
        name, addr = parse_sender("Joao Silva <joao@example.com>")
        assert name == "Joao Silva"
        assert addr == "joao@example.com"

    def test_bare_email_returns_none_name(self) -> None:
        name, addr = parse_sender("plain@example.com")
        assert name is None
        assert addr == "plain@example.com"

    def test_lowercases_email_address(self) -> None:
        _, addr = parse_sender("MixedCase@Example.COM")
        assert addr == "mixedcase@example.com"

    def test_empty_string(self) -> None:
        name, addr = parse_sender("")
        assert name is None
        assert addr == ""


# =========================================================================== #
# gmail_imap.py — `_convert()` and helpers, exercised via real .eml fixtures
# =========================================================================== #


@pytest.fixture
def client() -> GmailIMAPClient:
    """Construct a client without connecting (no IMAP I/O)."""
    return GmailIMAPClient()


class TestConvertSimpleEmail:
    def test_yields_raw_email_with_correct_fields(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "simple.eml", uid="42")
        result = client._convert(msg)

        assert isinstance(result, RawEmail)
        assert result.uid == "42"
        assert result.message_id == "simple-abc123@example.com"  # brackets stripped
        assert result.sender_email == "joao.silva@example.com"
        assert "Reuniao" in result.subject
        assert "remarcar" in result.body_text
        assert result.body_html == ""
        assert result.has_attachments is False
        assert result.spam_flag is False
        assert result.folder == client._folder

    def test_sender_email_is_lowercase(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "simple.eml")
        msg.from_values = FakeAddress(name="X", email="UPPER@EXAMPLE.COM")
        result = client._convert(msg)
        assert result.sender_email == "upper@example.com"


class TestConvertHtmlHeavyEmail:
    def test_keeps_both_text_and_html_bodies(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "html_heavy.eml")
        result = client._convert(msg)

        assert result.body_text  # text/plain alternative was present
        assert result.body_html  # text/html alternative was present
        assert result.has_html is True
        assert "50%" in result.body_text or "50% off" in result.body_text

    def test_detects_server_spam_flag(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "html_heavy.eml")
        result = client._convert(msg)
        # Fixture has X-Spam-Status: Yes, score=7.2
        assert result.spam_flag is True

    def test_html_body_can_be_cleaned_to_text_downstream(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "html_heavy.eml")
        raw = client._convert(msg)
        cleaned = clean_html_to_text(raw.body_html)

        # Tracking pixel removed
        assert "track.example.com" not in cleaned
        # Script gone
        assert "analytics.track" not in cleaned
        # Style gone
        assert "font-family" not in cleaned
        # Real link preserved
        assert "https://promo.example.com" in cleaned
        # Content preserved
        assert "SAVE50" in cleaned


class TestConvertEmailWithAttachment:
    def test_extracts_attachment_metadata(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "with_attachment.eml")
        result = client._convert(msg)

        assert result.has_attachments is True
        assert len(result.attachments) == 1
        att = result.attachments[0]
        assert att.filename == "relatorio_q1_2026.pdf"
        assert att.content_type == "application/pdf"
        assert att.size_bytes > 0
        assert att.was_skipped is False
        assert att.content is not None

    def test_attachment_skipped_when_over_limit(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        # Force a tiny limit so the fixture's payload exceeds it
        client._max_attachment_bytes = 5  # 5 bytes
        msg = _load_eml(fixtures_dir / "with_attachment.eml")
        result = client._convert(msg)

        att = result.attachments[0]
        assert att.was_skipped is True
        assert att.content is None
        assert att.size_bytes > 5  # metadata still preserved

    def test_text_body_present_alongside_attachment(
        self, client: GmailIMAPClient, fixtures_dir: Path
    ) -> None:
        msg = _load_eml(fixtures_dir / "with_attachment.eml")
        result = client._convert(msg)
        assert "relatorio" in result.body_text.lower()


class TestExtractMessageIdHelper:
    def test_strips_angle_brackets(self, client: GmailIMAPClient) -> None:
        msg = _load_eml_minimal(uid="9", message_id="<abc@x.com>")
        assert client._extract_message_id(msg) == "abc@x.com"

    def test_falls_back_to_uid_when_missing(self, client: GmailIMAPClient) -> None:
        msg = _load_eml_minimal(uid="9", message_id=None)
        assert client._extract_message_id(msg) == "no-id-9"


class TestFlattenHeadersHelper:
    def test_joins_tuple_values_with_comma(self, client: GmailIMAPClient) -> None:
        flat = client._flatten_headers({"received": ("by host1", "by host2")})
        assert flat["received"] == "by host1, by host2"

    def test_handles_non_tuple_value(self, client: GmailIMAPClient) -> None:
        flat = client._flatten_headers({"single": "just-a-string"})
        assert flat["single"] == "just-a-string"


class TestInferSpamFlagHelper:
    def test_x_spam_flag_yes(self, client: GmailIMAPClient) -> None:
        assert client._infer_spam_flag({"x-spam-flag": ("YES",)}) is True

    def test_x_spam_status_yes(self, client: GmailIMAPClient) -> None:
        assert client._infer_spam_flag({"x-spam-status": ("Yes, score=8",)}) is True

    def test_no_spam_headers(self, client: GmailIMAPClient) -> None:
        assert client._infer_spam_flag({"subject": ("hello",)}) is False

    def test_x_spam_flag_no(self, client: GmailIMAPClient) -> None:
        assert client._infer_spam_flag({"x-spam-flag": ("NO",)}) is False


def _load_eml_minimal(uid: str, message_id: str | None) -> FakeMailMessage:
    """Build a minimal FakeMailMessage for helper-level tests."""
    headers: dict[str, tuple[str, ...]] = {}
    if message_id is not None:
        headers["message-id"] = (message_id,)
    return FakeMailMessage(
        uid=uid,
        from_="x@y.z",
        from_values=FakeAddress("X", "x@y.z"),
        to=(),
        cc=(),
        subject="",
        date=None,
        text="",
        html="",
        headers=headers,
    )


# =========================================================================== #
# fetch_unread loop — integration with a mocked MailBox
# =========================================================================== #


class TestFetchUnread:
    """All these tests mock `imap_tools.MailBox` and exercise the loop logic."""

    def _connected_client_with_messages(
        self, messages: list[Any]
    ) -> tuple[GmailIMAPClient, MagicMock]:
        client = GmailIMAPClient()
        mailbox = MagicMock()
        mailbox.fetch.return_value = iter(messages)
        client._mailbox = mailbox  # bypass real connection
        return client, mailbox

    def test_yields_raw_email_for_each_message(
        self, fixtures_dir: Path
    ) -> None:
        msgs = [
            _load_eml(fixtures_dir / "simple.eml", uid="1"),
            _load_eml(fixtures_dir / "html_heavy.eml", uid="2"),
            _load_eml(fixtures_dir / "with_attachment.eml", uid="3"),
        ]
        client, _ = self._connected_client_with_messages(msgs)

        results = list(client.fetch_unread())
        assert len(results) == 3
        assert {r.uid for r in results} == {"1", "2", "3"}
        assert all(isinstance(r, RawEmail) for r in results)

    def test_passes_mark_seen_false_for_idempotency(
        self, fixtures_dir: Path
    ) -> None:
        """CRITICAL contract: emails MUST stay unread on the server until the
        orchestrator explicitly marks them seen post-delivery."""
        client, mailbox = self._connected_client_with_messages([])

        list(client.fetch_unread())

        _, kwargs = mailbox.fetch.call_args
        assert kwargs["mark_seen"] is False, (
            "fetch_unread() must pass mark_seen=False — see EmailClient.fetch_unread docstring"
        )

    def test_filters_by_unseen_only(self, fixtures_dir: Path) -> None:
        client, mailbox = self._connected_client_with_messages([])
        list(client.fetch_unread())

        # imap_tools' AND(seen=False) renders to the IMAP search criteria 'UNSEEN'
        from imap_tools import AND

        _, kwargs = mailbox.fetch.call_args
        criteria = kwargs["criteria"]
        # AND(seen=False) is what we passed; comparing the rendered string
        assert str(criteria) == str(AND(seen=False))

    def test_honors_limit_parameter(self, fixtures_dir: Path) -> None:
        client, mailbox = self._connected_client_with_messages([])
        list(client.fetch_unread(limit=7))

        _, kwargs = mailbox.fetch.call_args
        assert kwargs["limit"] == 7

    def test_caps_limit_at_settings_fetch_limit(self, fixtures_dir: Path) -> None:
        client, mailbox = self._connected_client_with_messages([])
        client._fetch_limit = 10
        list(client.fetch_unread(limit=999))  # request more than allowed

        _, kwargs = mailbox.fetch.call_args
        assert kwargs["limit"] == 10  # capped

    def test_skips_bad_message_and_continues(self, fixtures_dir: Path) -> None:
        good = _load_eml(fixtures_dir / "simple.eml", uid="1")
        bad = MagicMock()
        bad.uid = "2"
        # Accessing any of these will raise — _convert will fail
        bad.from_values = property(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
        bad.headers = {}
        good2 = _load_eml(fixtures_dir / "simple.eml", uid="3")

        client, _ = self._connected_client_with_messages([good, bad, good2])
        results = list(client.fetch_unread())

        # The bad one is skipped; the run is NOT aborted
        assert {r.uid for r in results} == {"1", "3"}

    def test_raises_when_not_connected(self) -> None:
        client = GmailIMAPClient()
        with pytest.raises(RuntimeError, match="not connected"):
            list(client.fetch_unread())


class TestMarkAsSeen:
    def test_calls_flag_with_seen_true(self) -> None:
        client = GmailIMAPClient()
        mailbox = MagicMock()
        client._mailbox = mailbox

        client.mark_as_seen("42")
        mailbox.flag.assert_called_once_with("42", "\\Seen", True)

    def test_raises_when_not_connected(self) -> None:
        client = GmailIMAPClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client.mark_as_seen("42")


class TestContextManager:
    @patch("src.email_client.gmail_imap.MailBox")
    def test_enter_connects_and_exit_disconnects(self, mailbox_cls: MagicMock) -> None:
        mailbox_instance = MagicMock()
        mailbox_cls.return_value.login.return_value = mailbox_instance

        with GmailIMAPClient() as client:
            assert client._mailbox is mailbox_instance

        # After exit, mailbox is released and logout was called
        mailbox_instance.logout.assert_called_once()

    def test_disconnect_swallows_logout_errors(self) -> None:
        client = GmailIMAPClient()
        mailbox = MagicMock()
        mailbox.logout.side_effect = OSError("connection already dead")
        client._mailbox = mailbox

        # Must not raise — disconnect contract from EmailClient ABC
        client.disconnect()
        assert client._mailbox is None

    def test_connect_is_idempotent(self) -> None:
        """Calling connect() twice should not open a second connection."""
        client = GmailIMAPClient()
        sentinel = MagicMock()
        client._mailbox = sentinel

        client.connect()  # should be a no-op
        assert client._mailbox is sentinel

    def test_disconnect_when_never_connected_is_noop(self) -> None:
        client = GmailIMAPClient()
        client.disconnect()  # should silently return — no AttributeError
        assert client._mailbox is None

    @patch("src.email_client.gmail_imap.MailBox")
    def test_login_error_is_logged_and_propagated(
        self, mailbox_cls: MagicMock
    ) -> None:
        from imap_tools import MailboxLoginError

        # Constructor signature varies across imap-tools versions; build the
        # exception by hand and populate the attrs its __str__ touches, so the
        # test isn't pinned to one specific release.
        exc = MailboxLoginError.__new__(MailboxLoginError)
        exc.command_result = ("NO", b"bad credentials")
        exc.expected = "OK"
        mailbox_cls.return_value.login.side_effect = exc
        client = GmailIMAPClient()
        with pytest.raises(MailboxLoginError):
            client.connect()
        assert client._mailbox is None  # failure leaves us in clean state


# =========================================================================== #
# Edge cases — defensive paths that complete the 100% coverage picture
# =========================================================================== #


class TestEdgeCases:
    def test_attachment_zero_bytes_is_not_marked_skipped(self) -> None:
        """An empty attachment (size 0, content None) is NOT a 'skipped' one —
        the skip flag should only fire when we actively dropped a payload."""
        att = Attachment(
            filename="empty.txt", content_type="text/plain", size_bytes=0, content=None
        )
        assert att.was_skipped is False

    def test_attachment_size_mb_property(self) -> None:
        att = Attachment(
            filename="a.bin", content_type="x/y", size_bytes=2 * 1024 * 1024, content=b"x"
        )
        assert att.size_mb == 2.0

    def test_extract_sender_falls_back_to_from_when_no_values(
        self, fixtures_dir: Path
    ) -> None:
        client = GmailIMAPClient()
        msg = _load_eml(fixtures_dir / "simple.eml")
        # Simulate imap-tools failing to parse address structure
        msg.from_values = None
        msg.from_ = "Fallback@Example.COM"
        result = client._convert(msg)
        assert result.sender_email == "fallback@example.com"
        assert result.sender_name is None

    def test_html_parse_failure_falls_back_to_raw_html(self) -> None:
        """If BeautifulSoup itself raises (e.g. monkey-patched parser), we must
        not crash — fall back to running html2text on the raw input."""
        with patch(
            "src.email_client.parser.BeautifulSoup",
            side_effect=RuntimeError("parser exploded"),
        ):
            out = clean_html_to_text("<p>still works</p>")
        assert "still works" in out

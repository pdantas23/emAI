"""Integration tests for `src/core/orchestrator.py`.

Coverage map:

  ┌────────────────────────────┬────────────────────────────────────────────┐
  │ Concern                    │ What we prove                              │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Happy path                 │ One run end-to-end: classify → summarize → │
  │                            │ render → send → persist → mark seen.       │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Token-saver gate           │ Irrelevant email NEVER reaches summarizer; │
  │                            │ row stored as `skipped_irrelevant`; never  │
  │                            │ enters the WhatsApp digest.                │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Dedup short-circuit        │ Already-processed email skips classifier,  │
  │                            │ summarizer AND WhatsApp.                   │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Per-email error isolation  │ classifier / summarizer exceptions on ONE  │
  │                            │ email leave the rest fully processed; the  │
  │                            │ failed email is NOT persisted and NOT      │
  │                            │ marked seen (so next run retries it).      │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Per-email delivery         │ Each relevant email triggers exactly ONE   │
  │                            │ `whatsapp.send(body)` call — no batching,  │
  │                            │ no concatenation (Twilio 21617 guard).     │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ WhatsApp delivery failure  │ Per-email: a failed send leaves only THAT  │
  │                            │ email unpersisted and unseen; previously   │
  │                            │ delivered emails in the same run stay      │
  │                            │ sealed; later emails still get their turn. │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Idempotency contract       │ DB row written BEFORE the IMAP \\Seen flag │
  │                            │ — verified via shared call recorder.       │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Storage fail-stop          │ `StorageError` from the state store        │
  │                            │ propagates and halts the run.              │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ IMAP seal failure          │ `mark_as_seen` raising AFTER the DB write  │
  │                            │ is logged but does NOT count as failure    │
  │                            │ (DB dedup will protect the next run).      │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Log tag vocabulary         │ Every operator-facing tag is emitted at    │
  │                            │ the right step: [FETCH] [SKIP] [CLASSIFY]  │
  │                            │ [IRRELEVANT] [SUMMARIZE] [SEND] [DONE]     │
  │                            │ [ERROR] [RUN].                             │
  └────────────────────────────┴────────────────────────────────────────────┘

We use **fakes**, not mocks, for the collaborator interfaces. A `MagicMock`
would let the orchestrator call any method and silently succeed; a fake makes
the contract explicit and failures loud. The state store is the REAL one,
backed by an in-memory SQLite — we already proved its behavior in
`test_storage.py` and want to exercise the orchestrator against the same
implementation production will run.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest
from loguru import logger as loguru_logger
from sqlmodel import create_engine

from src.ai.classifier import Classification, EmailClassifier, Priority
from src.ai.summarizer import EmailSummarizer, EmailSummary, SummaryFailure
from src.core.orchestrator import Orchestrator, RunStats
from src.email_client.base import EmailClient, RawEmail
from src.messaging.whatsapp_twilio import WhatsAppClient, WhatsAppDeliveryError
from src.storage.models import DeliveryStatus
from src.storage.state import StateStore, StorageError


# =========================================================================== #
# Fakes — explicit, programmable, and noisy on misuse.
# =========================================================================== #


class _CallRecorder:
    """Shared chronological log of every collaborator call.

    Used by the idempotency-order test to prove the DB write happens BEFORE
    the IMAP \\Seen flag for every single email. Each call is appended as a
    `(collaborator, action, identifier)` tuple so assertions read naturally.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []

    def record(self, collaborator: str, action: str, identifier: str) -> None:
        self.events.append((collaborator, action, identifier))


class FakeEmailClient(EmailClient):
    """In-memory replacement for `GmailIMAPClient`.

    Implements the full `EmailClient` contract so the orchestrator's
    `with self._email as mailbox:` block works unchanged. Tracks every
    `mark_as_seen` call in arrival order (used heavily in assertions).
    """

    def __init__(
        self,
        emails: list[RawEmail],
        *,
        recorder: _CallRecorder | None = None,
        mark_seen_failures: dict[str, Exception] | None = None,
    ) -> None:
        self._emails = emails
        self._recorder = recorder
        self._mark_seen_failures = mark_seen_failures or {}
        self.connected = False
        self.disconnected = False
        self.seen_uids: list[str] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def fetch_unread(self, limit: int | None = None) -> Iterator[RawEmail]:
        for email in self._emails:
            yield email

    def mark_as_seen(self, uid: str) -> None:
        if self._recorder is not None:
            self._recorder.record("imap", "mark_as_seen", uid)
        if uid in self._mark_seen_failures:
            raise self._mark_seen_failures[uid]
        self.seen_uids.append(uid)


class FakeClassifier:
    """Programmable classifier with a per-message_id verdict map.

    Default verdict is `(relevance=True, priority=medium)` so the obvious
    happy-path tests don't need explicit setup. Pass `raises_for` to make
    `classify` throw for specific message_ids — used by the loop-resilience
    tests.
    """

    def __init__(
        self,
        verdicts: dict[str, Classification] | None = None,
        *,
        raises_for: dict[str, Exception] | None = None,
    ) -> None:
        self.verdicts = verdicts or {}
        self.raises_for = raises_for or {}
        self.calls: list[str] = []  # message_ids we were asked about

    def classify(self, email: RawEmail) -> Classification:
        self.calls.append(email.message_id)
        if email.message_id in self.raises_for:
            raise self.raises_for[email.message_id]
        return self.verdicts.get(
            email.message_id,
            Classification(relevance=True, priority=Priority.medium, reason="ok"),
        )


class FakeSummarizer:
    """Programmable summarizer with per-message_id summaries / failures.

    The `calls` list is the load-bearing instrument for the token-saver
    test: it must remain EMPTY whenever the classifier rejects an email.
    """

    def __init__(
        self,
        summaries: dict[str, EmailSummary] | None = None,
        *,
        raises_for: dict[str, Exception] | None = None,
    ) -> None:
        self.summaries = summaries or {}
        self.raises_for = raises_for or {}
        self.calls: list[str] = []  # message_ids we were asked to summarize

    def summarize(
        self,
        email: RawEmail,
        *,
        priority: Priority = Priority.medium,
    ) -> EmailSummary:
        self.calls.append(email.message_id)
        if email.message_id in self.raises_for:
            raise self.raises_for[email.message_id]
        return self.summaries.get(
            email.message_id,
            EmailSummary(
                resumo=f"Resumo de {email.subject or email.message_id}",
                contexto="Contexto fixo de teste — duas frases bastam aqui.",
                acao="Tomar uma decisão até o fim do dia.",
            ),
        )


class FakeWhatsApp:
    """Programmable Twilio replacement.

    Tracks every `send(body)` call as a separate entry in `bodies`, so
    assertions can inspect both HOW MANY sends happened and WHAT was in
    each one. Set `raise_on_send` to simulate Twilio outages — either as a
    single exception (raised for every call) or as a dict keyed by a
    substring match against the body (raised only when that substring
    appears). The latter lets tests prove that a single failed send does
    NOT affect sibling emails in the same run.
    """

    def __init__(
        self,
        *,
        recorder: _CallRecorder | None = None,
        raise_on_send: Exception | None = None,
        raise_for_body_contains: dict[str, Exception] | None = None,
    ) -> None:
        self._recorder = recorder
        self._raise = raise_on_send
        self._raise_for_body_contains = raise_for_body_contains or {}
        self.bodies: list[str] = []
        self.returned_sids: list[str] = []

    @property
    def send_calls(self) -> int:
        return len(self.bodies)

    def send(self, body: str) -> str:
        self.bodies.append(body)
        if self._recorder is not None:
            self._recorder.record("whatsapp", "send", str(len(self.bodies)))
        if self._raise is not None:
            raise self._raise
        for needle, exc in self._raise_for_body_contains.items():
            if needle in body:
                raise exc
        sid = f"SM{len(self.bodies):032d}"
        self.returned_sids.append(sid)
        return sid


# =========================================================================== #
# Helpers
# =========================================================================== #


def _make_email(
    *,
    uid: str,
    message_id: str | None = None,
    subject: str = "Assunto teste",
    sender: str = "remetente@example.com",
) -> RawEmail:
    return RawEmail(
        uid=uid,
        message_id=message_id or f"<{uid}@test.local>",
        folder="INBOX",
        sender_email=sender,
        sender_name="Remetente Teste",
        subject=subject,
        date=datetime(2026, 4, 16, 9, 30),
        body_text="corpo do email",
    )


def _build_orchestrator(
    *,
    emails: list[RawEmail],
    store: StateStore,
    classifier: FakeClassifier | None = None,
    summarizer: FakeSummarizer | None = None,
    whatsapp: FakeWhatsApp | None = None,
    recorder: _CallRecorder | None = None,
    mark_seen_failures: dict[str, Exception] | None = None,
) -> tuple[Orchestrator, FakeEmailClient, FakeClassifier, FakeSummarizer, FakeWhatsApp]:
    """One-call wiring helper. Returns the orchestrator AND every fake so the
    test can assert against them without re-fetching from the orchestrator."""
    email_client = FakeEmailClient(
        emails, recorder=recorder, mark_seen_failures=mark_seen_failures
    )
    classifier = classifier or FakeClassifier()
    summarizer = summarizer or FakeSummarizer()
    whatsapp = whatsapp or FakeWhatsApp(recorder=recorder)

    orch = Orchestrator(
        email_client=email_client,
        classifier=classifier,  # type: ignore[arg-type]  — duck-typed
        summarizer=summarizer,  # type: ignore[arg-type]
        whatsapp=whatsapp,      # type: ignore[arg-type]
        state=store,
    )
    return orch, email_client, classifier, summarizer, whatsapp


# =========================================================================== #
# Fixtures
# =========================================================================== #


@pytest.fixture
def store() -> StateStore:
    return StateStore(engine=create_engine("sqlite:///:memory:"))


@pytest.fixture
def loguru_capture() -> Iterator[list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []
    handler_id = loguru_logger.add(
        lambda msg: captured.append(dict(msg.record)),
        level="DEBUG",
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(handler_id)


def _messages(records: list[dict[str, Any]]) -> list[str]:
    """Flatten captured loguru records to their `message` field."""
    return [r["message"] for r in records]


# =========================================================================== #
# 1. Happy path
# =========================================================================== #


class TestHappyPath:
    def test_single_relevant_email_runs_full_pipeline(
        self, store: StateStore
    ) -> None:
        email = _make_email(uid="1")
        orch, ec, clf, smr, wa = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier(
                {email.message_id: Classification(
                    relevance=True, priority=Priority.high, reason="prazo curto"
                )},
            ),
        )

        stats = orch.run()

        assert stats.fetched == 1
        assert stats.delivered == 1
        assert stats.failed == 0
        assert clf.calls == [email.message_id]
        assert smr.calls == [email.message_id]
        assert wa.send_calls == 1
        assert ec.seen_uids == [email.uid]
        assert ec.connected and ec.disconnected
        # Persisted with the expected fields.
        row = store.get(email.message_id)
        assert row is not None
        assert row.delivery_status == DeliveryStatus.delivered.value
        assert row.priority == "high"

    def test_empty_inbox_makes_no_calls_and_no_send(
        self, store: StateStore
    ) -> None:
        orch, ec, clf, smr, wa = _build_orchestrator(emails=[], store=store)
        stats = orch.run()
        assert stats == RunStats()  # everything zero
        assert clf.calls == []
        assert smr.calls == []
        assert wa.send_calls == 0
        assert ec.seen_uids == []


# =========================================================================== #
# 2. Token-saver gate — the most important test in this file
# =========================================================================== #


class TestRelevanceGate:
    def test_irrelevant_email_never_calls_summarizer(
        self, store: StateStore
    ) -> None:
        """The whole point of having a cheap classifier is that we DO NOT
        spend Sonnet tokens on emails that fail the gate. If this regresses
        the project loses its cost story."""
        email = _make_email(uid="1", subject="Newsletter semanal")
        irrelevant = Classification(
            relevance=False, priority=Priority.low, reason="newsletter"
        )

        orch, _, _, smr, wa = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier({email.message_id: irrelevant}),
        )
        stats = orch.run()

        assert smr.calls == [], "summarizer must NOT be called for irrelevant emails"
        assert wa.send_calls == 0, "no relevant emails → no WhatsApp send"
        assert stats.skipped_irrelevant == 1
        assert stats.delivered == 0

    def test_irrelevant_email_persisted_as_skipped_and_marked_seen(
        self, store: StateStore
    ) -> None:
        email = _make_email(uid="7")
        orch, ec, *_ = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier(
                {email.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="spam-ish"
                )},
            ),
        )
        orch.run()

        row = store.get(email.message_id)
        assert row is not None
        assert row.delivery_status == DeliveryStatus.skipped_irrelevant.value
        # Privacy contract: no summary fields exist on the row at all; all
        # we keep for skipped emails is the Message-ID/UID + verdict metadata.
        assert row.twilio_sids is None
        assert row.relevance is False
        assert ec.seen_uids == [email.uid]

    def test_mixed_inbox_only_relevant_reach_whatsapp(
        self, store: StateStore
    ) -> None:
        relevant = _make_email(uid="1", subject="Contrato urgente")
        irrelevant = _make_email(uid="2", subject="Promoção")
        orch, _, _, smr, wa = _build_orchestrator(
            emails=[relevant, irrelevant],
            store=store,
            classifier=FakeClassifier({
                relevant.message_id: Classification(
                    relevance=True, priority=Priority.high, reason="contrato"
                ),
                irrelevant.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="promo"
                ),
            }),
        )
        stats = orch.run()

        assert smr.calls == [relevant.message_id]
        assert wa.send_calls == 1, "only the relevant email gets its own send()"
        # The irrelevant email's subject must NOT appear in any sent body.
        joined = "\n".join(wa.bodies)
        assert "Contrato urgente" in joined
        assert "Promoção" not in joined
        assert stats.delivered == 1
        assert stats.skipped_irrelevant == 1


# =========================================================================== #
# 3. Dedup short-circuit
# =========================================================================== #


class TestDedup:
    def test_already_processed_skips_classifier_summarizer_and_whatsapp(
        self, store: StateStore
    ) -> None:
        """The dedup row is what guarantees no double-summarization across
        runs. Once `has_been_processed=True`, NOTHING further may run."""
        email = _make_email(uid="42")
        # Pre-seed the DB as if a prior run had already handled this email.
        store.mark_as_processed(
            message_id=email.message_id,
            uid=email.uid,
            relevance=True,
            priority=Priority.high.value,
            delivery_status=DeliveryStatus.delivered,
            twilio_sids=["SM" + "z" * 32],
        )

        orch, ec, clf, smr, wa = _build_orchestrator(
            emails=[email], store=store
        )
        stats = orch.run()

        assert clf.calls == [], "classifier must NOT run for already-processed emails"
        assert smr.calls == []
        assert wa.send_calls == 0
        assert ec.seen_uids == [], "we never re-mark emails we already finished"
        assert stats.skipped_already_processed == 1


# =========================================================================== #
# 4. Per-email error isolation — failures must NEVER kill the loop
# =========================================================================== #


class TestPerEmailErrorIsolation:
    def test_classifier_exception_on_one_email_does_not_block_others(
        self, store: StateStore
    ) -> None:
        a = _make_email(uid="1", subject="Email A")
        b = _make_email(uid="2", subject="Email B - vai quebrar")
        c = _make_email(uid="3", subject="Email C")

        orch, ec, _, smr, wa = _build_orchestrator(
            emails=[a, b, c],
            store=store,
            classifier=FakeClassifier(
                raises_for={b.message_id: RuntimeError("classifier exploded")},
            ),
        )
        stats = orch.run()

        # A and C still summarized + delivered (one send() each).
        assert sorted(smr.calls) == sorted([a.message_id, c.message_id])
        assert wa.send_calls == 2
        # B left untouched: no DB row, not marked seen.
        assert store.get(b.message_id) is None
        assert b.uid not in ec.seen_uids
        # A and C were marked seen.
        assert sorted(ec.seen_uids) == [a.uid, c.uid]
        assert stats.delivered == 2
        assert stats.failed == 1

    def test_summarizer_failure_on_one_email_does_not_block_others(
        self, store: StateStore
    ) -> None:
        a = _make_email(uid="1", subject="Email A")
        b = _make_email(uid="2", subject="Email B")
        c = _make_email(uid="3", subject="Email C")

        orch, ec, _, smr, wa = _build_orchestrator(
            emails=[a, b, c],
            store=store,
            summarizer=FakeSummarizer(
                raises_for={b.message_id: SummaryFailure("model returned junk")},
            ),
        )
        stats = orch.run()

        assert wa.send_calls == 2, "only A and C reach summarize+send; B fails before"
        # B left for retry.
        assert store.get(b.message_id) is None
        assert b.uid not in ec.seen_uids
        # A and C delivered.
        assert sorted(ec.seen_uids) == [a.uid, c.uid]
        assert stats.delivered == 2
        assert stats.failed == 1

    def test_summarizer_unexpected_exception_is_caught(
        self, store: StateStore
    ) -> None:
        """Anything (not just SummaryFailure) coming out of the summarizer
        must be guarded so the loop survives. We test with a plain
        ValueError — orchestrator must catch and continue."""
        a = _make_email(uid="1")
        b = _make_email(uid="2")

        orch, ec, _, _, wa = _build_orchestrator(
            emails=[a, b],
            store=store,
            summarizer=FakeSummarizer(
                raises_for={a.message_id: ValueError("totally unexpected")},
            ),
        )
        stats = orch.run()

        assert stats.failed == 1
        assert stats.delivered == 1
        assert ec.seen_uids == [b.uid]
        assert wa.send_calls == 1


# =========================================================================== #
# 5. Per-email delivery — one send() per email, IMAP arrival order preserved
# =========================================================================== #


class TestPerEmailDelivery:
    """Previously the orchestrator batched summaries into a single digest
    before handing the concatenation to Twilio — which tripped Twilio error
    21617 (body exceeds 1600 chars) as soon as more than one relevant email
    landed in a run. The refactor dropped the digest entirely: each relevant
    email goes out as its own WhatsApp message, in the order it was
    processed (IMAP arrival order)."""

    def test_each_relevant_email_triggers_its_own_send(
        self, store: StateStore
    ) -> None:
        a = _make_email(uid="1", subject="Primeiro")
        b = _make_email(uid="2", subject="Segundo")
        c = _make_email(uid="3", subject="Terceiro")

        orch, _, _, _, wa = _build_orchestrator(
            emails=[a, b, c], store=store,
        )
        orch.run()

        assert wa.send_calls == 3, "one send() per relevant email — no batching"
        # Each body contains exactly ONE email's subject (no concatenation).
        assert "Primeiro" in wa.bodies[0] and "Segundo" not in wa.bodies[0]
        assert "Segundo" in wa.bodies[1] and "Terceiro" not in wa.bodies[1]
        assert "Terceiro" in wa.bodies[2] and "Primeiro" not in wa.bodies[2]

    def test_sends_follow_imap_arrival_order(
        self, store: StateStore
    ) -> None:
        """Priority-based reordering was a property of the (now removed)
        digest. With immediate per-email sends we deliver in the order IMAP
        handed the emails to us — even when that mixes priority levels."""
        low = _make_email(uid="1", subject="LOW one")
        high = _make_email(uid="2", subject="HIGH one")
        medium = _make_email(uid="3", subject="MEDIUM one")

        orch, _, _, _, wa = _build_orchestrator(
            emails=[low, high, medium],
            store=store,
            classifier=FakeClassifier({
                low.message_id:    Classification(relevance=True, priority=Priority.low,    reason="-"),
                high.message_id:   Classification(relevance=True, priority=Priority.high,   reason="-"),
                medium.message_id: Classification(relevance=True, priority=Priority.medium, reason="-"),
            }),
        )
        orch.run()

        assert wa.send_calls == 3
        assert "LOW one" in wa.bodies[0]
        assert "HIGH one" in wa.bodies[1]
        assert "MEDIUM one" in wa.bodies[2]

    def test_each_sid_is_attached_only_to_its_own_db_row(
        self, store: StateStore
    ) -> None:
        """Before the refactor every delivered row carried the FULL csv of
        sids of the batch (audit trail for 'which Twilio message carried
        email X?'). With one send per email the relationship is now 1:1 —
        each row gets exactly its own sid and nothing else."""
        a = _make_email(uid="1")
        b = _make_email(uid="2")

        orch, *_ = _build_orchestrator(emails=[a, b], store=store)
        orch.run()

        row_a = store.get(a.message_id)
        row_b = store.get(b.message_id)
        assert row_a is not None and row_b is not None
        assert row_a.twilio_sids is not None and row_b.twilio_sids is not None
        # Stored as comma-joined csv; a single sid has zero commas.
        assert "," not in row_a.twilio_sids
        assert "," not in row_b.twilio_sids
        assert row_a.twilio_sids != row_b.twilio_sids


# =========================================================================== #
# 6. WhatsApp delivery failure — nothing persisted, nothing marked seen
# =========================================================================== #


class TestWhatsAppDeliveryFailure:
    def test_total_twilio_outage_fails_every_email_individually(
        self, store: StateStore
    ) -> None:
        """With per-email delivery, a Twilio outage still fails every email
        — but each one is tried on its own, so `send_calls` matches the
        number of relevant emails. None of them get persisted or marked
        seen; the next run re-fetches and retries each independently."""
        a = _make_email(uid="1")
        b = _make_email(uid="2")

        orch, ec, _, _, wa = _build_orchestrator(
            emails=[a, b],
            store=store,
            whatsapp=FakeWhatsApp(raise_on_send=WhatsAppDeliveryError("Twilio 503")),
        )
        stats = orch.run()

        assert wa.send_calls == 2, "each email gets its own send() attempt"
        assert store.get(a.message_id) is None
        assert store.get(b.message_id) is None
        assert ec.seen_uids == []
        assert stats.failed == 2
        assert stats.delivered == 0

    def test_single_email_failure_does_not_affect_siblings(
        self, store: StateStore
    ) -> None:
        """THE key property the digest refactor enables: one email's Twilio
        rejection (e.g. a one-off 21617 from a freakishly long summary)
        must NOT poison the rest of the run. Previously, the whole batch
        failed together."""
        a = _make_email(uid="1", subject="Vai passar")
        bad = _make_email(uid="2", subject="Explode no Twilio")
        c = _make_email(uid="3", subject="Tambem passa")

        orch, ec, _, _, wa = _build_orchestrator(
            emails=[a, bad, c],
            store=store,
            whatsapp=FakeWhatsApp(
                raise_for_body_contains={
                    "Explode no Twilio": WhatsAppDeliveryError("21617: too long"),
                },
            ),
        )
        stats = orch.run()

        assert wa.send_calls == 3, "we tried to send all three; only one failed"
        # A and C are sealed; bad is left for retry.
        assert store.get(a.message_id) is not None
        assert store.get(c.message_id) is not None
        assert store.get(bad.message_id) is None
        assert sorted(ec.seen_uids) == [a.uid, c.uid]
        assert stats.delivered == 2
        assert stats.failed == 1

    def test_irrelevant_emails_unaffected_when_relevant_send_fails(
        self, store: StateStore
    ) -> None:
        """The token-saver path is fully decoupled from delivery — an
        irrelevant email gets persisted + sealed BEFORE the digest is sent,
        so a later WhatsApp failure must not retroactively undo it."""
        skip = _make_email(uid="1", subject="Newsletter")
        send = _make_email(uid="2", subject="Importante")

        orch, ec, *_ = _build_orchestrator(
            emails=[skip, send],
            store=store,
            classifier=FakeClassifier({
                skip.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="newsletter"
                ),
                send.message_id: Classification(
                    relevance=True, priority=Priority.high, reason="-"
                ),
            }),
            whatsapp=FakeWhatsApp(raise_on_send=WhatsAppDeliveryError("boom")),
        )
        stats = orch.run()

        # Irrelevant email: sealed.
        assert store.get(skip.message_id) is not None
        assert skip.uid in ec.seen_uids
        assert stats.skipped_irrelevant == 1
        # Relevant email: NOT sealed (delivery failed).
        assert store.get(send.message_id) is None
        assert send.uid not in ec.seen_uids
        assert stats.failed == 1


# =========================================================================== #
# 7. Idempotency contract — DB row precedes IMAP \Seen
# =========================================================================== #


class TestIdempotencyOrder:
    def test_db_persist_happens_before_mark_as_seen_for_every_email(
        self, store: StateStore
    ) -> None:
        """A crash between persist and mark_as_seen must be safe: row in DB,
        email still unread on IMAP. If the order ever inverts, a crash
        between the two would silently drop the email forever (IMAP
        flagged seen, DB has no record)."""
        recorder = _CallRecorder()

        # Wrap mark_as_processed so it logs into the SAME recorder.
        original_persist = store.mark_as_processed

        def wrapped_persist(**kwargs: Any) -> Any:
            recorder.record("db", "mark_as_processed", kwargs["uid"])
            return original_persist(**kwargs)

        store.mark_as_processed = wrapped_persist  # type: ignore[method-assign]

        a = _make_email(uid="1")
        b = _make_email(uid="2")
        orch, *_ = _build_orchestrator(
            emails=[a, b], store=store, recorder=recorder,
        )
        orch.run()

        # Per UID, the DB record must come BEFORE the IMAP seen record.
        for uid in (a.uid, b.uid):
            db_idx = next(i for i, e in enumerate(recorder.events)
                          if e == ("db", "mark_as_processed", uid))
            imap_idx = next(i for i, e in enumerate(recorder.events)
                            if e == ("imap", "mark_as_seen", uid))
            assert db_idx < imap_idx, (
                f"For uid={uid}, DB write at idx {db_idx} must precede "
                f"IMAP \\Seen at idx {imap_idx}. Events: {recorder.events}"
            )


# =========================================================================== #
# 8. Storage fail-stop — StorageError must propagate and halt the run
# =========================================================================== #


class TestStorageFailStop:
    def test_storage_error_on_irrelevant_persist_propagates(
        self, store: StateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        email = _make_email(uid="1")
        monkeypatch.setattr(
            store, "mark_as_processed",
            lambda *a, **kw: (_ for _ in ()).throw(StorageError("DB down")),
        )
        orch, *_ = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier({
                email.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="irrelevant"
                ),
            }),
        )

        with pytest.raises(StorageError, match="DB down"):
            orch.run()

    def test_storage_error_mid_batch_halts_the_run(
        self, store: StateStore
    ) -> None:
        """If the DB dies between two delivered-email seals, we must abort
        immediately. Sealed emails are safe (DB row exists); unsealed ones
        will retry next run."""
        a = _make_email(uid="1")
        b = _make_email(uid="2")

        original = store.mark_as_processed
        call_count = {"n": 0}

        def fail_on_second(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise StorageError("DB lost connection")
            return original(*args, **kwargs)

        store.mark_as_processed = fail_on_second  # type: ignore[method-assign]

        orch, *_ = _build_orchestrator(emails=[a, b], store=store)
        with pytest.raises(StorageError, match="lost connection"):
            orch.run()


# =========================================================================== #
# 9. IMAP seal failure — best-effort, must not crash or fail-count
# =========================================================================== #


class TestIMAPSealFailure:
    def test_mark_as_seen_failure_after_db_write_is_logged_not_raised(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        """If IMAP `\\Seen` fails AFTER the DB row is committed, we MUST
        survive: dedup will protect the next run from a duplicate. The
        run_stats.delivered count still increments (the email did reach
        the recipient)."""
        email = _make_email(uid="1")
        orch, *_, wa = _build_orchestrator(
            emails=[email],
            store=store,
            mark_seen_failures={email.uid: RuntimeError("IMAP timeout")},
        )

        stats = orch.run()  # must NOT raise

        assert stats.delivered == 1
        # DB row exists, so the next run would correctly skip via dedup.
        assert store.get(email.message_id) is not None
        # An [ERROR] line surfaced the IMAP failure for the operator.
        msgs = _messages(loguru_capture)
        assert any("[ERROR]" in m and "mark IMAP" in m for m in msgs), (
            "must log an [ERROR] when mark_as_seen fails after a successful send"
        )

    def test_mark_as_seen_failure_on_irrelevant_email_is_tolerated(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        email = _make_email(uid="9")
        orch, *_ = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier({
                email.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="newsletter"
                ),
            }),
            mark_seen_failures={email.uid: RuntimeError("IMAP died")},
        )
        stats = orch.run()

        assert stats.skipped_irrelevant == 1  # still counts as a clean skip
        assert store.get(email.message_id) is not None
        msgs = _messages(loguru_capture)
        assert any("[ERROR]" in m and "mark IMAP" in m for m in msgs)


# =========================================================================== #
# 10. Log tag vocabulary — operator-facing grep contract
# =========================================================================== #


class TestLogTags:
    """Each test pins ONE tag to the scenario that should emit it. If we
    ever rename a tag (e.g. `[FETCH]` → `[FETCHED]`) every alert / dashboard
    that greps the production log breaks — these tests exist so that change
    is INTENTIONAL, not accidental."""

    def test_fetch_tag_emitted_with_email_count(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        a = _make_email(uid="1")
        orch, *_ = _build_orchestrator(emails=[a], store=store)
        orch.run()
        msgs = _messages(loguru_capture)
        assert any(m.startswith("[FETCH]") and "1 unread" in m for m in msgs)

    def test_skip_tag_emitted_for_already_processed(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        email = _make_email(uid="1")
        store.mark_as_processed(
            message_id=email.message_id,
            uid=email.uid,
            relevance=True,
            priority=Priority.high.value,
            delivery_status=DeliveryStatus.delivered,
        )
        orch, *_ = _build_orchestrator(emails=[email], store=store)
        orch.run()
        msgs = _messages(loguru_capture)
        assert any("[SKIP]" in m and "already processed" in m for m in msgs)

    def test_classify_tag_emitted_with_verdict_fields(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        email = _make_email(uid="1")
        orch, *_ = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier({
                email.message_id: Classification(
                    relevance=True, priority=Priority.high, reason="prazo"
                ),
            }),
        )
        orch.run()
        msgs = _messages(loguru_capture)
        line = next(m for m in msgs if "[CLASSIFY]" in m)
        assert "relevance=True" in line
        assert "priority=high" in line
        assert "prazo" in line

    def test_irrelevant_tag_emitted_for_rejected_email(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        email = _make_email(uid="1")
        orch, *_ = _build_orchestrator(
            emails=[email],
            store=store,
            classifier=FakeClassifier({
                email.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="newsletter"
                ),
            }),
        )
        orch.run()
        msgs = _messages(loguru_capture)
        assert any(
            "[IRRELEVANT]" in m and "newsletter" in m and email.message_id in m
            for m in msgs
        )

    def test_summarize_tag_emitted_after_summary_built(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        email = _make_email(uid="1", subject="Renovação")
        orch, *_ = _build_orchestrator(emails=[email], store=store)
        orch.run()
        msgs = _messages(loguru_capture)
        assert any("[SUMMARIZE]" in m and "Renovação" in m for m in msgs)

    def test_send_tag_emitted_once_per_delivered_email_with_sid(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        a = _make_email(uid="1")
        b = _make_email(uid="2")
        orch, *_ = _build_orchestrator(emails=[a, b], store=store)
        orch.run()
        msgs = _messages(loguru_capture)
        send_lines = [m for m in msgs if "[SEND]" in m]
        assert len(send_lines) == 2, "one [SEND] per delivered email (no batching)"
        for line in send_lines:
            assert "uid=" in line
            assert "sid=SM" in line

    def test_done_tag_emitted_per_persisted_email(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        a = _make_email(uid="1")
        b = _make_email(uid="2")
        orch, *_ = _build_orchestrator(emails=[a, b], store=store)
        orch.run()
        msgs = _messages(loguru_capture)
        done_lines = [m for m in msgs if m.startswith("[DONE]")]
        assert len(done_lines) == 2
        assert all("status=delivered" in m for m in done_lines)

    def test_error_tag_emitted_when_summarizer_fails(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        bad = _make_email(uid="1")
        orch, *_ = _build_orchestrator(
            emails=[bad],
            store=store,
            summarizer=FakeSummarizer(
                raises_for={bad.message_id: SummaryFailure("model returned empty")},
            ),
        )
        orch.run()
        msgs = _messages(loguru_capture)
        assert any(
            "[ERROR]" in m and "summarizer failed" in m and "leaving unmarked" in m
            for m in msgs
        )

    def test_run_tag_carries_aggregated_stats(
        self, store: StateStore, loguru_capture: list[dict[str, Any]]
    ) -> None:
        a = _make_email(uid="1")
        irrelevant = _make_email(uid="2")
        orch, *_ = _build_orchestrator(
            emails=[a, irrelevant],
            store=store,
            classifier=FakeClassifier({
                irrelevant.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="r"
                ),
            }),
        )
        orch.run()
        msgs = _messages(loguru_capture)
        line = next(m for m in msgs if m.startswith("[RUN]"))
        assert "fetched" in line
        assert "delivered" in line
        assert "irrelevant" in line


# =========================================================================== #
# 11. RunStats — aggregated outcome contract
# =========================================================================== #


class TestRunStats:
    def test_counts_are_non_overlapping_for_mixed_inbox(
        self, store: StateStore
    ) -> None:
        delivered = _make_email(uid="1")
        irrelevant = _make_email(uid="2")
        already = _make_email(uid="3")
        failed = _make_email(uid="4")

        # Pre-seed the "already processed" one.
        store.mark_as_processed(
            message_id=already.message_id,
            uid=already.uid,
            relevance=True,
            priority=Priority.high.value,
            delivery_status=DeliveryStatus.delivered,
        )

        orch, *_ = _build_orchestrator(
            emails=[delivered, irrelevant, already, failed],
            store=store,
            classifier=FakeClassifier({
                irrelevant.message_id: Classification(
                    relevance=False, priority=Priority.low, reason="r"
                ),
            }),
            summarizer=FakeSummarizer(
                raises_for={failed.message_id: SummaryFailure("nope")},
            ),
        )
        stats = orch.run()

        assert stats.fetched == 4
        assert stats.delivered == 1
        assert stats.skipped_irrelevant == 1
        assert stats.skipped_already_processed == 1
        assert stats.failed == 1
        # Sanity: every fetched email is accounted for exactly once.
        accounted = (
            stats.delivered
            + stats.skipped_irrelevant
            + stats.skipped_already_processed
            + stats.failed
        )
        assert accounted == stats.fetched

    def test_delivered_message_sids_are_collected(
        self, store: StateStore
    ) -> None:
        a = _make_email(uid="1")
        b = _make_email(uid="2")
        orch, *_ = _build_orchestrator(emails=[a, b], store=store)
        stats = orch.run()
        # One sid per delivered email — individual sends, no shared batch.
        assert len(stats.delivered_message_sids) == 2
        assert all(sid.startswith("SM") for sid in stats.delivered_message_sids)

    def test_processed_this_run_sums_terminal_buckets(self) -> None:
        """`processed_this_run` is the convenience callers (CLI / scheduler)
        will use to decide if anything terminal happened. It must EXCLUDE
        the dedup skip count (those weren't processed this run, they were
        processed in a prior run)."""
        stats = RunStats(
            fetched=5,
            skipped_already_processed=2,
            skipped_irrelevant=1,
            delivered=2,
            failed=0,
        )
        assert stats.processed_this_run == 3  # 1 irrelevant + 2 delivered


# =========================================================================== #
# 12. Real-collaborator assertions — sanity that the orchestrator's contract
#     matches the production class signatures (not just our fakes).
# =========================================================================== #


class TestRealCollaboratorContract:
    """If we ever drift the real classifier / summarizer / whatsapp APIs,
    the fakes above silently keep working. These tests pin that the real
    classes still accept the same constructor arguments the orchestrator
    expects so the production wiring path remains valid."""

    def test_orchestrator_accepts_real_concrete_collaborators(
        self, store: StateStore
    ) -> None:
        # We don't `run()` here — that would hit the network. We just
        # construct, which is the failure mode that would break wiring.
        orch = Orchestrator(
            email_client=FakeEmailClient([]),
            classifier=EmailClassifier(llm=_NullLLM()),  # type: ignore[arg-type]
            summarizer=EmailSummarizer(llm=_NullLLM()),  # type: ignore[arg-type]
            whatsapp=WhatsAppClient(
                account_sid="ACxxx",
                auth_token="fake",
                whatsapp_from="whatsapp:+14155238886",
                whatsapp_to="whatsapp:+5511999999999",
                client=_NullTwilio(),  # type: ignore[arg-type]
            ),
            state=store,
        )
        # An empty-inbox run must work end-to-end with the real classes.
        stats = orch.run()
        assert stats.fetched == 0


# Two trivial null-objects so we can construct the real collaborators
# without touching a single network library. We never call into them.
class _NullLLM:
    def complete(self, *a: Any, **kw: Any) -> Any:
        raise AssertionError("must not be called in this test")


class _NullTwilio:
    pass

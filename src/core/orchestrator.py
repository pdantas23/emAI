r"""End-to-end pipeline coordinator — fetch → classify → summarize → deliver → persist.

This module is the **conductor**. It owns no business logic of its own:
its only job is to call the specialized layers in the correct order, in the
correct conditions, and with the correct error semantics.

----------------------------------------------------------------------------
Pipeline (per run)
----------------------------------------------------------------------------

    1. EmailClient.fetch_unread()                    [FETCH]
    2. for each email:
        a. StateStore.has_been_processed()           [SKIP if True]
        b. EmailClassifier.classify()
            • relevance=False ──► persist as `skipped_irrelevant`,
                                  mark IMAP `\Seen`, NEVER call summarizer
                                  (token-saver gate)
            • relevance=True  ──► summarize → format → send → persist → mark seen
    3. There is no batch/digest phase. Each relevant email is its own
       standalone WhatsApp message, sent immediately after summarization so
       Evolution never sees a concatenated body (which previously tripped the
       1600-char error 21617 once more than one email landed in a run).

----------------------------------------------------------------------------
Error policy (per email)
----------------------------------------------------------------------------

A failure in IA / messaging for one email **never breaks the loop**: the
exception is logged, the email is left UNTOUCHED on IMAP and absent from the
DB, and the next pipeline run will retry it. This is the symmetric counterpart
to the storage layer's fail-stop policy: storage problems halt the run
(`StorageError` propagates), per-email problems are isolated.

A `WhatsAppDeliveryError` on one email therefore only fails THAT email —
previous emails in the same run are already delivered and sealed, and
subsequent emails still get their own send attempts.

----------------------------------------------------------------------------
Idempotency contract (the load-bearing invariant)
----------------------------------------------------------------------------

For each email we ALWAYS commit the DB row BEFORE marking the message as
`\Seen` on the server. If we crash between the two, the next run sees the
email as still unread on IMAP, but `has_been_processed()` returns True and
we politely skip it (logging `[SKIP]`). The recipient never gets a duplicate
WhatsApp briefing.

----------------------------------------------------------------------------
Logging contract (operator-facing)
----------------------------------------------------------------------------

The pipeline emits a fixed vocabulary of bracket-tagged INFO lines so an
operator can grep the live log in a war room:

    [FETCH] 5 unread emails returned by IMAP (folder='INBOX')
    [SKIP]  uid=42 message_id=<...> already processed — moving on
    [CLASSIFY] uid=42 → relevance=True priority=high reason='prazo curto'
    [IRRELEVANT] uid=51 message_id=<...> classifier rejected: 'newsletter'
    [SUMMARIZE] uid=42 produced summary for 'Renovação de contrato'
    [SEND]  uid=42 delivered (sid=SMxxx)
    [DONE]  uid=42 persisted (id=17, status=delivered)
    [ERROR] uid=42 summarizer failed: SummaryFailure('...') — leaving unmarked
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ai.classifier import Classification, EmailClassifier
from src.ai.summarizer import (
    EmailSummarizer,
    EmailSummary,
    SummaryFailure,
    format_for_whatsapp,
)
from src.email_client.base import EmailClient, RawEmail
from src.messaging.evolution_api import EvolutionClient, WhatsAppDeliveryError
from src.storage.models import DeliveryStatus
from src.storage.state import StateStore, StorageError
from src.utils.logger import log


# --------------------------------------------------------------------------- #
# Run summary — what we report back to the caller / scheduler.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RunStats:
    """Aggregated outcome of a single orchestrator pass.

    The scheduler / CLI uses this to decide exit code and emit one final
    summary line. Counts are non-overlapping: an email lands in exactly one
    of the buckets below (excluding `fetched`, which is the grand total).
    """

    fetched: int = 0
    skipped_already_processed: int = 0
    skipped_irrelevant: int = 0
    delivered: int = 0
    failed: int = 0
    delivered_message_sids: list[str] = field(default_factory=list)

    @property
    def processed_this_run(self) -> int:
        """Emails that made it through to a terminal DB row this run."""
        return self.skipped_irrelevant + self.delivered

    def as_log_dict(self) -> dict[str, int]:
        """Compact one-liner payload for the final [RUN] INFO log."""
        return {
            "fetched": self.fetched,
            "already_processed": self.skipped_already_processed,
            "irrelevant": self.skipped_irrelevant,
            "delivered": self.delivered,
            "failed": self.failed,
        }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class Orchestrator:
    """Wires every collaborator into the pipeline. Fully dependency-injected.

    The constructor accepts ready-made instances (preferred path for tests
    and for callers that already wired settings) or builds defaults. None
    of the collaborators are constructed lazily inside `run()` — that would
    make per-call costs surprising and hide construction errors behind the
    first business operation.
    """

    def __init__(
        self,
        *,
        email_client: EmailClient,
        classifier: EmailClassifier,
        summarizer: EmailSummarizer,
        whatsapp: EvolutionClient,
        state: StateStore,
    ) -> None:
        self._email = email_client
        self._classifier = classifier
        self._summarizer = summarizer
        self._whatsapp = whatsapp
        self._state = state

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> RunStats:
        """Execute one full pipeline pass and return the aggregated stats.

        The IMAP connection is opened/closed via the context manager so a
        crash anywhere in the loop still releases the socket. Storage
        errors are NOT caught here — they're fatal by design (see
        `StateStore` docstring).
        """
        stats = RunStats()

        with self._email as mailbox:
            unread = list(mailbox.fetch_unread())
            stats.fetched = len(unread)
            log.info("[FETCH] {} unread email(s) returned by IMAP", stats.fetched)

            for email in unread:
                outcome = self._process_one(email)
                self._tally(outcome, stats)

        log.info("[RUN] pipeline finished — {}", stats.as_log_dict())
        return stats

    # ------------------------------------------------------------------ #
    # Per-email pipeline
    # ------------------------------------------------------------------ #

    def _process_one(self, email: RawEmail) -> "_EmailOutcome":
        """Classify + (optionally) summarize a single email.

        Per-email errors are logged and converted to `_EmailOutcome.failed`
        so the caller never sees a raw exception bubble up — that would
        break the for-loop and starve the rest of the unread queue.

        Returns a tagged outcome describing what should happen next:
          • `_EmailOutcome.already_processed` — already in DB, no work
          • `_EmailOutcome.irrelevant`        — persist + mark seen now
          • `_EmailOutcome.ready_to_send`     — send immediately, then seal
          • `_EmailOutcome.failed`            — log only; leave untouched

        Storage errors are NOT caught here (fail-stop bubbles to `run()`).
        """
        # ---- Dedup gate ----
        if self._state.has_been_processed(email.message_id):
            log.info(
                "[SKIP] uid={} message_id={} already processed — moving on",
                email.uid, email.message_id,
            )
            return _EmailOutcome(kind="already_processed")

        # ---- Classifier (cheap haiku gate) ----
        try:
            classification = self._classifier.classify(email)
        except Exception as exc:  # noqa: BLE001 — guard the loop
            log.error(
                "[ERROR] uid={} classifier raised unexpectedly ({}): {} "
                "— leaving unmarked for next run",
                email.uid, type(exc).__name__, exc,
            )
            return _EmailOutcome(kind="failed")

        log.info(
            "[CLASSIFY] uid={} → relevance={} priority={} reason={!r}",
            email.uid,
            classification.relevance,
            classification.priority.value,
            classification.reason,
        )

        # ---- Token-saver gate: irrelevant short-circuits ----
        if not classification.relevance:
            log.info(
                "[IRRELEVANT] uid={} message_id={} classifier rejected: {!r}",
                email.uid, email.message_id, classification.reason,
            )
            return _EmailOutcome(
                kind="irrelevant",
                email=email,
                classification=classification,
            )

        # ---- Summarizer (expensive Sonnet call) ----
        try:
            summary = self._summarizer.summarize(
                email, priority=classification.priority
            )
        except SummaryFailure as exc:
            log.error(
                "[ERROR] uid={} summarizer failed: {} — leaving unmarked for retry",
                email.uid, exc,
            )
            return _EmailOutcome(kind="failed")
        except Exception as exc:  # noqa: BLE001 — guard the loop on unexpected
            log.error(
                "[ERROR] uid={} summarizer raised unexpectedly ({}): {} "
                "— leaving unmarked",
                email.uid, type(exc).__name__, exc,
            )
            return _EmailOutcome(kind="failed")

        log.info(
            "[SUMMARIZE] uid={} produced summary for {!r}",
            email.uid, email.subject or "(sem assunto)",
        )

        rendered = format_for_whatsapp(email, summary, classification)
        return _EmailOutcome(
            kind="ready_to_send",
            email=email,
            classification=classification,
            summary=summary,
            rendered=rendered,
        )

    # ------------------------------------------------------------------ #
    # Per-outcome dispatch — keeps `run()` flat and readable
    # ------------------------------------------------------------------ #

    def _tally(
        self,
        outcome: "_EmailOutcome",
        stats: RunStats,
    ) -> None:
        r"""Apply the per-email outcome to stats and execute side effects.

        Irrelevant and ready-to-send emails both seal here (persist + mark
        seen), but ready-to-send first issues its own standalone WhatsApp
        `send()`. There is no batching: every relevant email travels to
        Evolution on its own, which keeps each body well under the 1600-char
        concatenated-body limit that Evolution enforces (error 21617).
        """
        if outcome.kind == "already_processed":
            stats.skipped_already_processed += 1
            return

        if outcome.kind == "failed":
            stats.failed += 1
            return

        if outcome.kind == "irrelevant":
            assert outcome.email is not None and outcome.classification is not None
            try:
                self._persist_and_mark_seen(
                    email=outcome.email,
                    classification=outcome.classification,
                    delivery_status=DeliveryStatus.skipped_irrelevant,
                    message_ids=None,
                )
            except StorageError:
                # Storage is fail-stop — let it propagate to run()/caller.
                raise
            except Exception as exc:  # noqa: BLE001 — IMAP mark_as_seen
                log.error(
                    "[ERROR] uid={} could not mark IMAP \\Seen after persist: "
                    "{}: {} — DB row exists, will be skipped on next run",
                    outcome.email.uid, type(exc).__name__, exc,
                )
            stats.skipped_irrelevant += 1
            return

        if outcome.kind == "ready_to_send":
            assert (
                outcome.email is not None
                and outcome.classification is not None
                and outcome.summary is not None
                and outcome.rendered is not None
            )
            self._deliver_one(outcome, stats)
            return

    # ------------------------------------------------------------------ #
    # Single-email delivery — one send() per email, immediately sealed
    # ------------------------------------------------------------------ #

    def _deliver_one(
        self,
        outcome: "_EmailOutcome",
        stats: RunStats,
    ) -> None:
        r"""Send one email's WhatsApp briefing, then persist + mark seen.

        On success: DB row goes in BEFORE the IMAP `\Seen` flag (idempotency
        invariant). On `WhatsAppDeliveryError`: nothing is persisted and
        nothing is marked seen, so the next run retries only this email —
        emails already delivered earlier in the SAME run are unaffected.
        """
        assert outcome.email is not None
        assert outcome.classification is not None
        assert outcome.summary is not None
        assert outcome.rendered is not None

        email = outcome.email

        try:
            sid = self._whatsapp.send(outcome.rendered)
        except WhatsAppDeliveryError as exc:
            log.error(
                "[ERROR] uid={} WhatsApp delivery failed: {} "
                "— leaving unmarked for next run",
                email.uid, exc,
            )
            stats.failed += 1
            return

        log.info("[SEND] uid={} delivered (sid={})", email.uid, sid)
        stats.delivered_message_sids.append(sid)

        # The summary stayed in memory only long enough to render the
        # WhatsApp body above; the state store never sees it.
        try:
            self._persist_and_mark_seen(
                email=email,
                classification=outcome.classification,
                delivery_status=DeliveryStatus.delivered,
                message_ids=[sid],
            )
        except StorageError:
            # Catastrophic — abort the run; this email reached WhatsApp but
            # we couldn't record it. Fail-stop so the operator notices.
            raise
        except Exception as exc:  # noqa: BLE001 — IMAP mark_as_seen
            log.error(
                "[ERROR] uid={} delivered but could not mark IMAP \\Seen "
                "({}: {}) — DB row exists; next run will skip via dedup",
                email.uid, type(exc).__name__, exc,
            )
        stats.delivered += 1

    # ------------------------------------------------------------------ #
    # The two-step seal: DB first, IMAP second.
    # ------------------------------------------------------------------ #

    def _persist_and_mark_seen(
        self,
        *,
        email: RawEmail,
        classification: Classification,
        delivery_status: DeliveryStatus,
        message_ids: list[str] | None,
    ) -> None:
        r"""Commit the DB row, THEN mark the IMAP message as `\Seen`.

        Order matters: a row written without the IMAP flag is recoverable
        (next run sees it as unread but `has_been_processed=True` ⇒ skip).
        The reverse — IMAP flagged but no DB row — would silently drop the
        email forever. The only way to honor that contract is DB-first.

        **Privacy note.** We hand the state store only the non-sensitive
        metadata it is allowed to retain (Message-ID, UID, relevance flag,
        priority value, delivery status, Evolution SIDs). Everything else we
        received — subject, sender, classification reason, summary body —
        stays in this function's memory and is garbage-collected as soon
        as it returns. The state store's signature enforces this too.

        Raises:
            StorageError: re-raised from the StateStore call. This is
                fatal by design — the caller (`run`) propagates it.
        """
        record = self._state.mark_as_processed(
            message_id=email.message_id,
            uid=email.uid,
            relevance=classification.relevance,
            priority=classification.priority.value,
            delivery_status=delivery_status,
            message_ids=message_ids,
        )
        log.info(
            "[DONE] uid={} persisted (id={}, status={})",
            email.uid, record.id, delivery_status.value,
        )
        # mark_as_seen is best-effort: a failure here is logged by the
        # caller, not raised. The DB dedup row already protects us from
        # double-processing on the next run.
        self._email.mark_as_seen(email.uid)


# --------------------------------------------------------------------------- #
# Internal value objects — keep the orchestrator's interfaces precise
# without leaking transient state into the public API surface.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _EmailOutcome:
    """Tagged union describing what `_process_one` decided about an email.

    Using a small dataclass instead of a `dict` lets mypy keep us honest
    about which fields are present per outcome kind.
    """

    kind: str  # "already_processed" | "irrelevant" | "ready_to_send" | "failed"
    email: RawEmail | None = None
    classification: Classification | None = None
    summary: EmailSummary | None = None
    rendered: str | None = None

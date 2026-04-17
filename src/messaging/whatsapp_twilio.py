"""WhatsApp delivery via Twilio.

This is the **last mile** of the pipeline. By the time we get here, an email
has already been fetched, classified as relevant, summarized, formatted for
WhatsApp, and chunked to fit Twilio's 4096-char limit. Our only job is to
hand the resulting payloads to Twilio and report success or failure.

The contract with the orchestrator is short and load-bearing:

    try:
        whatsapp.send_many(payloads)
    except WhatsAppDeliveryError:
        # DO NOT mark the email as seen. The next pipeline run will
        # pick it up again and retry the whole thing.
        log.error(...)
        continue

That's the **idempotency guarantee** of the whole system: an email is only
marked `\\Seen` on the IMAP server AFTER its summary has been successfully
delivered. A delivery failure (Twilio outage, our credentials revoked,
network blip on the way out) leaves the source email untouched, so the next
run is a clean retry.

Retry policy mirrors `LLMClient`:
  - 3 attempts, exponential backoff 1s → 2s → 4s
  - Retried: 429 (rate limit), 5xx (server)
  - NOT retried: 4xx other than 429 (auth, bad request, forbidden) — these
    will not get better with more attempts; failing fast surfaces config
    bugs immediately.
"""

from __future__ import annotations

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioClient
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.messaging.formatter import chunk_messages, truncate
from src.utils.logger import log

# --------------------------------------------------------------------------- #
# Public exception
# --------------------------------------------------------------------------- #


class WhatsAppDeliveryError(Exception):
    """Raised when a WhatsApp message could not be delivered.

    The orchestrator MUST treat this as a signal to **leave the source email
    unmarked** so the next run retries the full pipeline for it.
    """


# --------------------------------------------------------------------------- #
# Retry classification — Twilio uses HTTP semantics, so we key off `status`
# --------------------------------------------------------------------------- #

# 429 = rate limit; 500/502/503/504 = transient server problems. Everything
# else is either success (handled) or a permanent client error we shouldn't
# hammer the API about.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient Twilio errors that justify a retry.

    We deliberately do NOT retry on plain `Exception` (e.g. ConnectionError)
    inside this predicate — those bubble up to `send_one()` which converts
    them to `WhatsAppDeliveryError` after a single attempt. If we wanted to
    retry network errors too we'd add `isinstance(exc, ConnectionError)` here;
    keeping it tight for now because Twilio's SDK has its own urllib3-level
    retry on raw connection issues already.
    """
    return (
        isinstance(exc, TwilioRestException)
        and exc.status in _RETRYABLE_STATUSES
    )


def _log_before_sleep(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    sleep_s = state.next_action.sleep if state.next_action else 0.0
    status = getattr(exc, "status", "?")
    log.warning(
        "Twilio send failed (attempt {}, status={}): {} — retrying in {:.1f}s",
        state.attempt_number, status, exc, sleep_s,
    )


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class WhatsAppClient:
    """Thin wrapper around the Twilio REST client for WhatsApp delivery.

    Accepts credentials via constructor for dependency injection. The
    orchestrator builds this from `UserRuntimeConfig`; no global `settings`
    singleton is read.
    """

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        whatsapp_from: str,
        whatsapp_to: str,
        client: TwilioClient | None = None,
    ) -> None:
        self._sender: str = whatsapp_from
        self._recipient: str = whatsapp_to
        self._account_sid: str = account_sid
        self._auth_token: str = auth_token
        # Allow injection for tests; otherwise build lazily on first send.
        self._client: TwilioClient | None = client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def send(self, body: str) -> str:
        """Send a single WhatsApp message. Returns the Twilio message SID.

        Truncates `body` defensively if it exceeds the WhatsApp char cap.
        Callers SHOULD pre-chunk via `messaging.formatter.chunk_messages`,
        but we never trust the caller to be perfect — silently dropping a
        report because someone forgot to chunk would be a much worse bug
        than truncating it.

        Raises:
            WhatsAppDeliveryError: when Twilio rejects the message after
                the retry budget is exhausted, or when an unexpected error
                occurs. The orchestrator treats this as a signal to NOT
                mark the source email as seen.
        """
        safe_body = truncate(body)

        try:
            sid = self._send_with_retry(safe_body)
        except TwilioRestException as exc:
            # Either non-retryable (4xx other than 429) or we exhausted the
            # retry budget on a 5xx/429.
            log.error(
                "Twilio rejected WhatsApp message (status={}, code={}): {}",
                exc.status, exc.code, exc.msg,
            )
            raise WhatsAppDeliveryError(
                f"Twilio error {exc.status} (code={exc.code}): {exc.msg}"
            ) from exc
        except Exception as exc:
            # Network errors, DNS failures, anything outside Twilio's protocol.
            # We treat them all as delivery failures — the orchestrator will
            # leave the email unmarked and the next run will retry.
            log.error(
                "Unexpected error sending WhatsApp message: {}: {}",
                type(exc).__name__, exc,
            )
            raise WhatsAppDeliveryError(
                f"Unexpected delivery error: {type(exc).__name__}: {exc}"
            ) from exc

        log.info("WhatsApp delivered → {} (sid={})", self._recipient, sid)
        return sid

    def send_many(self, bodies: list[str]) -> list[str]:
        """Send a sequence of pre-chunked messages. Fail-fast.

        The first failure aborts the rest. Rationale: from the orchestrator's
        point of view, the source email maps to ONE logical "report" — a
        partial delivery is a worse user experience than no delivery, because
        the recipient sees half a briefing and has no idea something is
        missing. Failing fast lets the orchestrator leave the email unmarked
        and retry the entire payload on the next run.

        Returns the list of message SIDs in the same order as `bodies`.
        """
        sids: list[str] = []
        for i, body in enumerate(bodies, start=1):
            log.debug("Sending WhatsApp chunk {}/{} ({} chars)", i, len(bodies), len(body))
            sids.append(self.send(body))
        return sids

    def send_chunked(self, messages: list[str]) -> list[str]:
        """Convenience: chunk a list of rendered messages then send them.

        Equivalent to `send_many(chunk_messages(messages))` — exposed as one
        call so the orchestrator doesn't need to import the formatter.
        """
        return self.send_many(chunk_messages(messages))

    # ------------------------------------------------------------------ #
    # Internal: retry-decorated single-message dispatch
    # ------------------------------------------------------------------ #

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_before_sleep,
    )
    def _send_with_retry(self, body: str) -> str:
        client = self._twilio_client()
        message = client.messages.create(
            from_=self._sender,
            to=self._recipient,
            body=body,
        )
        return message.sid

    # ------------------------------------------------------------------ #
    # Lazy SDK client construction
    # ------------------------------------------------------------------ #

    def _twilio_client(self) -> TwilioClient:
        if self._client is None:
            self._client = TwilioClient(self._account_sid, self._auth_token)
        return self._client

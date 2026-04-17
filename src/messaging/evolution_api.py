"""WhatsApp delivery via Evolution API.

This is the **last mile** of the pipeline. By the time we get here, an email
has already been fetched, classified as relevant, summarized, and formatted
for WhatsApp. Our only job is to POST the payload to Evolution and report
success or failure.

The contract with the orchestrator is identical to the old Twilio client:

    try:
        messaging.send(rendered_body)
    except WhatsAppDeliveryError:
        # DO NOT mark the email as seen. The next pipeline run will
        # pick it up again and retry the whole thing.
        log.error(...)
        continue

Evolution API does NOT impose the 1600-char / 4096-char limits that Twilio
enforces. We still truncate at a generous 65536-char safety cap (no sane
WhatsApp message is longer), but the practical limit is gone.

Retry policy:
  - 3 attempts, exponential backoff 1s → 2s → 4s
  - Retried: 429 (rate limit), 5xx (server)
  - NOT retried: 4xx other than 429 (auth, bad request)
"""

from __future__ import annotations

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import log

# --------------------------------------------------------------------------- #
# Public exception
# --------------------------------------------------------------------------- #

# Safety cap: no sane WhatsApp message should exceed this.
MAX_MESSAGE_CHARS: int = 65_536


class WhatsAppDeliveryError(Exception):
    """Raised when a WhatsApp message could not be delivered.

    The orchestrator MUST treat this as a signal to **leave the source email
    unmarked** so the next run retries the full pipeline for it.
    """


# --------------------------------------------------------------------------- #
# Retry classification
# --------------------------------------------------------------------------- #

_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class _EvolutionHTTPError(Exception):
    """Internal wrapper to carry the HTTP status for retry classification."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        super().__init__(f"HTTP {status}: {detail}")


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _EvolutionHTTPError) and exc.status in _RETRYABLE_STATUSES


def _log_before_sleep(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    sleep_s = state.next_action.sleep if state.next_action else 0.0
    status = getattr(exc, "status", "?")
    log.warning(
        "Evolution send failed (attempt {}, status={}): {} — retrying in {:.1f}s",
        state.attempt_number, status, exc, sleep_s,
    )


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class EvolutionClient:
    """Thin wrapper around the Evolution API for WhatsApp delivery.

    Accepts credentials via constructor for dependency injection. The
    orchestrator builds this from `UserRuntimeConfig`.

    Evolution API endpoint: POST {base_url}/message/sendText/{instance}
    Headers: apikey: {api_key}
    Body: {"number": "5511999999999", "text": "message body"}
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        instance: str,
        whatsapp_to: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        # Normalize base_url: strip trailing slash
        self._base_url: str = base_url.rstrip("/")
        self._api_key: str = api_key
        self._instance: str = instance
        self._recipient: str = self._normalize_number(whatsapp_to)
        # Allow injection for tests; otherwise build lazily on first send.
        self._client: httpx.Client | None = http_client

    # ------------------------------------------------------------------ #
    # Public API — same interface as the old WhatsAppClient
    # ------------------------------------------------------------------ #

    def send(self, body: str) -> str:
        """Send a single WhatsApp message. Returns the Evolution message key.

        Raises:
            WhatsAppDeliveryError: when Evolution rejects the message after
                the retry budget is exhausted, or on unexpected errors.
        """
        # Safety truncation (Evolution has no hard limit, but be sane)
        if len(body) > MAX_MESSAGE_CHARS:
            body = body[:MAX_MESSAGE_CHARS]

        try:
            message_key = self._send_with_retry(body)
        except _EvolutionHTTPError as exc:
            log.error(
                "Evolution rejected message (status={}): {}",
                exc.status, exc,
            )
            raise WhatsAppDeliveryError(
                f"Evolution error {exc.status}: {exc}"
            ) from exc
        except Exception as exc:
            log.error(
                "Unexpected error sending WhatsApp via Evolution: {}: {}",
                type(exc).__name__, exc,
            )
            raise WhatsAppDeliveryError(
                f"Unexpected delivery error: {type(exc).__name__}: {exc}"
            ) from exc

        log.info("WhatsApp delivered → {} (key={})", self._recipient, message_key)
        return message_key

    def send_many(self, bodies: list[str]) -> list[str]:
        """Send a sequence of messages. Fail-fast on first error.

        Returns the list of message keys in the same order as `bodies`.
        """
        keys: list[str] = []
        for i, body in enumerate(bodies, start=1):
            log.debug("Sending WhatsApp chunk {}/{} ({} chars)", i, len(bodies), len(body))
            keys.append(self.send(body))
        return keys

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
        client = self._http_client()
        url = f"{self._base_url}/message/sendText/{self._instance}"

        response = client.post(
            url,
            json={
                "number": self._recipient,
                "text": body,
            },
            headers={"apikey": self._api_key},
            timeout=30.0,
        )

        if response.status_code >= 400:
            detail = response.text[:500]
            raise _EvolutionHTTPError(response.status_code, detail)

        data = response.json()
        # Evolution returns {"key": {"id": "MESSAGE_ID"}} on success
        key = data.get("key", {}).get("id", "unknown")
        return key

    # ------------------------------------------------------------------ #
    # Lazy HTTP client construction
    # ------------------------------------------------------------------ #

    def _http_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_number(number: str) -> str:
        """Strip 'whatsapp:+' prefix if present, keep only digits."""
        cleaned = number.replace("whatsapp:", "").replace("+", "").strip()
        return cleaned

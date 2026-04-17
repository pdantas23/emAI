"""Integration tests for `src/messaging/`.

Coverage map:

  ┌─────────────────────────┬──────────────────────────────────────────────┐
  │ Module                  │ What we prove                                │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ formatter.truncate      │ Honors the 4096-char cap, marker only when   │
  │                         │ actually cut, boundary at exact limit.       │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ formatter.chunk_messages│ Empty/single/multi packing, ORDER preserved, │
  │                         │ exact-limit boundaries, oversized isolation, │
  │                         │ every chunk ≤ cap.                           │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ WhatsAppClient.send     │ Lazy SDK init, defensive truncation,         │
  │                         │ tenacity retry on 429/5xx, fail-fast on      │
  │                         │ 4xx, exception chaining.                     │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ WhatsAppClient.send_many│ Fail-fast — first failure aborts the rest.   │
  │                         │ No partial-delivery surprises for the user.  │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ _is_retryable           │ HTTP-status-driven classification matches    │
  │                         │ the documented policy (429 + 5xx only).      │
  └─────────────────────────┴──────────────────────────────────────────────┘

Mocking strategy: we use the `client=` injection point on `WhatsAppClient`
to swap in a `MagicMock` for the Twilio SDK. No HTTP, no patching of module
internals, no risk of accidentally hitting the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from twilio.base.exceptions import TwilioRestException

from src.messaging.formatter import (
    DEFAULT_SEPARATOR,
    WHATSAPP_MAX_CHARS,
    chunk_messages,
    truncate,
)
from src.messaging.whatsapp_twilio import (
    WhatsAppClient,
    WhatsAppDeliveryError,
    _is_retryable,
)


# =========================================================================== #
# Global fixtures
# =========================================================================== #


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same trick as the AI suite: tenacity sleeps via `time.sleep`, so we
    patch it to a no-op. Without this, retry tests sit through 7s of sleep."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def _twilio_message(sid: str = "SM" + "x" * 32) -> SimpleNamespace:
    """Mirror the subset of `twilio.rest.api.v2010.account.message.MessageInstance`
    we actually consume — just `.sid`."""
    return SimpleNamespace(sid=sid)


def _twilio_error(status: int, *, code: int | None = None, msg: str = "boom") -> TwilioRestException:
    return TwilioRestException(status, "/2010-04-01/Messages.json", msg=msg, code=code)


def _make_client(mock_sdk: MagicMock | None = None) -> tuple[WhatsAppClient, MagicMock]:
    """Build a `WhatsAppClient` wired to a fresh MagicMock Twilio SDK.

    Returns the client plus the mock so tests can assert on `.messages.create`
    call arguments without re-deriving the path through the SDK tree.
    """
    sdk = mock_sdk or MagicMock(name="TwilioClient")
    return WhatsAppClient(client=sdk), sdk


# =========================================================================== #
# 1. formatter.truncate
# =========================================================================== #


class TestTruncate:
    def test_short_message_returned_unchanged(self) -> None:
        assert truncate("hello") == "hello"

    def test_message_at_exact_limit_returned_unchanged(self) -> None:
        """Boundary: a message of EXACTLY 4096 chars must NOT be touched —
        otherwise the truncation marker would push it over the limit."""
        msg = "x" * WHATSAPP_MAX_CHARS
        assert truncate(msg) == msg
        assert len(truncate(msg)) == WHATSAPP_MAX_CHARS

    def test_oversized_message_capped_at_exact_limit(self) -> None:
        """The truncation marker is INCLUDED in the cap — the result must
        always fit, marker and all."""
        result = truncate("x" * 5000)
        assert len(result) == WHATSAPP_MAX_CHARS

    def test_truncation_marker_appended_when_cut(self) -> None:
        result = truncate("x" * 5000)
        assert result.endswith("…[mensagem truncada]")

    def test_no_marker_when_no_truncation_happened(self) -> None:
        """We only signal truncation when we actually cut — otherwise we'd
        confuse the user with a marker on a perfectly fine message."""
        result = truncate("hello world")
        assert "truncada" not in result

    def test_custom_max_chars_respected(self) -> None:
        """Useful for tests and for hypothetical future channels with smaller
        limits (e.g. SMS at 160)."""
        result = truncate("x" * 200, max_chars=100)
        assert len(result) == 100
        assert result.endswith("…[mensagem truncada]")

    def test_pathological_max_smaller_than_marker(self) -> None:
        """If someone calls truncate with max_chars=10, the marker alone
        (~21 chars) is longer than the budget. We must still return ≤10
        chars and never raise."""
        result = truncate("x" * 100, max_chars=10)
        assert len(result) <= 10


# =========================================================================== #
# 2. formatter.chunk_messages
# =========================================================================== #


class TestChunkMessages:
    def test_empty_input_returns_empty_list(self) -> None:
        assert chunk_messages([]) == []

    def test_single_small_message_returns_one_chunk(self) -> None:
        assert chunk_messages(["hello"]) == ["hello"]

    def test_single_message_at_exact_limit_returns_one_chunk(self) -> None:
        """Boundary: a single message of EXACTLY 4096 chars must pass through
        as its own chunk, untouched (not truncated, not isolated weirdly)."""
        msg = "x" * WHATSAPP_MAX_CHARS
        chunks = chunk_messages([msg])
        assert chunks == [msg]
        assert len(chunks[0]) == WHATSAPP_MAX_CHARS

    def test_two_small_messages_joined_in_one_chunk(self) -> None:
        result = chunk_messages(["a", "b"])
        assert result == ["a" + DEFAULT_SEPARATOR + "b"]

    def test_two_messages_at_exact_combined_limit_join(self) -> None:
        """Boundary: when message_a + separator + message_b == 4096 EXACTLY,
        the two must fit in a single chunk. One char more and they must split."""
        sep_len = len(DEFAULT_SEPARATOR)
        # Total = 2 * payload + sep_len = WHATSAPP_MAX_CHARS
        payload_len = (WHATSAPP_MAX_CHARS - sep_len) // 2
        # If WHATSAPP_MAX_CHARS - sep_len is odd we lose 1 char to integer
        # division, so reconstruct precisely.
        a = "a" * payload_len
        b = "b" * (WHATSAPP_MAX_CHARS - sep_len - payload_len)

        result = chunk_messages([a, b])
        assert len(result) == 1, "Messages summing to exactly the cap must join"
        assert len(result[0]) == WHATSAPP_MAX_CHARS

    def test_two_messages_one_char_over_limit_split(self) -> None:
        """The other side of the boundary: one extra character forces a split."""
        sep_len = len(DEFAULT_SEPARATOR)
        payload_len = (WHATSAPP_MAX_CHARS - sep_len) // 2
        a = "a" * payload_len
        b = "b" * (WHATSAPP_MAX_CHARS - sep_len - payload_len + 1)  # +1 → overflow

        result = chunk_messages([a, b])
        assert len(result) == 2, "One char over the cap must force a split"
        assert result[0] == a
        assert result[1] == b

    def test_oversized_single_message_isolated_and_truncated(self) -> None:
        giant = "X" * 10_000
        result = chunk_messages(["small", giant, "tail"])

        assert len(result) == 3
        assert result[0] == "small"
        assert len(result[1]) == WHATSAPP_MAX_CHARS, "Giant must be truncated"
        assert result[1].endswith("…[mensagem truncada]")
        assert result[2] == "tail"

    def test_oversized_at_start_does_not_eat_subsequent_messages(self) -> None:
        """Regression: an oversized first message used to swallow the
        otherwise-empty current chunk and the small follow-up. Verify
        we still see all three."""
        result = chunk_messages(["X" * 10_000, "small1", "small2"])
        assert len(result) == 2
        assert len(result[0]) == WHATSAPP_MAX_CHARS  # truncated giant alone
        assert "small1" in result[1] and "small2" in result[1]

    def test_three_small_messages_pack_into_one_chunk(self) -> None:
        """Greedy packing: if everything fits, it should fit in ONE chunk —
        not three. Otherwise the recipient gets needless message spam."""
        result = chunk_messages(["one", "two", "three"])
        assert len(result) == 1

    def test_order_is_preserved(self) -> None:
        """Order matters — the orchestrator passes priority-sorted briefings
        (urgent first). Reordering would bury the urgent stuff."""
        result = chunk_messages(["X" * 3000, "Y" * 3000, "Z" * 3000])
        joined = "".join(result)
        assert joined.index("X") < joined.index("Y") < joined.index("Z")

    def test_every_chunk_respects_max_chars(self) -> None:
        """Property test masquerading as a unit test: regardless of input,
        no chunk exceeds the cap. If this ever fails we have a Twilio bug."""
        messages = ["x" * 1500] * 10  # 10 × 1500 chars = 15KB total
        for chunk in chunk_messages(messages):
            assert len(chunk) <= WHATSAPP_MAX_CHARS

    def test_custom_separator_used_in_join(self) -> None:
        result = chunk_messages(["a", "b"], separator="|||")
        assert result == ["a|||b"]


# =========================================================================== #
# 3. WhatsAppClient.send — happy path & defensive behavior
# =========================================================================== #


class TestSendHappyPath:
    def test_returns_twilio_message_sid(self) -> None:
        client, sdk = _make_client()
        sdk.messages.create.return_value = _twilio_message("SM123")

        sid = client.send("hello")
        assert sid == "SM123"

    def test_passes_from_to_body_to_twilio(self) -> None:
        """The exact call shape Twilio expects — easy to break, easy to test."""
        client, sdk = _make_client()
        sdk.messages.create.return_value = _twilio_message()

        client.send("hello world")

        sdk.messages.create.assert_called_once()
        kwargs = sdk.messages.create.call_args.kwargs
        assert kwargs["from_"] == client._sender
        assert kwargs["to"] == client._recipient
        assert kwargs["body"] == "hello world"

    def test_lazy_sdk_construction_when_no_client_injected(self) -> None:
        """No injected client → `_client` is None until first send. Important
        because the orchestrator may construct the client at startup before
        we're sure the network is reachable."""
        client = WhatsAppClient()
        assert client._client is None

    def test_lazy_sdk_built_with_account_credentials_on_first_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no client is injected, the first send() must build the real
        Twilio SDK using the configured account_sid + auth_token. We patch
        the SDK constructor itself so no network call is attempted."""
        captured: dict[str, str] = {}

        def fake_twilio_ctor(sid: str, token: str) -> MagicMock:
            captured["sid"] = sid
            captured["token"] = token
            sdk = MagicMock()
            sdk.messages.create.return_value = _twilio_message("SM-lazy")
            return sdk

        monkeypatch.setattr(
            "src.messaging.whatsapp_twilio.TwilioClient", fake_twilio_ctor
        )

        client = WhatsAppClient()  # no client= injected
        assert client.send("hi") == "SM-lazy"
        # Credentials were forwarded to the SDK constructor — exactly once.
        assert captured["sid"].startswith("AC")
        assert len(captured["token"]) > 0


class TestSendDefensiveTruncation:
    def test_oversized_body_is_truncated_before_reaching_twilio(self) -> None:
        """Defense in depth: even if a caller forgets to chunk, we never let
        the request exceed Twilio's 4096-char limit. Twilio would reject it
        with 400 — better to ship a truncated report than no report."""
        client, sdk = _make_client()
        sdk.messages.create.return_value = _twilio_message()

        client.send("x" * 5000)

        sent_body = sdk.messages.create.call_args.kwargs["body"]
        assert len(sent_body) == WHATSAPP_MAX_CHARS
        assert sent_body.endswith("…[mensagem truncada]")


# =========================================================================== #
# 4. WhatsAppClient.send — retry behavior
# =========================================================================== #


class TestSendRetryOn5xx:
    def test_500_then_success_retries_and_returns_sid(self) -> None:
        """Transient server error → tenacity retries → second call succeeds."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_error(500, msg="internal error"),
            _twilio_message("SM-recovered"),
        ]

        sid = client.send("hello")
        assert sid == "SM-recovered"
        assert sdk.messages.create.call_count == 2

    def test_503_then_success_retries(self) -> None:
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_error(503),
            _twilio_message("SM-ok"),
        ]
        assert client.send("hi") == "SM-ok"
        assert sdk.messages.create.call_count == 2

    def test_429_then_success_retries(self) -> None:
        """Rate-limited → back off and try again."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_error(429, code=20429, msg="too many requests"),
            _twilio_message("SM-after-throttle"),
        ]
        assert client.send("hi") == "SM-after-throttle"
        assert sdk.messages.create.call_count == 2

    def test_repeated_5xx_exhausts_budget_then_raises(self) -> None:
        """3 attempts (the tenacity stop_after_attempt setting) — then up."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [_twilio_error(500)] * 5  # more than budget

        with pytest.raises(WhatsAppDeliveryError, match="500"):
            client.send("hi")

        # Must have used the full budget — no more, no less.
        assert sdk.messages.create.call_count == 3

    def test_repeated_429_exhausts_budget_then_raises(self) -> None:
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [_twilio_error(429)] * 5

        with pytest.raises(WhatsAppDeliveryError):
            client.send("hi")

        assert sdk.messages.create.call_count == 3

    def test_delivery_error_chains_original_twilio_exception(self) -> None:
        """When we give up, the underlying Twilio exception must be on
        `__cause__` so operators can pattern-match in error tracking."""
        client, sdk = _make_client()
        original = _twilio_error(500, code=20500, msg="upstream timeout")
        sdk.messages.create.side_effect = [original] * 3

        with pytest.raises(WhatsAppDeliveryError) as exc_info:
            client.send("hi")
        assert isinstance(exc_info.value.__cause__, TwilioRestException)
        assert exc_info.value.__cause__.status == 500


class TestSendNonRetryable:
    """4xx (other than 429) and unrelated exceptions are PERMANENT failures.
    Retrying them just wastes time and rate-limit budget."""

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_status_does_not_trigger_retry(self, status: int) -> None:
        client, sdk = _make_client()
        sdk.messages.create.side_effect = _twilio_error(status, msg="nope")

        with pytest.raises(WhatsAppDeliveryError, match=str(status)):
            client.send("hi")

        # ONE attempt, no retries — that's the whole point.
        assert sdk.messages.create.call_count == 1

    def test_non_twilio_exception_is_wrapped_as_delivery_error(self) -> None:
        """A network blip surfacing as a generic exception still has to
        translate into the error type the orchestrator knows about."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = ConnectionError("DNS lookup failed")

        with pytest.raises(WhatsAppDeliveryError) as exc_info:
            client.send("hi")
        assert isinstance(exc_info.value.__cause__, ConnectionError)
        # No retry on plain Exception either — single attempt.
        assert sdk.messages.create.call_count == 1


# =========================================================================== #
# 5. WhatsAppClient.send_many — fail-fast semantics
# =========================================================================== #


class TestSendMany:
    def test_all_succeed_returns_sids_in_order(self) -> None:
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_message("SM-1"),
            _twilio_message("SM-2"),
            _twilio_message("SM-3"),
        ]

        sids = client.send_many(["a", "b", "c"])
        assert sids == ["SM-1", "SM-2", "SM-3"]

    def test_empty_list_returns_empty_and_makes_no_calls(self) -> None:
        client, sdk = _make_client()
        assert client.send_many([]) == []
        sdk.messages.create.assert_not_called()

    def test_fail_fast_aborts_on_first_failure(self) -> None:
        """The CRITICAL test: if message #2 fails permanently, message #3
        must NOT be sent. A partial digest is worse than no digest — the
        user has no way to know they're missing the third briefing."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_message("SM-1"),
            _twilio_error(401, msg="auth"),  # permanent failure on #2
            _twilio_message("SM-3"),         # MUST NOT be reached
        ]

        with pytest.raises(WhatsAppDeliveryError, match="401"):
            client.send_many(["a", "b", "c"])

        # Exactly TWO attempts: success + permanent failure. The third
        # message must never have been dispatched.
        assert sdk.messages.create.call_count == 2

    def test_fail_after_retries_exhausted_still_aborts_remainder(self) -> None:
        """Same fail-fast contract when failure happens AFTER retry budget:
        once we give up on chunk N, chunks N+1, N+2... must not be sent."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_message("SM-1"),
            _twilio_error(500), _twilio_error(500), _twilio_error(500),  # exhaust on #2
            _twilio_message("SM-3"),  # MUST NOT be reached
        ]

        with pytest.raises(WhatsAppDeliveryError):
            client.send_many(["a", "b", "c"])

        # 1 success + 3 failed retries on #2 = 4 total. #3 was never tried.
        assert sdk.messages.create.call_count == 4


class TestSendChunked:
    def test_chunks_then_sends_in_order(self) -> None:
        """The convenience entry point: caller hands a list of pre-rendered
        briefings, we chunk + send."""
        client, sdk = _make_client()
        sdk.messages.create.side_effect = [
            _twilio_message("SM-1"),
            _twilio_message("SM-2"),
        ]

        # Two messages that combined exceed the cap → chunked into 2 calls.
        big = "x" * 3000
        sids = client.send_chunked([big, big])

        assert sids == ["SM-1", "SM-2"]
        assert sdk.messages.create.call_count == 2

    def test_chunked_empty_list_makes_no_calls(self) -> None:
        client, sdk = _make_client()
        assert client.send_chunked([]) == []
        sdk.messages.create.assert_not_called()


# =========================================================================== #
# 6. _is_retryable — HTTP status policy
# =========================================================================== #


class TestRetryClassification:
    """The retry whitelist is the contract that protects us from hammering
    Twilio with permanent 4xx errors. Pin it down explicitly."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_statuses(self, status: int) -> None:
        assert _is_retryable(_twilio_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 410, 422, 451])
    def test_non_retryable_4xx_statuses(self, status: int) -> None:
        assert _is_retryable(_twilio_error(status)) is False

    def test_non_twilio_exceptions_are_not_retryable(self) -> None:
        """A `ConnectionError` is NOT a TwilioRestException, so it should
        not enter the retry path. (We translate it to DeliveryError instead.)"""
        assert _is_retryable(ConnectionError("dns")) is False
        assert _is_retryable(ValueError("bad input")) is False
        assert _is_retryable(RuntimeError("whatever")) is False

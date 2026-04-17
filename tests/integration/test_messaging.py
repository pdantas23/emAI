"""Integration tests for `src/messaging/`.

Coverage map:

  ┌─────────────────────────┬──────────────────────────────────────────────┐
  │ Module                  │ What we prove                                │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ formatter.truncate      │ Honors the safety cap, marker only when      │
  │                         │ actually cut, boundary at exact limit.       │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ formatter.chunk_messages│ Empty/single/multi packing, ORDER preserved, │
  │                         │ exact-limit boundaries, oversized isolation, │
  │                         │ every chunk ≤ cap.                           │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ EvolutionClient.send    │ Lazy HTTP client init, retry on 429/5xx,     │
  │                         │ fail-fast on 4xx, exception chaining.        │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ EvolutionClient.send_many│ Fail-fast — first failure aborts the rest.  │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ _is_retryable           │ HTTP-status-driven classification matches    │
  │                         │ the documented policy (429 + 5xx only).      │
  └─────────────────────────┴──────────────────────────────────────────────┘

Mocking strategy: we inject a `MagicMock` httpx.Client via the `http_client=`
parameter. No real HTTP calls, no patching of module internals.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from src.messaging.formatter import (
    DEFAULT_SEPARATOR,
    WHATSAPP_MAX_CHARS,
    chunk_messages,
    truncate,
)
from src.messaging.evolution_api import (
    EvolutionClient,
    WhatsAppDeliveryError,
    _EvolutionHTTPError,
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


def _ok_response(message_id: str = "MSG-123") -> httpx.Response:
    """Simulate a successful Evolution API response."""
    return httpx.Response(
        200,
        json={"key": {"id": message_id}},
        request=httpx.Request("POST", "https://evo.test/message/sendText/inst"),
    )


def _error_response(status: int, body: str = "error") -> httpx.Response:
    """Simulate an Evolution API error response."""
    return httpx.Response(
        status,
        text=body,
        request=httpx.Request("POST", "https://evo.test/message/sendText/inst"),
    )


def _make_client(mock_http: MagicMock | None = None) -> tuple[EvolutionClient, MagicMock]:
    """Build an `EvolutionClient` wired to a fresh MagicMock httpx.Client."""
    http = mock_http or MagicMock(name="httpx.Client")
    return EvolutionClient(
        base_url="https://evo.test",
        api_key="fake-key",
        instance="test-instance",
        whatsapp_to="5511999999999",
        http_client=http,
    ), http


# =========================================================================== #
# 1. formatter.truncate
# =========================================================================== #


class TestTruncate:
    def test_short_message_returned_unchanged(self) -> None:
        assert truncate("hello") == "hello"

    def test_message_at_exact_limit_returned_unchanged(self) -> None:
        msg = "x" * WHATSAPP_MAX_CHARS
        assert truncate(msg) == msg
        assert len(truncate(msg)) == WHATSAPP_MAX_CHARS

    def test_oversized_message_capped_at_exact_limit(self) -> None:
        result = truncate("x" * (WHATSAPP_MAX_CHARS + 1000))
        assert len(result) == WHATSAPP_MAX_CHARS

    def test_truncation_marker_appended_when_cut(self) -> None:
        result = truncate("x" * (WHATSAPP_MAX_CHARS + 1000))
        assert result.endswith("…[mensagem truncada]")

    def test_no_marker_when_no_truncation_happened(self) -> None:
        result = truncate("hello world")
        assert "truncada" not in result

    def test_custom_max_chars_respected(self) -> None:
        result = truncate("x" * 200, max_chars=100)
        assert len(result) == 100
        assert result.endswith("…[mensagem truncada]")

    def test_pathological_max_smaller_than_marker(self) -> None:
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
        msg = "x" * WHATSAPP_MAX_CHARS
        chunks = chunk_messages([msg])
        assert chunks == [msg]

    def test_two_small_messages_joined_in_one_chunk(self) -> None:
        result = chunk_messages(["a", "b"])
        assert result == ["a" + DEFAULT_SEPARATOR + "b"]

    def test_two_messages_at_exact_combined_limit_join(self) -> None:
        sep_len = len(DEFAULT_SEPARATOR)
        payload_len = (WHATSAPP_MAX_CHARS - sep_len) // 2
        a = "a" * payload_len
        b = "b" * (WHATSAPP_MAX_CHARS - sep_len - payload_len)
        result = chunk_messages([a, b])
        assert len(result) == 1
        assert len(result[0]) == WHATSAPP_MAX_CHARS

    def test_two_messages_one_char_over_limit_split(self) -> None:
        sep_len = len(DEFAULT_SEPARATOR)
        payload_len = (WHATSAPP_MAX_CHARS - sep_len) // 2
        a = "a" * payload_len
        b = "b" * (WHATSAPP_MAX_CHARS - sep_len - payload_len + 1)
        result = chunk_messages([a, b])
        assert len(result) == 2

    def test_oversized_single_message_isolated_and_truncated(self) -> None:
        giant = "X" * (WHATSAPP_MAX_CHARS + 5000)
        result = chunk_messages(["small", giant, "tail"])
        assert len(result) == 3
        assert result[0] == "small"
        assert len(result[1]) == WHATSAPP_MAX_CHARS
        assert result[1].endswith("…[mensagem truncada]")
        assert result[2] == "tail"

    def test_three_small_messages_pack_into_one_chunk(self) -> None:
        result = chunk_messages(["one", "two", "three"])
        assert len(result) == 1

    def test_order_is_preserved(self) -> None:
        result = chunk_messages(["X" * 30000, "Y" * 30000, "Z" * 30000])
        joined = "".join(result)
        assert joined.index("X") < joined.index("Y") < joined.index("Z")

    def test_every_chunk_respects_max_chars(self) -> None:
        messages = ["x" * 15000] * 10
        for chunk in chunk_messages(messages):
            assert len(chunk) <= WHATSAPP_MAX_CHARS

    def test_custom_separator_used_in_join(self) -> None:
        result = chunk_messages(["a", "b"], separator="|||")
        assert result == ["a|||b"]


# =========================================================================== #
# 3. EvolutionClient.send — happy path
# =========================================================================== #


class TestSendHappyPath:
    def test_returns_evolution_message_key(self) -> None:
        client, http = _make_client()
        http.post.return_value = _ok_response("MSG-abc")

        key = client.send("hello")
        assert key == "MSG-abc"

    def test_passes_correct_payload_to_evolution(self) -> None:
        client, http = _make_client()
        http.post.return_value = _ok_response()

        client.send("hello world")

        http.post.assert_called_once()
        kwargs = http.post.call_args.kwargs
        assert kwargs["json"]["number"] == "5511999999999"
        assert kwargs["json"]["text"] == "hello world"
        assert kwargs["headers"]["apikey"] == "fake-key"

    def test_lazy_http_client_construction(self) -> None:
        client = EvolutionClient(
            base_url="https://evo.test",
            api_key="fake",
            instance="inst",
            whatsapp_to="5511999999999",
        )
        assert client._client is None

    def test_normalizes_whatsapp_number(self) -> None:
        client = EvolutionClient(
            base_url="https://evo.test",
            api_key="fake",
            instance="inst",
            whatsapp_to="whatsapp:+5511999999999",
        )
        assert client._recipient == "5511999999999"


# =========================================================================== #
# 4. EvolutionClient.send — retry behavior
# =========================================================================== #


class TestSendRetryOn5xx:
    def test_500_then_success_retries_and_returns_key(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [
            _error_response(500, "internal error"),
            _ok_response("MSG-recovered"),
        ]
        key = client.send("hello")
        assert key == "MSG-recovered"
        assert http.post.call_count == 2

    def test_429_then_success_retries(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [
            _error_response(429, "rate limited"),
            _ok_response("MSG-ok"),
        ]
        assert client.send("hi") == "MSG-ok"
        assert http.post.call_count == 2

    def test_repeated_5xx_exhausts_budget_then_raises(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [_error_response(500)] * 5

        with pytest.raises(WhatsAppDeliveryError, match="500"):
            client.send("hi")

        assert http.post.call_count == 3

    def test_delivery_error_chains_original_exception(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [_error_response(500, "upstream timeout")] * 3

        with pytest.raises(WhatsAppDeliveryError) as exc_info:
            client.send("hi")
        assert isinstance(exc_info.value.__cause__, _EvolutionHTTPError)
        assert exc_info.value.__cause__.status == 500


class TestSendNonRetryable:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_status_does_not_trigger_retry(self, status: int) -> None:
        client, http = _make_client()
        http.post.return_value = _error_response(status, "nope")

        with pytest.raises(WhatsAppDeliveryError, match=str(status)):
            client.send("hi")

        assert http.post.call_count == 1

    def test_non_http_exception_is_wrapped_as_delivery_error(self) -> None:
        client, http = _make_client()
        http.post.side_effect = ConnectionError("DNS lookup failed")  # exception is fine for side_effect

        with pytest.raises(WhatsAppDeliveryError) as exc_info:
            client.send("hi")
        assert isinstance(exc_info.value.__cause__, ConnectionError)
        assert http.post.call_count == 1


# =========================================================================== #
# 5. EvolutionClient.send_many — fail-fast semantics
# =========================================================================== #


class TestSendMany:
    def test_all_succeed_returns_keys_in_order(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [
            _ok_response("MSG-1"),
            _ok_response("MSG-2"),
            _ok_response("MSG-3"),
        ]
        keys = client.send_many(["a", "b", "c"])
        assert keys == ["MSG-1", "MSG-2", "MSG-3"]

    def test_empty_list_returns_empty_and_makes_no_calls(self) -> None:
        client, http = _make_client()
        assert client.send_many([]) == []
        http.post.assert_not_called()

    def test_fail_fast_aborts_on_first_failure(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [
            _ok_response("MSG-1"),
            _error_response(401, "auth"),
            _ok_response("MSG-3"),  # MUST NOT be reached
        ]

        with pytest.raises(WhatsAppDeliveryError, match="401"):
            client.send_many(["a", "b", "c"])

        assert http.post.call_count == 2

    def test_fail_after_retries_exhausted_still_aborts_remainder(self) -> None:
        client, http = _make_client()
        http.post.side_effect = [
            _ok_response("MSG-1"),
            _error_response(500), _error_response(500), _error_response(500),
            _ok_response("MSG-3"),  # MUST NOT be reached
        ]

        with pytest.raises(WhatsAppDeliveryError):
            client.send_many(["a", "b", "c"])

        assert http.post.call_count == 4


# =========================================================================== #
# 6. _is_retryable — HTTP status policy
# =========================================================================== #


class TestRetryClassification:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_statuses(self, status: int) -> None:
        assert _is_retryable(_EvolutionHTTPError(status, "test")) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 410, 422, 451])
    def test_non_retryable_4xx_statuses(self, status: int) -> None:
        assert _is_retryable(_EvolutionHTTPError(status, "test")) is False

    def test_non_evolution_exceptions_are_not_retryable(self) -> None:
        assert _is_retryable(ConnectionError("dns")) is False
        assert _is_retryable(ValueError("bad input")) is False
        assert _is_retryable(RuntimeError("whatever")) is False

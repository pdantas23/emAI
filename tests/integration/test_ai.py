"""Integration tests for the AI layer (`src/ai/`).

Coverage map:

  ┌─────────────────────────┬──────────────────────────────────────────────┐
  │ Module                  │ What we prove                                │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ LLMClient               │ Retry budget, primary→fallback handoff,      │
  │                         │ model translation, both-fail behavior,       │
  │                         │ retryable vs non-retryable classification.   │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ EmailClassifier         │ Strict + tolerant JSON parsing, fail-OPEN    │
  │                         │ on every failure mode, body truncation,      │
  │                         │ uses the cheap (haiku) model.                │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ EmailSummarizer         │ Strict + tolerant JSON parsing, fail-LOUD    │
  │                         │ (SummaryFailure) on every failure mode,      │
  │                         │ uses the expensive (sonnet) model,           │
  │                         │ priority is forwarded into the prompt.       │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ format_for_whatsapp /   │ Single-asterisk bold, correct emoji badges   │
  │ format_digest           │ per priority, all three sections rendered,   │
  │                         │ separator between briefings in a digest.     │
  └─────────────────────────┴──────────────────────────────────────────────┘

Mocking philosophy:
  - LLMClient tests patch the SDK *classes* (`anthropic.Anthropic`,
    `openai.OpenAI`) so we exercise the real retry/fallback wiring without
    touching the network.
  - Classifier and Summarizer tests inject a `FakeLLMClient` directly —
    we already test the SDK plumbing one layer down, no need to repeat it.
  - Formatter tests are pure: no mocks, no fixtures.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
import pytest

from config.settings import LLMProvider
from src.ai.classifier import (
    Classification,
    EmailClassifier,
    Priority,
)
from src.ai.llm_client import (
    LLMClient,
    LLMError,
    LLMResponse,
    _translate_model,
)
from src.ai.summarizer import (
    EmailSummarizer,
    EmailSummary,
    SummaryFailure,
    format_digest,
    format_for_whatsapp,
)
from src.email_client.base import RawEmail


# =========================================================================== #
# Global test fixtures
# =========================================================================== #


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tenacity retries fire instantly so we don't sit through 7s of sleep
    per LLMClient test. We patch `time.sleep` because tenacity's `nap` module
    delegates to it; this is the smallest, safest hammer."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


@pytest.fixture
def sample_email() -> RawEmail:
    """A tiny but realistic email used across classifier/summarizer tests."""
    return RawEmail(
        uid="42",
        message_id="<msg-42@example.com>",
        folder="INBOX",
        sender_name="Maria Souza",
        sender_email="maria@cliente-importante.com.br",
        to=["philip@example.com"],
        subject="Renovação do contrato — prazo até sexta",
        date=datetime(2026, 4, 16, 10, 30),
        body_text=(
            "Oi Philip, tudo bem?\n\n"
            "Conforme conversamos, anexo a proposta de renovação. "
            "Preciso da sua decisão até sexta (18/04) para fechar com o jurídico.\n\n"
            "Abraço,\nMaria"
        ),
    )


# =========================================================================== #
# Helpers — build SDK exception instances + fake SDK responses
# =========================================================================== #


def _httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.fake.local/v1/messages")


def _httpx_response(status: int = 429) -> httpx.Response:
    return httpx.Response(status, request=_httpx_request())


def _anthropic_rate_limit() -> anthropic.RateLimitError:
    return anthropic.RateLimitError("rate limited", response=_httpx_response(429), body=None)


def _anthropic_500() -> anthropic.InternalServerError:
    return anthropic.InternalServerError("boom", response=_httpx_response(500), body=None)


def _anthropic_auth_error() -> anthropic.AuthenticationError:
    return anthropic.AuthenticationError("bad key", response=_httpx_response(401), body=None)


def _openai_rate_limit() -> openai.RateLimitError:
    return openai.RateLimitError("rate limited", response=_httpx_response(429), body=None)


def _openai_500() -> openai.InternalServerError:
    return openai.InternalServerError("boom", response=_httpx_response(500), body=None)


def _fake_anthropic_response(text: str) -> SimpleNamespace:
    """Mirror the subset of `anthropic.types.Message` LLMClient reads."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=50, output_tokens=20),
        stop_reason="end_turn",
    )


def _fake_openai_response(text: str) -> SimpleNamespace:
    """Mirror the subset of `openai.types.ChatCompletion` LLMClient reads."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
    )


# =========================================================================== #
# Fake LLMClient for classifier/summarizer tests — we don't need the SDK plumbing
# =========================================================================== #


class FakeLLMClient:
    """Records every call and returns canned responses or raises canned errors.

    Pass `LLMResponse` instances OR `Exception` instances to the constructor.
    Calls consume them in order. If the call list runs dry, the test fails
    loudly — that's how we catch "summarizer accidentally called LLM twice".
    """

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self._queue: list[LLMResponse | Exception] = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._queue:
            raise AssertionError("FakeLLMClient: unexpected extra call")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(text: str, *, model: str = "claude-haiku-4-5-20251001") -> LLMResponse:
    return LLMResponse(
        text=text, model=model, provider="anthropic", tokens_in=50, tokens_out=20
    )


# =========================================================================== #
# 1. LLMClient — retry & fallback wiring
# =========================================================================== #


class TestLLMClientHappyPath:
    def test_primary_returns_immediately_on_success(self) -> None:
        with patch("src.ai.llm_client.anthropic.Anthropic") as MockAnth:
            MockAnth.return_value.messages.create.return_value = _fake_anthropic_response("hi!")

            response = LLMClient().complete("hello")

        assert response.text == "hi!"
        assert response.provider == "anthropic"
        assert response.tokens_in == 50 and response.tokens_out == 20
        # No retries fired.
        assert MockAnth.return_value.messages.create.call_count == 1

    def test_primary_retries_then_succeeds_within_budget(self) -> None:
        """Two transient 429s, then 200 OK — the response should still come from
        the primary; fallback should never be invoked."""
        with patch("src.ai.llm_client.anthropic.Anthropic") as MockAnth, \
             patch("src.ai.llm_client.openai.OpenAI") as MockOAI:
            MockAnth.return_value.messages.create.side_effect = [
                _anthropic_rate_limit(),
                _anthropic_rate_limit(),
                _fake_anthropic_response("recovered"),
            ]

            response = LLMClient().complete("hello")

        assert response.text == "recovered"
        assert response.provider == "anthropic"
        assert MockAnth.return_value.messages.create.call_count == 3
        MockOAI.assert_not_called()


class TestLLMClientFallback:
    def test_primary_exhausts_retries_then_fallback_succeeds(self) -> None:
        """The most important fallback test: Anthropic dies completely (3x 500s),
        OpenAI takes over and answers. Caller must see the OpenAI response."""
        with patch("src.ai.llm_client.anthropic.Anthropic") as MockAnth, \
             patch("src.ai.llm_client.openai.OpenAI") as MockOAI:
            MockAnth.return_value.messages.create.side_effect = [
                _anthropic_500(), _anthropic_500(), _anthropic_500(),
            ]
            MockOAI.return_value.chat.completions.create.return_value = (
                _fake_openai_response("from openai")
            )

            response = LLMClient().complete("hello", model="claude-sonnet-4-6")

        assert response.text == "from openai"
        assert response.provider == "openai"
        # Anthropic burned its full retry budget.
        assert MockAnth.return_value.messages.create.call_count == 3
        # OpenAI was called exactly once and with the TRANSLATED model.
        MockOAI.return_value.chat.completions.create.assert_called_once()
        called_model = MockOAI.return_value.chat.completions.create.call_args.kwargs["model"]
        assert called_model == "gpt-4o", (
            "Fallback must translate claude-sonnet-4-6 → gpt-4o"
        )

    def test_non_retryable_primary_error_still_falls_back(self) -> None:
        """AuthenticationError is NOT in the retryable set, so tenacity reraises
        immediately. But `complete()` catches it and tries the fallback anyway —
        we don't want a single bad config to break the whole pipeline."""
        with patch("src.ai.llm_client.anthropic.Anthropic") as MockAnth, \
             patch("src.ai.llm_client.openai.OpenAI") as MockOAI:
            MockAnth.return_value.messages.create.side_effect = _anthropic_auth_error()
            MockOAI.return_value.chat.completions.create.return_value = (
                _fake_openai_response("saved by openai")
            )

            response = LLMClient().complete("hello")

        assert response.text == "saved by openai"
        assert response.provider == "openai"
        # No retries — auth errors fail fast on the primary.
        assert MockAnth.return_value.messages.create.call_count == 1

    def test_both_providers_fail_raises_LLMError_with_chain(self) -> None:
        """Primary 3x500, fallback 3x500 → LLMError mentioning both providers,
        with the primary exception preserved on `__cause__`."""
        with patch("src.ai.llm_client.anthropic.Anthropic") as MockAnth, \
             patch("src.ai.llm_client.openai.OpenAI") as MockOAI:
            MockAnth.return_value.messages.create.side_effect = [_anthropic_500()] * 3
            MockOAI.return_value.chat.completions.create.side_effect = [_openai_500()] * 3

            with pytest.raises(LLMError) as exc_info:
                LLMClient().complete("hello")

        msg = str(exc_info.value)
        assert "anthropic" in msg.lower() and "openai" in msg.lower()
        # The primary failure must be chained for traceability.
        assert isinstance(exc_info.value.__cause__, anthropic.InternalServerError)


class TestModelTranslation:
    """The translation map is the contract that makes provider-swap a 1-line
    config change. If someone removes a mapping, this should fail."""

    @pytest.mark.parametrize(
        "model,target,expected",
        [
            ("claude-sonnet-4-6", LLMProvider.openai, "gpt-4o"),
            ("claude-haiku-4-5-20251001", LLMProvider.openai, "gpt-4o-mini"),
            ("gpt-4o", LLMProvider.anthropic, "claude-sonnet-4-6"),
            ("gpt-4o-mini", LLMProvider.anthropic, "claude-haiku-4-5-20251001"),
        ],
    )
    def test_known_models_translate(self, model: str, target: LLMProvider, expected: str) -> None:
        assert _translate_model(model, target=target) == expected

    def test_unknown_model_falls_through_unchanged(self) -> None:
        """Intentional design: unknown models pass through unchanged so the
        target SDK fails loudly with BadRequestError instead of silently
        calling some unrelated default."""
        assert _translate_model("custom-finetune-v3", target=LLMProvider.openai) == "custom-finetune-v3"


# =========================================================================== #
# 2. EmailClassifier — strict + tolerant + fail-OPEN
# =========================================================================== #


class TestClassifierHappyPath:
    def test_clean_json_parses_to_classification(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(
            '{"relevance": true, "priority": "high", "reason": "prazo até sexta"}'
        ))
        verdict = EmailClassifier(llm=fake).classify(sample_email)

        assert verdict.relevance is True
        assert verdict.priority is Priority.high
        assert verdict.reason == "prazo até sexta"

    def test_uses_cheap_classifier_model(self, sample_email: RawEmail) -> None:
        """The whole point of the classifier is to NOT pay Sonnet prices."""
        fake = FakeLLMClient(_ok('{"relevance": false, "priority": "low", "reason": "spam"}'))
        EmailClassifier(llm=fake).classify(sample_email)

        assert fake.calls[0]["model"] == "claude-haiku-4-5-20251001"

    def test_uses_zero_temperature_for_determinism(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok('{"relevance": true, "priority": "low", "reason": "ok"}'))
        EmailClassifier(llm=fake).classify(sample_email)
        assert fake.calls[0]["temperature"] == 0.0

    def test_prompt_contains_sender_subject_and_body(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok('{"relevance": true, "priority": "low", "reason": "ok"}'))
        EmailClassifier(llm=fake).classify(sample_email)

        prompt = fake.calls[0]["prompt"]
        assert "Maria Souza" in prompt
        assert "maria@cliente-importante.com.br" in prompt
        assert "Renovação do contrato" in prompt
        assert "fechar com o jurídico" in prompt

    def test_is_relevant_convenience_returns_bool(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok('{"relevance": true, "priority": "medium", "reason": "ok"}'))
        assert EmailClassifier(llm=fake).is_relevant(sample_email) is True

        fake2 = FakeLLMClient(_ok('{"relevance": false, "priority": "low", "reason": "newsletter"}'))
        assert EmailClassifier(llm=fake2).is_relevant(sample_email) is False


class TestClassifierTolerantParsing:
    """Real models drift. The classifier must absorb the common drift modes
    instead of failing or — worse — silently dropping the email."""

    def test_strips_json_fences(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(
            '```json\n{"relevance": true, "priority": "high", "reason": "urgente"}\n```'
        ))
        verdict = EmailClassifier(llm=fake).classify(sample_email)
        assert verdict.relevance is True
        assert verdict.priority is Priority.high

    def test_extracts_json_from_leading_prose(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(
            'Claro, aqui está minha análise:\n\n'
            '{"relevance": true, "priority": "medium", "reason": "pedido de revisão"}'
        ))
        verdict = EmailClassifier(llm=fake).classify(sample_email)
        assert verdict.priority is Priority.medium

    def test_handles_bare_fences_no_language_tag(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(
            '```\n{"relevance": false, "priority": "low", "reason": "spam"}\n```'
        ))
        verdict = EmailClassifier(llm=fake).classify(sample_email)
        assert verdict.relevance is False


class TestClassifierFailOpen:
    """When in doubt, ALWAYS let the email through. Each test below represents
    a different failure mode that previously caused us to consider dropping
    emails — they must all converge on `relevance=True, priority=medium`."""

    def _assert_safe_default(self, verdict: Classification) -> None:
        assert verdict.relevance is True, "fail-open: relevance must be True"
        assert verdict.priority is Priority.medium, "fail-open: priority must be medium"
        assert verdict.reason.lower().startswith("fallback"), (
            "Reason should advertise the fallback so operators can spot it in logs"
        )

    def test_empty_response_falls_back_safely(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(""))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))

    def test_no_json_at_all_falls_back_safely(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok("This is just prose with no JSON anywhere."))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))

    def test_invalid_priority_value_falls_back_safely(self, sample_email: RawEmail) -> None:
        """Model returned an enum value that's not in our schema."""
        fake = FakeLLMClient(_ok(
            '{"relevance": true, "priority": "URGENT", "reason": "asap"}'
        ))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))

    def test_extra_field_falls_back_safely(self, sample_email: RawEmail) -> None:
        """`extra="forbid"` rejects hallucinated fields — that triggers the safe default."""
        fake = FakeLLMClient(_ok(
            '{"relevance": true, "priority": "low", "reason": "ok", "category": "X"}'
        ))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))

    def test_missing_required_field_falls_back_safely(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok('{"relevance": true}'))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))

    def test_truncated_json_falls_back_safely(self, sample_email: RawEmail) -> None:
        """Model hit max_tokens mid-response. We must not crash."""
        fake = FakeLLMClient(_ok('{"relevance": true, "priority": "high", "reaso'))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))

    def test_llm_error_falls_back_safely(self, sample_email: RawEmail) -> None:
        """Both providers down? Still don't lose the email."""
        fake = FakeLLMClient(LLMError("both providers exhausted"))
        self._assert_safe_default(EmailClassifier(llm=fake).classify(sample_email))


class TestClassifierBodyTruncation:
    def test_long_body_is_truncated_in_prompt(self, sample_email: RawEmail) -> None:
        """Newsletters can be 50KB. The classifier must cap the cost per call.

        We use a unique marker character (`Q`) that does NOT appear in the
        Portuguese prompt template, so the count is an exact read on what
        crossed the truncation boundary.
        """
        long_email = sample_email.model_copy(update={"body_text": "Q" * 5000})
        fake = FakeLLMClient(_ok('{"relevance": false, "priority": "low", "reason": "spam"}'))
        EmailClassifier(llm=fake).classify(long_email)

        prompt = fake.calls[0]["prompt"]
        assert "[...truncated...]" in prompt
        # Direct check: nowhere in the prompt should there be a run of MORE than
        # 1500 contiguous Q's. (Counting Q's globally is ruined by Portuguese
        # words like "que", "qualquer" in the prompt template.)
        import re
        longest_q_run = max((len(m) for m in re.findall(r"Q+", prompt)), default=0)
        assert longest_q_run == 1500, (
            f"Body must be truncated to a single 1500-Q run; got run of {longest_q_run}"
        )


# =========================================================================== #
# 3. EmailSummarizer — strict + tolerant + fail-LOUD
# =========================================================================== #


class TestSummarizerHappyPath:
    def _good_payload(self) -> str:
        return (
            '{"resumo": "Maria pede decisão sobre renovação até sexta (18/04).",'
            ' "contexto": "Cliente importante. Proposta enviada em anexo. Sem decisão até sexta o jurídico não fecha.",'
            ' "acao": "Decidir sobre a renovação até 2026-04-18."}'
        )

    def test_clean_json_parses_to_summary(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(self._good_payload(), model="claude-sonnet-4-6"))
        summary = EmailSummarizer(llm=fake).summarize(sample_email, priority=Priority.high)

        assert "Maria" in summary.resumo
        assert "Cliente importante" in summary.contexto
        assert summary.acao.startswith("Decidir")

    def test_uses_expensive_sonnet_model(self, sample_email: RawEmail) -> None:
        """The summarizer must always default to the high-quality model. If
        someone accidentally swaps it for haiku, this test screams."""
        fake = FakeLLMClient(_ok(self._good_payload()))
        EmailSummarizer(llm=fake).summarize(sample_email)
        assert fake.calls[0]["model"] == "claude-sonnet-4-6"

    def test_priority_is_forwarded_into_prompt(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(self._good_payload()))
        EmailSummarizer(llm=fake).summarize(sample_email, priority=Priority.high)
        # Prompt template includes "**Prioridade detectada:** {{PRIORITY}}".
        # Match the value next to the label in either bolded or plain form.
        prompt = fake.calls[0]["prompt"]
        assert "Prioridade detectada:** high" in prompt or "Prioridade detectada: high" in prompt

    def test_prompt_contains_sender_subject_and_body(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(self._good_payload()))
        EmailSummarizer(llm=fake).summarize(sample_email, priority=Priority.medium)
        prompt = fake.calls[0]["prompt"]
        assert "Maria Souza" in prompt
        assert "Renovação do contrato" in prompt
        assert "fechar com o jurídico" in prompt


class TestSummarizerTolerantParsing:
    _PAYLOAD = (
        '{"resumo": "r", "contexto": "c", "acao": "a"}'
    )

    def test_strips_json_fences(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(f"```json\n{self._PAYLOAD}\n```"))
        summary = EmailSummarizer(llm=fake).summarize(sample_email)
        assert summary.resumo == "r"

    def test_extracts_json_from_leading_prose(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(f"Aqui está o briefing:\n\n{self._PAYLOAD}"))
        summary = EmailSummarizer(llm=fake).summarize(sample_email)
        assert summary.contexto == "c"


class TestSummarizerFailLoud:
    """The summarizer is the OPPOSITE of the classifier — half-baked summaries
    on WhatsApp are worse than no message. Every error mode must raise
    SummaryFailure so the orchestrator can skip THIS email and move on."""

    def test_empty_response_raises(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok(""))
        with pytest.raises(SummaryFailure, match="Empty response"):
            EmailSummarizer(llm=fake).summarize(sample_email)

    def test_no_json_at_all_raises(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok("Desculpe, não consegui processar este email."))
        with pytest.raises(SummaryFailure, match="No JSON object"):
            EmailSummarizer(llm=fake).summarize(sample_email)

    def test_missing_required_field_raises(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok('{"resumo": "r", "contexto": "c"}'))  # no acao
        with pytest.raises(SummaryFailure, match="validation failed"):
            EmailSummarizer(llm=fake).summarize(sample_email)

    def test_extra_field_raises(self, sample_email: RawEmail) -> None:
        """The model invented a `prioridade` key — refuse to ship."""
        fake = FakeLLMClient(_ok(
            '{"resumo": "r", "contexto": "c", "acao": "a", "prioridade": "alta"}'
        ))
        with pytest.raises(SummaryFailure, match="validation failed"):
            EmailSummarizer(llm=fake).summarize(sample_email)

    def test_field_too_long_raises(self, sample_email: RawEmail) -> None:
        """The Pydantic max_length cap is what protects WhatsApp from a 5KB
        'summary'. Verify it actually fires."""
        oversized = "x" * 1000
        fake = FakeLLMClient(_ok(
            f'{{"resumo": "{oversized}", "contexto": "c", "acao": "a"}}'
        ))
        with pytest.raises(SummaryFailure, match="validation failed"):
            EmailSummarizer(llm=fake).summarize(sample_email)

    def test_truncated_json_raises(self, sample_email: RawEmail) -> None:
        fake = FakeLLMClient(_ok('{"resumo": "r", "contexto": "c", "acao": "a'))
        with pytest.raises(SummaryFailure):
            EmailSummarizer(llm=fake).summarize(sample_email)

    def test_llm_error_chains_into_summary_failure(self, sample_email: RawEmail) -> None:
        original = LLMError("both providers exhausted")
        fake = FakeLLMClient(original)
        with pytest.raises(SummaryFailure, match="LLM call failed") as exc_info:
            EmailSummarizer(llm=fake).summarize(sample_email)
        assert exc_info.value.__cause__ is original


# =========================================================================== #
# 4. WhatsApp formatter — pure rendering
# =========================================================================== #


@pytest.fixture
def sample_summary() -> EmailSummary:
    return EmailSummary(
        resumo="Maria pede decisão sobre renovação até sexta (18/04).",
        contexto="Cliente importante; proposta em anexo; jurídico aguarda go/no-go.",
        acao="Decidir sobre a renovação até 2026-04-18.",
    )


class TestPriorityBadges:
    """Every priority must produce a visually distinct badge so the recipient
    can scan a digest of 10+ messages and spot the urgent ones at a glance."""

    @pytest.mark.parametrize(
        "priority,expected_emoji,expected_label",
        [
            (Priority.high,   "🔴", "URGENTE"),
            (Priority.medium, "🟡", "Atenção"),
            (Priority.low,    "🟢", "Informativo"),
        ],
    )
    def test_each_priority_renders_its_badge(
        self,
        sample_email: RawEmail,
        sample_summary: EmailSummary,
        priority: Priority,
        expected_emoji: str,
        expected_label: str,
    ) -> None:
        classif = Classification(relevance=True, priority=priority, reason="t")
        msg = format_for_whatsapp(sample_email, sample_summary, classif)

        first_line = msg.splitlines()[0]
        assert first_line.startswith(expected_emoji), (
            f"Badge for {priority.value} must lead with {expected_emoji}"
        )
        assert expected_label in first_line


class TestWhatsAppMarkdown:
    """WhatsApp uses single-asterisk bold. Double asterisks render as literal
    text on the phone — so this is a real, user-visible bug if it regresses."""

    def test_bold_uses_single_asterisk_not_double(
        self, sample_email: RawEmail, sample_summary: EmailSummary
    ) -> None:
        classif = Classification(relevance=True, priority=Priority.high, reason="t")
        msg = format_for_whatsapp(sample_email, sample_summary, classif)
        assert "**" not in msg, "Double-asterisk bold breaks on WhatsApp"
        # And we DO use single-asterisk bold somewhere.
        assert "*De:*" in msg
        assert "*Assunto:*" in msg
        assert "*Resumo*" in msg

    def test_separator_line_is_present_under_badge(
        self, sample_email: RawEmail, sample_summary: EmailSummary
    ) -> None:
        classif = Classification(relevance=True, priority=Priority.medium, reason="t")
        lines = format_for_whatsapp(sample_email, sample_summary, classif).splitlines()
        # Line 0 is the badge, line 1 must be the visual divider.
        assert lines[1] == "━" * 18

    def test_all_three_sections_render_with_their_glyphs(
        self, sample_email: RawEmail, sample_summary: EmailSummary
    ) -> None:
        classif = Classification(relevance=True, priority=Priority.medium, reason="t")
        msg = format_for_whatsapp(sample_email, sample_summary, classif)

        assert "💡 *Resumo*" in msg
        assert "🧭 *Contexto*" in msg
        assert "⚡ *Ação Necessária*" in msg

        # Section bodies must follow the headers.
        assert sample_summary.resumo in msg
        assert sample_summary.contexto in msg
        assert sample_summary.acao in msg

    def test_section_order_is_resumo_then_contexto_then_acao(
        self, sample_email: RawEmail, sample_summary: EmailSummary
    ) -> None:
        """The reading order matters — exec briefing format starts with the
        TL;DR and ends with the ask. Verify the order is fixed."""
        classif = Classification(relevance=True, priority=Priority.high, reason="t")
        msg = format_for_whatsapp(sample_email, sample_summary, classif)
        i_resumo = msg.index("*Resumo*")
        i_contexto = msg.index("*Contexto*")
        i_acao = msg.index("*Ação Necessária*")
        assert i_resumo < i_contexto < i_acao


class TestSenderAndSubjectFormatting:
    def test_full_sender_displayed_when_name_present(
        self, sample_email: RawEmail, sample_summary: EmailSummary
    ) -> None:
        classif = Classification(relevance=True, priority=Priority.medium, reason="t")
        msg = format_for_whatsapp(sample_email, sample_summary, classif)
        assert "Maria Souza <maria@cliente-importante.com.br>" in msg

    def test_email_only_displayed_when_name_missing(
        self, sample_summary: EmailSummary
    ) -> None:
        anon = RawEmail(
            uid="1", message_id="<x>", folder="INBOX",
            sender_name=None, sender_email="anon@example.com",
            subject="hi", date=datetime(2026, 4, 16),
        )
        classif = Classification(relevance=True, priority=Priority.low, reason="t")
        msg = format_for_whatsapp(anon, sample_summary, classif)
        assert "anon@example.com" in msg
        assert "<anon@example.com>" not in msg, (
            "When there's no display name we should NOT wrap the address in <>"
        )

    def test_empty_subject_falls_back_to_placeholder(
        self, sample_summary: EmailSummary
    ) -> None:
        no_subj = RawEmail(
            uid="1", message_id="<x>", folder="INBOX",
            sender_name="X", sender_email="x@x.com",
            subject="", date=datetime(2026, 4, 16),
        )
        classif = Classification(relevance=True, priority=Priority.medium, reason="t")
        msg = format_for_whatsapp(no_subj, sample_summary, classif)
        assert "(sem assunto)" in msg


class TestDigest:
    def test_empty_list_returns_empty_string(self) -> None:
        assert format_digest([]) == ""

    def test_single_message_has_no_extra_separator(self) -> None:
        """A digest of one shouldn't add any glue — the user would just see a
        message ending with a stray bar."""
        result = format_digest(["only message"])
        assert result == "only message"

    def test_multiple_messages_joined_with_separator_glue(self) -> None:
        result = format_digest(["first", "second", "third"])
        bar = "━" * 18
        # Exactly TWO separators for THREE messages (joins, not bookends).
        assert result.count(bar) == 2
        # Order is preserved.
        i_first = result.index("first")
        i_second = result.index("second")
        i_third = result.index("third")
        assert i_first < i_second < i_third

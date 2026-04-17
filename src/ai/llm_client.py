"""Unified LLM client with retry + automatic provider fallback.

Wraps the Anthropic and OpenAI SDKs behind a single `complete()` method.
The orchestrator depends on `LLMClient` ONLY — never on the SDKs directly.
Switching primary/fallback is therefore a settings change, never a code change.

Retry policy (per call, per provider — implemented via `tenacity`):
  - 3 attempts max, exponential backoff: 1s → 2s → 4s (capped at 8s)
  - Retried: network errors, server 5xx, rate limit (429), timeouts
  - NOT retried: auth errors, validation errors, content policy violations
    (these will not get better with retries — fail fast)

Fallback policy:
  - When the primary provider exhausts its retries, the fallback provider is
    invoked once (with the equivalent model name from a translation map).
  - The fallback also gets its own retry budget.
  - If both fail, an `LLMError` is raised with both exceptions chained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
import openai
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import LLMProvider, settings
from src.utils.logger import log

# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMResponse:
    """Provider-agnostic LLM completion result."""

    text: str
    model: str
    provider: str  # "anthropic" or "openai"
    tokens_in: int
    tokens_out: int
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class LLMError(Exception):
    """Raised when both primary and fallback providers have failed.

    The original primary exception is chained as `__cause__` (raise ... from ...).
    """


# --------------------------------------------------------------------------- #
# Retry classification — which exceptions we retry vs. let propagate
# --------------------------------------------------------------------------- #

_RETRYABLE_ANTHROPIC: tuple[type[BaseException], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

_RETRYABLE_OPENAI: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _log_before_sleep(state: RetryCallState) -> None:
    """Tenacity hook: log every retry attempt via loguru."""
    exc = state.outcome.exception() if state.outcome else None
    sleep_s = state.next_action.sleep if state.next_action else 0.0
    log.warning(
        "LLM call failed (attempt {}): {} — retrying in {:.1f}s",
        state.attempt_number,
        type(exc).__name__ if exc else "Unknown",
        sleep_s,
    )


# --------------------------------------------------------------------------- #
# Cross-provider model name translation
# --------------------------------------------------------------------------- #

_ANTHROPIC_TO_OPENAI: dict[str, str] = {
    "claude-opus-4-6":             "gpt-4o",
    "claude-sonnet-4-6":           "gpt-4o",
    "claude-haiku-4-5":            "gpt-4o-mini",
    "claude-haiku-4-5-20251001":   "gpt-4o-mini",
}
_OPENAI_TO_ANTHROPIC: dict[str, str] = {
    "gpt-4o":       "claude-sonnet-4-6",
    "gpt-4o-mini":  "claude-haiku-4-5-20251001",
    "gpt-4-turbo":  "claude-sonnet-4-6",
}


def _translate_model(model: str, *, target: LLMProvider) -> str:
    """Map a model name to its closest equivalent on the target provider.

    If no mapping exists, the original string is returned — the target SDK
    will then reject it with a clear `BadRequestError`. That's intentional:
    we'd rather fail loudly than silently call the wrong model.
    """
    if target is LLMProvider.openai:
        return _ANTHROPIC_TO_OPENAI.get(model, model)
    return _OPENAI_TO_ANTHROPIC.get(model, model)


# --------------------------------------------------------------------------- #
# Public client
# --------------------------------------------------------------------------- #


class LLMClient:
    """Unified, retrying, fallback-capable LLM client.

    Reads provider/model defaults from `settings.llm`. Override per-call via
    the `model`, `max_tokens`, `temperature` kwargs of `complete()`.

    Example:
        client = LLMClient()
        # Cheap classifier call:
        verdict = client.complete(prompt, model=settings.llm.classifier_model)
        # Expensive summarization call (uses settings.llm.model by default):
        summary = client.complete(prompt, system="You are a strict editor.")
    """

    def __init__(self) -> None:
        cfg = settings.llm
        self._primary: LLMProvider = cfg.provider
        self._fallback: LLMProvider = (
            LLMProvider.openai
            if cfg.provider is LLMProvider.anthropic
            else LLMProvider.anthropic
        )
        self._default_model: str = cfg.model
        self._default_max_tokens: int = cfg.max_tokens
        self._default_temperature: float = cfg.temperature
        self._timeout: int = cfg.timeout_seconds

        # Lazy SDK clients — built on first use, NOT at import time.
        # This keeps tests fast and lets the orchestrator construct LLMClient
        # even when only one provider's API key is configured.
        self._anthropic_sdk: anthropic.Anthropic | None = None
        self._openai_sdk: openai.OpenAI | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send `prompt` to the primary provider; on terminal failure,
        retry the request with the fallback provider before giving up.

        Args:
            prompt:        User message.
            system:        Optional system prompt.
            model:         Override `settings.llm.model` (e.g. for the cheap
                           classifier model on filter calls).
            max_tokens:    Override `settings.llm.max_tokens`.
            temperature:   Override `settings.llm.temperature`.

        Returns:
            `LLMResponse` from whichever provider succeeded.

        Raises:
            LLMError: When both primary and fallback fail. The primary
                exception is chained via `__cause__` for traceability.
        """
        target_model = model or self._default_model
        max_tok = max_tokens or self._default_max_tokens
        temp = (
            temperature if temperature is not None else self._default_temperature
        )

        # ---- Primary attempt (with its own retry budget) ----
        try:
            return self._dispatch(
                self._primary, prompt, system, target_model, max_tok, temp
            )
        except Exception as primary_exc:
            log.error(
                "Primary provider '{}' exhausted retries: {}: {}",
                self._primary.value,
                type(primary_exc).__name__,
                primary_exc,
            )

            # ---- Fallback attempt (independent retry budget) ----
            fallback_model = _translate_model(target_model, target=self._fallback)
            log.warning(
                "Falling back to '{}' with model '{}'",
                self._fallback.value,
                fallback_model,
            )
            try:
                return self._dispatch(
                    self._fallback, prompt, system, fallback_model, max_tok, temp
                )
            except Exception as fallback_exc:
                log.error(
                    "Fallback provider '{}' also failed: {}: {}",
                    self._fallback.value,
                    type(fallback_exc).__name__,
                    fallback_exc,
                )
                raise LLMError(
                    f"Both providers failed. "
                    f"Primary ({self._primary.value}): {primary_exc!r}; "
                    f"Fallback ({self._fallback.value}): {fallback_exc!r}"
                ) from primary_exc

    # ------------------------------------------------------------------ #
    # Internal: dispatch to the right provider implementation
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        provider: LLMProvider,
        prompt: str,
        system: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        if provider is LLMProvider.anthropic:
            return self._call_anthropic(prompt, system, model, max_tokens, temperature)
        return self._call_openai(prompt, system, model, max_tokens, temperature)

    # ------------------------------------------------------------------ #
    # Anthropic call (decorated with retry)
    # ------------------------------------------------------------------ #

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_ANTHROPIC),
        before_sleep=_log_before_sleep,
    )
    def _call_anthropic(
        self,
        prompt: str,
        system: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        client = self._anthropic_client()
        log.debug(
            "→ Anthropic ({}, max_tokens={}, temperature={})",
            model, max_tokens, temperature,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        msg = client.messages.create(**kwargs)

        # Anthropic returns content as a list of typed blocks. The first
        # text block is what we want; fall back to "" if the response is empty.
        text = ""
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break

        return LLMResponse(
            text=text,
            model=model,
            provider="anthropic",
            tokens_in=msg.usage.input_tokens,
            tokens_out=msg.usage.output_tokens,
            finish_reason=msg.stop_reason,
        )

    # ------------------------------------------------------------------ #
    # OpenAI call (decorated with retry)
    # ------------------------------------------------------------------ #

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_OPENAI),
        before_sleep=_log_before_sleep,
    )
    def _call_openai(
        self,
        prompt: str,
        system: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        client = self._openai_client()
        log.debug(
            "→ OpenAI ({}, max_tokens={}, temperature={})",
            model, max_tokens, temperature,
        )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = completion.choices[0]
        usage = completion.usage

        return LLMResponse(
            text=choice.message.content or "",
            model=model,
            provider="openai",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )

    # ------------------------------------------------------------------ #
    # Lazy SDK client builders — instantiate on first use, never at import
    # ------------------------------------------------------------------ #

    def _anthropic_client(self) -> anthropic.Anthropic:
        if self._anthropic_sdk is None:
            api_key = settings.anthropic_api_key
            if api_key is None:
                raise LLMError(
                    "Anthropic API key is not configured. Set ANTHROPIC_API_KEY in .env."
                )
            self._anthropic_sdk = anthropic.Anthropic(
                api_key=api_key.get_secret_value(),
                timeout=float(self._timeout),
            )
        return self._anthropic_sdk

    def _openai_client(self) -> openai.OpenAI:
        if self._openai_sdk is None:
            api_key = settings.openai_api_key
            if api_key is None:
                raise LLMError(
                    "OpenAI API key is not configured. Set OPENAI_API_KEY in .env."
                )
            self._openai_sdk = openai.OpenAI(
                api_key=api_key.get_secret_value(),
                timeout=float(self._timeout),
            )
        return self._openai_sdk

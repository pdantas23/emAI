"""Executive-grade email summarization for WhatsApp delivery.

This module sits **after** the classifier in the pipeline. By the time we
get here, the orchestrator has already decided the email is worth a Sonnet
call. Our job is to convert one `RawEmail` into a tight, executive-style
briefing that lands well on a phone screen.

Architecture mirrors the classifier intentionally:

    LLM (strict JSON contract) → Pydantic model → format_for_whatsapp(...)

Why split summarization from formatting?

1. **Testability.** We can assert on what the model said (the three structured
   fields) without coupling the test to the exact emoji a designer might
   tweak next week.

2. **Multiple delivery channels.** Tomorrow we may want to email the same
   briefing to a board, post it to Slack, or render it on a dashboard.
   The summary is the data; the formatter is one of several views.

3. **Prompt iteration without churning the renderer.** When we tweak the
   prompt to make `contexto` more concise, the WhatsApp template doesn't
   move.

Cost model: Sonnet, ~6000-char body cap, max_tokens=600. A typical
summarization runs around 1-2k input + 200-400 output tokens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from config.settings import PROJECT_ROOT, settings
from src.ai.classifier import Classification, Priority
from src.ai.llm_client import LLMClient, LLMError
from src.email_client.base import RawEmail
from src.email_client.parser import extract_text_from_email
from src.utils.logger import log

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_PROMPT_PATH: Path = PROJECT_ROOT / "prompts" / "summarize_executive.md"

# Sonnet handles long context well, but cost scales linearly. 6000 chars is
# enough for very long threads while keeping the per-email cost predictable.
_BODY_CHAR_LIMIT: int = 6000

# Prompt explicitly caps each field; 600 tokens is plenty of slack and still
# bounds the worst-case spend.
_MAX_TOKENS: int = 600

# A touch of temperature gives the prose a less robotic feel without ever
# threatening determinism on facts (the prompt forbids invention).
_TEMPERATURE: float = 0.3

# Same tolerant-parse helpers as the classifier. We can't share them across
# modules without creating circular friction; copy is cheaper than abstraction
# for two call sites.
_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# WhatsApp visual vocabulary. Keep these in ONE place so the look-and-feel
# is consistent across every message we send.
_PRIORITY_BADGE: dict[Priority, str] = {
    Priority.high:   "🔴 *URGENTE*",
    Priority.medium: "🟡 *Atenção*",
    Priority.low:    "🟢 *Informativo*",
}

# WhatsApp doesn't render `---` as an <hr>, but it DOES render this kind of
# Unicode bar nicely on both iOS and Android — we use it to separate emails
# in a multi-email digest.
_SEPARATOR: str = "━━━━━━━━━━━━━━━━━━"


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #


class EmailSummary(BaseModel):
    """The three executive fields the LLM must return.

    Length caps mirror the prompt. Pydantic enforcing them here means a
    misbehaving model triggers a `ValidationError` we can catch and log,
    instead of silently shipping a 2000-char "summary" to WhatsApp.
    """

    resumo: str = Field(..., min_length=1, max_length=200)
    contexto: str = Field(..., min_length=1, max_length=800)
    acao: str = Field(..., min_length=1, max_length=300)

    model_config = {"extra": "forbid"}


class SummaryFailure(Exception):
    """Raised when the summarizer cannot produce a usable summary.

    Unlike the classifier (which fails open with a safe default), the
    summarizer fails LOUD: a half-baked summary on WhatsApp is worse than
    no message at all. The orchestrator decides what to do — typically:
    skip this email, log the failure, and move on without marking the
    email as seen so the next run gets another shot.
    """


# --------------------------------------------------------------------------- #
# Summarizer
# --------------------------------------------------------------------------- #


class EmailSummarizer:
    """Generate a structured executive summary using the high-quality model.

    Defaults to `settings.llm.model` (Sonnet) — the expensive one — because
    by construction we only reach this code path when the cheap classifier
    has already approved the email.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        prompt_path: Path | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self._model = model or settings.llm.model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        email: RawEmail,
        *,
        priority: Priority = Priority.medium,
    ) -> EmailSummary:
        """Produce an `EmailSummary` for the given email.

        `priority` is forwarded into the prompt so the LLM knows how much
        urgency to bake into the language (e.g. an `high` email's `acao`
        field tends to come back more imperative). It does NOT affect
        which fields are returned.

        Raises:
            SummaryFailure: when the LLM call fails outright or the model's
                output cannot be parsed. The orchestrator should treat this
                as a per-email skip, not a global outage.
        """
        prompt = self._build_prompt(email, priority=priority)

        try:
            response = self._llm.complete(
                prompt,
                model=self._model,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
        except LLMError as exc:
            log.error(
                "Summarizer LLM call failed for uid={} ({}): {}",
                email.uid, type(exc).__name__, exc,
            )
            raise SummaryFailure(
                f"LLM call failed for uid={email.uid}: {exc}"
            ) from exc

        summary = self._parse_response(response.text, uid=email.uid)
        log.info(
            "Summarized uid={} (model={}, tokens={}/{})",
            email.uid, response.model, response.tokens_in, response.tokens_out,
        )
        return summary

    # ------------------------------------------------------------------ #
    # Prompt assembly — same `str.replace` trick as the classifier so the
    # markdown braces in the JSON example don't break str.format.
    # ------------------------------------------------------------------ #

    def _build_prompt(self, email: RawEmail, *, priority: Priority) -> str:
        body = extract_text_from_email(email, strip_signature=True)
        body_truncated = body[:_BODY_CHAR_LIMIT]
        if len(body) > _BODY_CHAR_LIMIT:
            body_truncated += "\n\n[...truncado por limite de tokens...]"

        sender_full = (
            f"{email.sender_name} <{email.sender_email}>"
            if email.sender_name
            else email.sender_email
        )

        return (
            self._prompt_template
            .replace("{{SENDER}}", sender_full)
            .replace("{{SUBJECT}}", email.subject or "(sem assunto)")
            .replace("{{PRIORITY}}", priority.value)
            .replace("{{BODY}}", body_truncated or "(corpo vazio)")
        )

    # ------------------------------------------------------------------ #
    # Response parsing — strict, with the same fence/preamble tolerances
    # the classifier uses.
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: str, *, uid: str) -> EmailSummary:
        if not raw or not raw.strip():
            raise SummaryFailure(f"Empty response from summarizer for uid={uid}")

        cleaned = _FENCE_RE.sub("", raw).strip()

        json_text = cleaned
        try:
            json.loads(json_text)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(cleaned)
            if not match:
                raise SummaryFailure(
                    f"No JSON object in summarizer output for uid={uid}: {raw[:200]!r}"
                )
            json_text = match.group(0)

        try:
            return EmailSummary.model_validate_json(json_text)
        except ValidationError as exc:
            raise SummaryFailure(
                f"Schema validation failed for uid={uid}: "
                f"{exc.errors(include_url=False)} — raw={raw[:200]!r}"
            ) from exc


# --------------------------------------------------------------------------- #
# WhatsApp Markdown rendering
# --------------------------------------------------------------------------- #


def format_for_whatsapp(
    email: RawEmail,
    summary: EmailSummary,
    classification: Classification,
) -> str:
    """Render one email's summary as a single WhatsApp message.

    Important formatting notes for WhatsApp Markdown:
      - Bold uses *single asterisks*, NOT **double** (double renders literal).
      - Italics use _underscores_.
      - Newlines render as <br>; double newline becomes paragraph break.
      - Emojis render natively on every modern WhatsApp client.

    The layout is:

        🔴 *URGENTE*
        ━━━━━━━━━━━━━━━━━━
        ✉️ *De:* João Silva <joao@x.com>
        📌 *Assunto:* Renovação de contrato

        💡 *Resumo*
        Frase de até 140 caracteres aqui.

        🧭 *Contexto*
        Duas a quatro frases factuais aqui.

        ⚡ *Ação Necessária*
        Verbo no infinitivo + objeto direto.
    """
    badge = _PRIORITY_BADGE[classification.priority]
    sender_display = (
        f"{email.sender_name} <{email.sender_email}>"
        if email.sender_name
        else email.sender_email
    )
    subject = email.subject or "(sem assunto)"

    parts = [
        badge,
        _SEPARATOR,
        f"✉️ *De:* {sender_display}",
        f"📌 *Assunto:* {subject}",
        "",
        "💡 *Resumo*",
        summary.resumo,
        "",
        "🧭 *Contexto*",
        summary.contexto,
        "",
        "⚡ *Ação Necessária*",
        summary.acao,
    ]
    return "\n".join(parts)


def format_digest(messages: list[str]) -> str:
    """Join multiple per-email messages into a single WhatsApp digest.

    Adds a visible separator between emails so the recipient can scan the
    block without losing track of where one briefing ends and the next
    begins. We do NOT add a leading separator — the first email's badge
    already serves as the visual anchor.
    """
    if not messages:
        return ""
    glue = f"\n\n{_SEPARATOR}\n\n"
    return glue.join(messages)

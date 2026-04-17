"""Cheap LLM-based relevance/priority classifier.

This is the **cost gate** of the pipeline. Every unread email is run through
this classifier before the orchestrator decides whether to invoke the (much
more expensive) summarizer. The classifier uses the model configured in
`settings.llm.classifier_model` — by default `claude-haiku-4-5-20251001`,
which is roughly an order of magnitude cheaper than Sonnet.

Design principles:

1. **Strict JSON contract.** The prompt commands the model to return a single
   JSON object with `relevance` (bool), `priority` ("low"|"medium"|"high"),
   and `reason` (≤120 chars). We parse it with Pydantic so any drift fails
   loudly here instead of polluting the rest of the pipeline.

2. **Tolerant parsing.** Some models stubbornly wrap their JSON in ```json```
   fences or add a leading sentence. We strip fences and extract the first
   `{...}` block via regex before calling `model_validate_json`.

3. **Safe default on failure.** If parsing or the LLM call fails entirely,
   we return `Classification(relevance=True, priority=medium, ...)`. The
   philosophy: an unfiltered email costs one Sonnet call; a dropped email
   may cost a missed business opportunity. Always prefer the former.

4. **Body truncation.** We send at most `_BODY_CHAR_LIMIT` characters of the
   email body to the classifier. The first ~1500 chars almost always contain
   enough signal to decide relevance, and capping protects us from sending
   100KB newsletters into a "cheap" model that suddenly becomes expensive.

The orchestrator usage pattern is intentionally minimal:

    classifier = EmailClassifier()
    if classifier.is_relevant(email):
        summary = summarizer.summarize(email)
        ...
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from config.settings import PROJECT_ROOT
from src.ai.llm_client import LLMClient, LLMError
from src.email_client.base import RawEmail
from src.email_client.parser import extract_text_from_email
from src.utils.logger import log

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_PROMPT_PATH: Path = PROJECT_ROOT / "prompts" / "classify_priority.md"

# How many characters of the email body to ship to the classifier.
# Empirically, the relevance signal is in the first paragraph; 1500 chars is
# enough for a polite greeting + opening + a hint of the ask, while keeping
# the per-call cost bounded for newsletters / forwarded threads / etc.
_BODY_CHAR_LIMIT: int = 1500

# The classifier should answer in a few dozen tokens. 200 leaves slack for
# multilingual `reason` fields without ever paying for a runaway response.
_MAX_TOKENS: int = 200

# Determinism matters here: same email → same verdict, every time.
_TEMPERATURE: float = 0.0

# Regex to extract the first JSON object from a possibly-fenced response.
# Non-greedy so we stop at the first balanced-looking brace; we then let
# Pydantic do the real validation.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Marker we look for to strip ```json ... ``` fences (common GPT habit even
# when the prompt forbids them).
_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Output schema — shared with the rest of the pipeline.
# --------------------------------------------------------------------------- #


class Priority(str, Enum):
    """Three-level priority. Matches the values the prompt is allowed to emit."""

    low = "low"
    medium = "medium"
    high = "high"


class Classification(BaseModel):
    """Structured classifier verdict.

    `model_config` rejects unknown fields so a model that hallucinates extra
    keys (e.g. "category") triggers a parse error rather than silently
    discarding data.
    """

    relevance: bool = Field(..., description="True if the email deserves summarization")
    priority: Priority = Field(..., description="Bucket used by the WhatsApp formatter")
    reason: str = Field(
        "",
        max_length=240,
        description="Short justification — preserved for logging and audits",
    )

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


class EmailClassifier:
    """Wraps the cheap LLM call that decides if an email is worth summarizing.

    Construction is cheap and side-effect free: the prompt template is read
    once from disk, but the LLM SDKs are still lazy (built on first use).
    Inject a custom `LLMClient` in tests to assert on call arguments without
    hitting the network.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        prompt_path: Path | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm if llm is not None else LLMClient()
        self._prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self._model = model or getattr(self._llm, "_classifier_model", "claude-haiku-4-5-20251001")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(self, email: RawEmail) -> Classification:
        """Return the LLM's relevance + priority verdict for a single email.

        Never raises on malformed model output: on any failure we fall back
        to a safe `relevance=True, priority=medium` so the orchestrator
        defaults to summarizing rather than silently dropping the email.
        """
        prompt = self._build_prompt(email)

        try:
            response = self._llm.complete(
                prompt,
                model=self._model,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
        except LLMError as exc:
            log.error(
                "Classifier LLM call failed for uid={} ({}): {}",
                email.uid, type(exc).__name__, exc,
            )
            return self._safe_default("LLM call failed")

        verdict = self._parse_response(response.text, uid=email.uid)
        log.info(
            "Classified uid={} → relevance={} priority={} (model={}, tokens={}/{})",
            email.uid,
            verdict.relevance,
            verdict.priority.value,
            response.model,
            response.tokens_in,
            response.tokens_out,
        )
        return verdict

    def is_relevant(self, email: RawEmail) -> bool:
        """Convenience boolean for the orchestrator's `if`-gate."""
        return self.classify(email).relevance

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #

    def _build_prompt(self, email: RawEmail) -> str:
        """Render the prompt template with the email's sender, subject and body.

        We use plain `str.replace` (not `str.format`) because the markdown
        template contains literal `{` characters in its JSON example and
        `format` would choke on them.
        """
        body = extract_text_from_email(email, strip_signature=True)
        body_truncated = body[:_BODY_CHAR_LIMIT]
        if len(body) > _BODY_CHAR_LIMIT:
            body_truncated += "\n\n[...truncated...]"

        sender = email.sender_name or email.sender_email
        sender_full = f"{sender} <{email.sender_email}>" if email.sender_name else email.sender_email

        return (
            self._prompt_template
            .replace("{{SENDER}}", sender_full)
            .replace("{{SUBJECT}}", email.subject or "(sem assunto)")
            .replace("{{BODY}}", body_truncated or "(corpo vazio)")
        )

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: str, *, uid: str) -> Classification:
        """Parse the model's JSON output into a `Classification`.

        Handles three real-world failure modes we've seen during testing:
          1. Stray ```json``` fences around the object.
          2. A leading apology / explanation before the JSON.
          3. Extra fields ("category", "tags", ...) the model invents.

        Anything truly unparseable returns the safe default so the orchestrator
        keeps making forward progress.
        """
        if not raw or not raw.strip():
            log.warning("Classifier returned empty body for uid={}", uid)
            return self._safe_default("empty response")

        cleaned = _FENCE_RE.sub("", raw).strip()

        # First try the cleaned text directly; fall back to extracting the
        # first {...} block if there's leading prose.
        json_text = cleaned
        try:
            json.loads(json_text)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(cleaned)
            if not match:
                log.warning(
                    "Classifier output had no JSON object for uid={}: {!r}",
                    uid, raw[:200],
                )
                return self._safe_default("no JSON found")
            json_text = match.group(0)

        try:
            return Classification.model_validate_json(json_text)
        except ValidationError as exc:
            log.warning(
                "Classifier output failed schema validation for uid={}: {} — raw={!r}",
                uid, exc.errors(include_url=False), raw[:200],
            )
            return self._safe_default("schema validation failed")

    # ------------------------------------------------------------------ #
    # Safe default
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_default(reason: str) -> Classification:
        """Fail-open verdict: when in doubt, let the summarizer have a look.

        Why `medium` instead of `high`? `high` would force the orchestrator
        to surface the email immediately (push notification / interrupt-style
        delivery), and we don't want a parse failure to spam the recipient.
        `medium` gets the email summarized and included in the next batch.
        """
        return Classification(
            relevance=True,
            priority=Priority.medium,
            reason=f"Fallback: {reason}",
        )

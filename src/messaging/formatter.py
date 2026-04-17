"""WhatsApp transport-layer formatting: truncation, chunking.

This module is **complementary** to `src/ai/summarizer.py::format_for_whatsapp`.
The split is intentional and follows the same data/presentation/transport
layering we use everywhere else in this codebase:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ src/ai/summarizer.py                                                │
  │   • EmailSummary (data)                                             │
  │   • format_for_whatsapp(...)  → semantic render (badges, sections)  │
  │   • format_digest(...)        → join several summaries with glue    │
  └────────────────────────┬────────────────────────────────────────────┘
                           │ produces well-formed strings of arbitrary length
                           ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ src/messaging/formatter.py  (you are here)                          │
  │   • truncate(...)             → safety cap for very large messages   │
  │   • chunk_messages(...)       → pack many messages into batches      │
  └────────────────────────┬────────────────────────────────────────────┘
                           │ produces a list of payloads
                           ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ src/messaging/evolution_api.py                                      │
  │   • EvolutionClient.send_many(payloads)                             │
  └─────────────────────────────────────────────────────────────────────┘

Evolution API does NOT enforce the 1600/4096-char limits that Twilio did.
We keep a generous 65536-char safety cap and optional chunking for
readability, but the practical limit is gone.
"""

from __future__ import annotations

# Safety cap for a single WhatsApp message. Evolution API has no hard limit,
# but no sane briefing should exceed this.
WHATSAPP_MAX_CHARS: int = 65_536

# What we append when a single message is longer than the cap. Kept short so
# the truncation eats as little real content as possible.
_TRUNCATION_TAIL: str = "\n\n…[mensagem truncada]"

# Default glue between briefings inside one chunk. Mirrors the visual bar
# already used by the summarizer's digest formatter so the look is uniform
# regardless of which layer assembled the chunk.
DEFAULT_SEPARATOR: str = "\n\n" + ("━" * 18) + "\n\n"


def truncate(message: str, *, max_chars: int = WHATSAPP_MAX_CHARS) -> str:
    """Return `message` capped at `max_chars`, with a clear truncation marker.

    The cap includes the trailing marker so the result is GUARANTEED to fit.
    If the input is already short enough, it's returned unchanged (no marker
    added — we only signal truncation when we actually had to cut).
    """
    if len(message) <= max_chars:
        return message

    keep = max_chars - len(_TRUNCATION_TAIL)
    if keep <= 0:
        # Pathological case: max_chars smaller than the marker itself.
        # Just return the marker prefix so we never exceed the limit.
        return _TRUNCATION_TAIL[:max_chars]
    return message[:keep] + _TRUNCATION_TAIL


def chunk_messages(
    messages: list[str],
    *,
    max_chars: int = WHATSAPP_MAX_CHARS,
    separator: str = DEFAULT_SEPARATOR,
) -> list[str]:
    """Pack already-rendered messages into the smallest number of WhatsApp-sized
    payloads, never breaking an individual message across two payloads.

    Algorithm — greedy bin-packing in original order:
      1. Walk `messages` left-to-right.
      2. If adding the next message (with separator) would overflow the
         current chunk, close the chunk and start a new one.
      3. If a SINGLE message exceeds `max_chars`, it gets its own chunk
         after being passed through `truncate(...)`. Truncating a giant
         summary is ugly but better than refusing to deliver it.

    Returns an empty list when given an empty list (the caller can decide
    whether sending nothing is OK or an error).

    Order is preserved — important when the orchestrator passes in priority-
    sorted briefings (urgent first).
    """
    if not messages:
        return []

    chunks: list[str] = []
    current: str = ""
    sep_len = len(separator)

    for raw in messages:
        # Step 1: if THIS message alone is too big, ship the current chunk
        # first, then place the truncated giant on its own.
        if len(raw) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(truncate(raw, max_chars=max_chars))
            continue

        # Step 2: try to fit it in the current chunk.
        if not current:
            # Empty chunk — just take it.
            current = raw
            continue

        projected = len(current) + sep_len + len(raw)
        if projected <= max_chars:
            current = current + separator + raw
        else:
            # Doesn't fit — flush and start a new chunk with this message.
            chunks.append(current)
            current = raw

    if current:
        chunks.append(current)

    return chunks

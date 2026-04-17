"""HTML → clean text conversion and email body extraction.

This module is intentionally pure (no I/O, no LLM calls). Given a `RawEmail`,
it returns the best plain-text representation suitable for downstream LLM input,
while stripping noise (scripts, tracking pixels, multi-blank-lines, etc.).
"""

from __future__ import annotations

import re
from email.utils import parseaddr

import html2text
from bs4 import BeautifulSoup

from src.email_client.base import RawEmail
from src.utils.logger import log

# --------------------------------------------------------------------------- #
# Module-level helpers (compiled once)
# --------------------------------------------------------------------------- #

# Collapse runs of >2 blank lines into exactly one blank line.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Standard email signature delimiter (RFC 3676 §4.3): "-- " on its own line.
# Everything below it is the signature; we strip it on demand.
_SIGNATURE_RE = re.compile(r"\n-- ?\n.*", re.DOTALL)

# Tags that are pure noise in an email body.
_NOISE_TAGS = ("script", "style", "head", "meta", "link", "noscript", "title")


def _make_html_converter() -> html2text.HTML2Text:
    """Configure html2text for LLM-friendly output.

    Decisions:
    - body_width=0:        do not wrap lines (LLM doesn't care; wrapping breaks URLs)
    - ignore_links=False:  KEEP urls — the LLM benefits from seeing where links go
    - ignore_images=True:  drop image markup (we already strip <img> trackers separately)
    """
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.unicode_snob = True
    h.skip_internal_links = True
    h.escape_snob = True
    return h


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def clean_html_to_text(html: str) -> str:
    """Convert an HTML email body into clean Markdown-ish plain text.

    Pipeline:
      1. BeautifulSoup strips <script>, <style>, <head>, etc.
      2. 1x1 tracking pixels are removed.
      3. html2text converts what's left to Markdown.
      4. Whitespace is normalized.

    Returns an empty string if the input is empty or unparseable.
    """
    if not html or not html.strip():
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag_name in _NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        # Drop 1x1 tracking pixels (common in marketing emails).
        for img in soup.find_all("img"):
            w = str(img.get("width", "")).strip()
            h_attr = str(img.get("height", "")).strip()
            if w == "1" and h_attr == "1":
                img.decompose()
        cleaned_html = str(soup)
    except Exception as exc:
        # Never let parsing kill the pipeline — fall back to raw HTML.
        log.warning("BeautifulSoup parse failed ({}); using raw HTML", exc)
        cleaned_html = html

    text = _make_html_converter().handle(cleaned_html)
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()
    return text


def extract_text_from_email(email: RawEmail, *, strip_signature: bool = False) -> str:
    """Return the best plain-text representation of an email body.

    Strategy:
    - Prefer `text/plain` when present and non-empty (cheapest, cleanest).
    - Fall back to converting `text/html` via `clean_html_to_text`.
    - Return empty string if the email has no body at all.

    Set `strip_signature=True` to drop everything after the standard
    "-- " signature delimiter (useful before sending to LLM to save tokens).
    """
    if email.body_text and email.body_text.strip():
        text = email.body_text
    elif email.body_html:
        text = clean_html_to_text(email.body_html)
    else:
        return ""

    if strip_signature:
        text = _SIGNATURE_RE.sub("", text)

    return text.strip()


def parse_sender(raw_from: str) -> tuple[str | None, str]:
    """Parse a `From:` header value into `(display_name, email)`.

    >>> parse_sender("João Silva <joao@example.com>")
    ('João Silva', 'joao@example.com')
    >>> parse_sender("plain@example.com")
    (None, 'plain@example.com')
    """
    name, addr = parseaddr(raw_from)
    return (name or None, addr.lower())

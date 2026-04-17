# emAI

AI-powered email triage agent: reads unread emails via IMAP, summarizes them with an LLM, and delivers an executive report to WhatsApp.

## Quickstart

```bash
# 1. Clone and enter
cd emAI

# 2. Install (Python 3.11+)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# Edit .env with real credentials

# 4. Validate config (will fail loudly if anything is missing)
python -c "from config.settings import settings; print('OK', settings.app_env)"
```

## Project Layout

```
src/
├── core/          # Orchestrator + custom exceptions
├── email_client/  # IMAP fetch + MIME parsing
├── filters/       # Pre-LLM filters (cheap → expensive)
├── ai/            # LLM client + summarizer + classifier
├── messaging/     # WhatsApp delivery (Twilio)
├── storage/       # SQLite state (processed emails)
└── utils/         # Logger, retry helpers

config/    # Pydantic Settings + YAML rules
prompts/   # Versioned LLM prompts (Markdown)
tests/     # unit/ + integration/ + fixtures (.eml)
logs/      # Rotating logs + state DB
```

## Security

- `.env` is gitignored. **Never commit credentials.**
- Gmail/Outlook: use **App Passwords**, not your real account password.
- Production: replace `.env` with AWS Secrets Manager / Doppler / Vault.

## Status

MVP under construction. See scope doc in chat history.
# emAIl

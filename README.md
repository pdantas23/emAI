# emAI

AI-powered email triage agent: reads unread emails via IMAP, summarizes them with an LLM, and delivers an executive report to WhatsApp.

## Architecture

```
Admin Provisiona          Usuario Configura          Engine Roda
(Streamlit Admin)         (Streamlit User)           (Orchestrator)
      |                         |                         |
      v                         v                         v
 user_credentials          user_settings            load_user_config()
 (API keys, AES-256)       (email, WhatsApp)         |
      |                         |                     v
      +----------+--------------+            UserRuntimeConfig
                 |                                    |
                 v                                    v
          Supabase Postgres                  build_orchestrator()
          (RLS enabled)                               |
                 ^                  +---------+-------+--------+--------+
                 |                  |         |       |        |        |
                 |                  v         v       v        v        v
        system_audit_logs        IMAP    Classifier Summarizer WhatsApp StateStore
        (append-only)           (Gmail)  (Haiku)   (Sonnet)   (Twilio) (Postgres)
```

Every credential mutation (create, update, delete) is recorded in
`system_audit_logs` with the actor, target user, and which fields changed
(field names only -- never the actual key values).

### Flow

1. **Admin Provisiona** -- via Streamlit Admin tab (password-protected), the admin
   registers API keys (Anthropic, Twilio, Gmail App Password) for each user.
   Keys are encrypted with AES-256-GCM before storage.

2. **Usuario Configura** -- via Streamlit User tab, the lawyer configures their
   Gmail address, WhatsApp number, and pipeline interval.

3. **Engine Roda** -- the CLI loads the user's config from the DB, builds the
   orchestrator with dependency-injected credentials, and runs the pipeline:
   fetch -> classify -> summarize -> send -> persist -> mark seen.

### Security Model

- **Zero .env for API keys.** The only environment variables are `DATABASE_URL`,
  `ENCRYPTION_KEY` (AES-256, 64 hex chars), and `ADMIN_PASSWORD`.
- **Encryption at rest.** All API keys stored in `user_credentials` are
  AES-256-GCM encrypted. The encryption key never reaches the database.
- **RLS on all tables.** `anon`/`authenticated` roles see zero rows.
  The backend uses `service_role` (Supabase) or a dedicated Postgres role.
- **Privacy contract.** No email content (subject, body, sender, summary) is
  persisted. Only metadata: message_id, uid, relevance, priority, delivery_status.
- **Secrets in memory only.** Decrypted keys live in `UserRuntimeConfig` for the
  duration of the pipeline run, then are garbage-collected.

## Quickstart

```bash
# 1. Clone and enter
cd emAI

# 2. Install (Python 3.11+)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure bootstrap (minimal .env)
cat > .env << 'EOF'
DATABASE_URL=postgresql://postgres:your-password@your-host:5432/postgres
ENCRYPTION_KEY=<64-char-hex-string>
ADMIN_PASSWORD=your-admin-password
EOF

# Generate an encryption key:
python -c "from src.storage.crypto import generate_key; print(generate_key())"

# 4. Run Supabase migrations
# Apply supabase/migrations/20260416120000_init_processed_emails.sql
# Apply supabase/migrations/20260417120000_user_credentials.sql

# 5. Launch the dashboard
streamlit run src/ui/app.py

# 6. Provision a user (Admin tab), configure email (User tab)

# 7. Run the pipeline
python src/main.py --user-id philip --once
# Or scheduled:
python src/main.py --user-id philip --interval 15
```

## Project Layout

```
config/
  settings.py            # Bootstrap config (DATABASE_URL, ENCRYPTION_KEY only)
  runtime_settings.py    # Per-user UserRuntimeConfig loaded from DB

src/
  main.py                # CLI entrypoint (--user-id required)
  core/orchestrator.py   # Pipeline coordinator
  ai/
    llm_client.py        # Unified LLM client (DI: accepts API keys via constructor)
    classifier.py        # Haiku relevance/priority gate
    summarizer.py        # Sonnet executive summary
  email_client/
    base.py              # Abstract EmailClient + RawEmail model
    gmail_imap.py        # Gmail IMAP (DI: accepts credentials via constructor)
    parser.py            # MIME parsing
  messaging/
    whatsapp_twilio.py   # Twilio WhatsApp (DI: accepts credentials via constructor)
    formatter.py         # Message formatting + chunking
  storage/
    models.py            # SQLModel tables (ProcessedEmail)
    state.py             # StateStore (dedup + audit)
    credentials.py       # CredentialStore (encrypted CRUD + audit hooks)
    user_settings.py     # UserSettingsStore (email, WhatsApp, interval)
    crypto.py            # AES-256-GCM encrypt/decrypt
    audit.py             # AuditStore (append-only system_audit_logs)
  ui/
    app.py               # Streamlit multi-tab dashboard
    admin_tab.py         # Admin: provision API keys
    user_tab.py          # User: configure email/WhatsApp
    validators.py        # Live key validation (Anthropic, Twilio, IMAP)

supabase/migrations/     # SQL migrations (user_credentials, user_settings, processed_emails)
prompts/                 # Versioned LLM prompt templates
tests/                   # Integration tests
```

## Security Checklist

- [x] API keys encrypted at rest (AES-256-GCM)
- [x] RLS on all tables (deny-all for anon/authenticated)
- [x] No email content persisted (privacy contract)
- [x] Admin tab password-protected
- [x] IMAP connection test before saving user settings
- [x] Key validation before saving credentials
- [x] Decrypted secrets garbage-collected after pipeline run
- [ ] HTTPS for Streamlit in production (use reverse proxy)
- [ ] Rate limiting on Admin login attempts
- [x] Audit log for credential changes (system_audit_logs, append-only)

## Status

Multi-user admin-provisioned architecture. Ready for internal deployment.

-- =============================================================================
-- emAI — admin-provisioned multi-user schema
-- =============================================================================
--
-- Migration:   20260417120000_user_credentials
-- Target DB:   Supabase Postgres (>= 15)
-- Purpose:     Create the tables that back the Admin → User → Engine flow:
--              • `user_credentials`  — API keys (AES-256-GCM encrypted at app level)
--              • `user_settings`     — per-user runtime config (email, WhatsApp, interval)
--
-- Design notes
-- -----------------------------------------------------------------------------
-- • Credentials are encrypted BEFORE they reach Postgres. The DB stores opaque
--   BYTEA blobs; the ENCRYPTION_KEY never leaves the app process. This means
--   a database dump or a Supabase dashboard screenshot reveals nothing.
-- • `user_id` is the join key across both tables AND `processed_emails`.
--   It is a human-readable slug set by the Admin (e.g. "philip", "joao").
-- • RLS is ON with deny-all for anon/authenticated — same pattern as
--   `processed_emails`. The backend uses the `service_role` key.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- Table: public.user_credentials
-- -----------------------------------------------------------------------------
-- Stores infrastructure keys provisioned by the Admin. Every sensitive value
-- is AES-256-GCM encrypted at the application layer before INSERT.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.user_credentials (
    id                  BIGSERIAL    PRIMARY KEY,
    user_id             TEXT         NOT NULL,

    -- ---- Infrastructure keys (encrypted BYTEA) ------------------------------
    anthropic_key       BYTEA,
    openai_key          BYTEA,
    twilio_sid          BYTEA,
    twilio_token        BYTEA,
    twilio_number       VARCHAR(30),          -- whatsapp:+E164 (not sensitive)

    -- ---- Supabase project credentials (encrypted) ---------------------------
    supabase_url        TEXT,                 -- not sensitive (public URL)
    supabase_key        BYTEA,               -- encrypted service_role key

    -- ---- Gmail / IMAP (encrypted) -------------------------------------------
    gmail_app_password  BYTEA,

    -- ---- Audit trail ---------------------------------------------------------
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_by          TEXT,                 -- admin who last modified

    -- ---- Constraints ---------------------------------------------------------
    CONSTRAINT user_credentials_user_id_uniq UNIQUE (user_id)
);

-- Trigger to auto-update `updated_at` on every UPDATE.
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_credentials_updated_at_trigger
    BEFORE UPDATE ON public.user_credentials
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- -----------------------------------------------------------------------------
-- Table: public.user_settings
-- -----------------------------------------------------------------------------
-- User-facing configuration: which email to monitor, where to send WhatsApp
-- reports, and how often to run the pipeline.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.user_settings (
    id                      BIGSERIAL    PRIMARY KEY,
    user_id                 TEXT         NOT NULL,

    -- ---- IMAP settings -------------------------------------------------------
    email                   VARCHAR(320) NOT NULL,
    imap_host               VARCHAR(255) NOT NULL DEFAULT 'imap.gmail.com',
    imap_port               INT          NOT NULL DEFAULT 993,

    -- ---- WhatsApp routing ----------------------------------------------------
    whatsapp_to             VARCHAR(30)  NOT NULL,

    -- ---- Scheduling ----------------------------------------------------------
    run_interval_minutes    INT          NOT NULL DEFAULT 30,

    -- ---- Bookkeeping ---------------------------------------------------------
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- ---- Constraints ---------------------------------------------------------
    CONSTRAINT user_settings_user_id_uniq UNIQUE (user_id),
    CONSTRAINT user_settings_user_id_fk
        FOREIGN KEY (user_id) REFERENCES public.user_credentials(user_id)
        ON DELETE CASCADE,
    CONSTRAINT user_settings_interval_chk
        CHECK (run_interval_minutes >= 1 AND run_interval_minutes <= 1440),
    CONSTRAINT user_settings_whatsapp_chk
        CHECK (whatsapp_to LIKE 'whatsapp:+%')
);

CREATE TRIGGER user_settings_updated_at_trigger
    BEFORE UPDATE ON public.user_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- -----------------------------------------------------------------------------
-- Add user_id to processed_emails for multi-user partitioning
-- -----------------------------------------------------------------------------

ALTER TABLE public.processed_emails
    ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Index for per-user queries (dashboard, list_recent)
CREATE INDEX IF NOT EXISTS processed_emails_user_id_idx
    ON public.processed_emails (user_id);


-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS user_credentials_user_id_idx
    ON public.user_credentials (user_id);

CREATE INDEX IF NOT EXISTS user_settings_user_id_idx
    ON public.user_settings (user_id);


-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE public.user_credentials ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_credentials FORCE ROW LEVEL SECURITY;

ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_settings FORCE ROW LEVEL SECURITY;

-- Deny-all for public-facing roles (same pattern as processed_emails).

CREATE POLICY user_credentials_anon_deny
    ON public.user_credentials AS RESTRICTIVE
    FOR ALL TO anon
    USING (FALSE) WITH CHECK (FALSE);

CREATE POLICY user_credentials_authenticated_deny
    ON public.user_credentials AS RESTRICTIVE
    FOR ALL TO authenticated
    USING (FALSE) WITH CHECK (FALSE);

CREATE POLICY user_settings_anon_deny
    ON public.user_settings AS RESTRICTIVE
    FOR ALL TO anon
    USING (FALSE) WITH CHECK (FALSE);

CREATE POLICY user_settings_authenticated_deny
    ON public.user_settings AS RESTRICTIVE
    FOR ALL TO authenticated
    USING (FALSE) WITH CHECK (FALSE);


-- =============================================================================
-- Role grants (self-hosted blueprint)
-- =============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'emai_app') THEN
        GRANT SELECT, INSERT, UPDATE ON public.user_credentials TO emai_app;
        GRANT SELECT, INSERT, UPDATE ON public.user_settings    TO emai_app;
        GRANT USAGE ON SEQUENCE public.user_credentials_id_seq  TO emai_app;
        GRANT USAGE ON SEQUENCE public.user_settings_id_seq     TO emai_app;
    END IF;
END
$$;

REVOKE ALL ON public.user_credentials FROM PUBLIC;
REVOKE ALL ON public.user_settings    FROM PUBLIC;


COMMIT;

-- =============================================================================
-- Rollback (operator reference only)
-- =============================================================================
--  BEGIN;
--    ALTER TABLE public.processed_emails DROP COLUMN IF EXISTS user_id;
--    DROP TABLE IF EXISTS public.user_settings CASCADE;
--    DROP TABLE IF EXISTS public.user_credentials CASCADE;
--    DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
--  COMMIT;
-- =============================================================================

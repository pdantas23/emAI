-- =============================================================================
-- emAI — Evolution API migration (replaces Twilio)
-- =============================================================================
--
-- Migration:   20260417140000_evolution_api
-- Target DB:   Supabase Postgres (>= 15)
-- Purpose:     Replace Twilio credentials with Evolution API credentials
--              in the `user_credentials` table.
--
-- Changes:
--   - DROP: twilio_sid, twilio_token, twilio_number
--   - ADD:  evolution_url (TEXT), evolution_api_key (BYTEA, encrypted),
--           evolution_instance (VARCHAR(100))
--   - ALTER: processed_emails.twilio_sids → message_ids
--   - ALTER: user_settings.whatsapp_to — remove whatsapp:+ constraint
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- Drop Twilio columns from user_credentials
-- -----------------------------------------------------------------------------

ALTER TABLE public.user_credentials
    DROP COLUMN IF EXISTS twilio_sid,
    DROP COLUMN IF EXISTS twilio_token,
    DROP COLUMN IF EXISTS twilio_number;

-- -----------------------------------------------------------------------------
-- Add Evolution API columns
-- -----------------------------------------------------------------------------

ALTER TABLE public.user_credentials
    ADD COLUMN IF NOT EXISTS evolution_url        TEXT,
    ADD COLUMN IF NOT EXISTS evolution_api_key    BYTEA,
    ADD COLUMN IF NOT EXISTS evolution_instance   VARCHAR(100);

-- -----------------------------------------------------------------------------
-- Rename twilio_sids → message_ids in processed_emails
-- -----------------------------------------------------------------------------

ALTER TABLE public.processed_emails
    RENAME COLUMN twilio_sids TO message_ids;

-- -----------------------------------------------------------------------------
-- Remove whatsapp:+ constraint from user_settings (Evolution uses plain numbers)
-- -----------------------------------------------------------------------------

ALTER TABLE public.user_settings
    DROP CONSTRAINT IF EXISTS user_settings_whatsapp_chk;

COMMIT;

-- =============================================================================
-- Rollback (operator reference only)
-- =============================================================================
--  BEGIN;
--    ALTER TABLE public.user_credentials
--        DROP COLUMN IF EXISTS evolution_url,
--        DROP COLUMN IF EXISTS evolution_api_key,
--        DROP COLUMN IF EXISTS evolution_instance;
--    ALTER TABLE public.user_credentials
--        ADD COLUMN IF NOT EXISTS twilio_sid BYTEA,
--        ADD COLUMN IF NOT EXISTS twilio_token BYTEA,
--        ADD COLUMN IF NOT EXISTS twilio_number VARCHAR(30);
--    ALTER TABLE public.processed_emails
--        RENAME COLUMN message_ids TO twilio_sids;
--    ALTER TABLE public.user_settings
--        ADD CONSTRAINT user_settings_whatsapp_chk CHECK (whatsapp_to LIKE 'whatsapp:+%');
--  COMMIT;
-- =============================================================================

-- =============================================================================
-- emAI — system audit log
-- =============================================================================
--
-- Migration:   20260417130000_system_audit_logs
-- Target DB:   Supabase Postgres (>= 15)
-- Purpose:     Immutable audit trail for credential and settings changes.
--              Every write to `user_credentials` or `user_settings` by the
--              Admin panel (or any future admin tool) inserts a row here.
--
-- Design notes
-- -----------------------------------------------------------------------------
-- * This table is APPEND-ONLY. No UPDATE or DELETE is granted to the app role.
-- * `metadata_json` stores context (which fields changed) but NEVER stores
--   the actual credential values — only field names like "anthropic_key".
-- * `actor` is the admin who performed the action; `target_user_id` is the
--   user whose record was changed.
-- * RLS deny-all for anon/authenticated — same pattern as all other tables.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS public.system_audit_logs (
    id                  BIGSERIAL       PRIMARY KEY,
    actor               TEXT            NOT NULL,
    action              TEXT            NOT NULL,
    target_user_id      TEXT            NOT NULL,
    metadata_json       JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Index for querying by target user (dashboard / investigation)
CREATE INDEX IF NOT EXISTS audit_logs_target_user_idx
    ON public.system_audit_logs (target_user_id);

-- Index for querying by actor (who did what)
CREATE INDEX IF NOT EXISTS audit_logs_actor_idx
    ON public.system_audit_logs (actor);

-- Index for time-range queries (recent activity)
CREATE INDEX IF NOT EXISTS audit_logs_created_at_idx
    ON public.system_audit_logs (created_at DESC);


-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE public.system_audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_audit_logs FORCE ROW LEVEL SECURITY;

CREATE POLICY audit_logs_anon_deny
    ON public.system_audit_logs AS RESTRICTIVE
    FOR ALL TO anon
    USING (FALSE) WITH CHECK (FALSE);

CREATE POLICY audit_logs_authenticated_deny
    ON public.system_audit_logs AS RESTRICTIVE
    FOR ALL TO authenticated
    USING (FALSE) WITH CHECK (FALSE);


-- =============================================================================
-- Role grants — app role can INSERT + SELECT only (append-only)
-- =============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'emai_app') THEN
        GRANT SELECT, INSERT ON public.system_audit_logs    TO emai_app;
        GRANT USAGE ON SEQUENCE public.system_audit_logs_id_seq TO emai_app;
    END IF;
END
$$;

REVOKE ALL ON public.system_audit_logs FROM PUBLIC;

COMMIT;

-- =============================================================================
-- Rollback (operator reference only)
-- =============================================================================
--  BEGIN;
--    DROP TABLE IF EXISTS public.system_audit_logs CASCADE;
--  COMMIT;
-- =============================================================================

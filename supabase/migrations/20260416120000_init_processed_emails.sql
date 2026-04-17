-- =============================================================================
-- emAI — initial schema
-- =============================================================================
--
-- Migration:   20260416120000_init_processed_emails
-- Target DB:   Supabase Postgres (>= 15)
-- Purpose:     Create the single source of truth for the email pipeline's
--              dedup + audit trail, lock it down with RLS, and enforce the
--              enum invariants that the Python layer (SQLModel) only asks
--              nicely for.
--
-- Design notes
-- -----------------------------------------------------------------------------
-- • ONE table (`processed_emails`) — one row per email that reached a
--   terminal state (delivered OR skipped_irrelevant). The SQLModel mirror
--   of this schema lives in `src/storage/models.py`; if you change one
--   side, change the other.
-- • Dedup key is `message_id` (RFC-822). IMAP UIDs drift across folder
--   moves — Message-IDs do not.
-- • Privacy contract: this table stores ONLY metadata. No subject, no
--   sender, no classifier reason, no LLM summary. Those are processed in
--   memory by the app and dropped before the INSERT. If you ever need to
--   add a content column back, update BOTH this migration and the privacy
--   docstring in `src/storage/models.py` first — and think hard about RLS.
-- • Failed deliveries are deliberately NOT persisted. Absence of a row ⇒
--   the next pipeline run re-fetches and retries the email. This is the
--   symmetric counterpart of the app's fail-stop storage policy.
-- • RLS is ON. No permissive policies are created for `anon` / `authenticated`
--   ⇒ PostgREST calls with those keys return zero rows. The backend connects
--   with the `service_role` key (which bypasses RLS) or a direct Postgres
--   role with explicit grants below. This guards against accidental
--   exposure if someone ever points the Supabase auto-API at this table.
-- • CHECK constraints enforce the same enum domains as the Python Enums.
--   Belt + braces: a rogue migration or a misbehaving client still can't
--   insert `priority='urgent'` or `delivery_status='pending'`.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- Extensions — none required. Listed here for auditability.
-- -----------------------------------------------------------------------------
-- (pgcrypto / uuid-ossp are NOT needed: we use BIGSERIAL for the PK and
--  rely on the RFC-822 Message-ID for external identity.)


-- -----------------------------------------------------------------------------
-- Table: public.processed_emails
-- -----------------------------------------------------------------------------
-- Column order intentionally matches the SQLModel definition so a diff of
-- the two files makes schema drift obvious at review time.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.processed_emails (
    -- ---- Primary key & dedup index ----------------------------------------
    id                        BIGSERIAL PRIMARY KEY,

    -- RFC-2822 caps a header line at 998 chars. Real Message-IDs are ~60
    -- chars; this upper bound exists only as a sanity guardrail.
    message_id                VARCHAR(998) NOT NULL,

    -- IMAP UID at processing time. Not unique (can change across folder
    -- moves or server re-indexing), but we index it for quick lookups
    -- during on-call investigations ("which row did UID 42 become?").
    uid                       VARCHAR(64)  NOT NULL,

    -- ---- Classifier verdict (non-sensitive booleans/enums only) -----------
    relevance                 BOOLEAN      NOT NULL,
    priority                  VARCHAR(10)  NOT NULL,

    -- ---- Delivery audit ---------------------------------------------------
    delivery_status           VARCHAR(24)  NOT NULL,

    -- Comma-separated Twilio SIDs. SIDs match ^SM[A-Za-z0-9]{32}$ and never
    -- contain commas, so CSV is unambiguous. Keeps psql output scannable.
    twilio_sids               VARCHAR(2048),

    -- ---- Bookkeeping ------------------------------------------------------
    -- `NOW()` returns TIMESTAMPTZ at the session TZ; Postgres stores it as
    -- UTC internally. The Python layer passes an explicit tz-aware UTC
    -- datetime, so this DEFAULT is just a safety net for ad-hoc inserts.
    processed_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- ---- Domain invariants (CHECK constraints = DB-level guardrails) ------
    CONSTRAINT processed_emails_priority_chk
        CHECK (priority IN ('low', 'medium', 'high')),

    CONSTRAINT processed_emails_delivery_status_chk
        CHECK (delivery_status IN ('delivered', 'skipped_irrelevant')),

    -- Skipped emails have no WhatsApp send, so no SIDs — guard at DB level.
    CONSTRAINT processed_emails_sids_match_status_chk
        CHECK (
            delivery_status <> 'skipped_irrelevant'
            OR twilio_sids IS NULL
        )
);


-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------
-- `message_id` gets a UNIQUE index — this is the dedup guarantee. The UNIQUE
-- also implies an index, so `has_been_processed()` is O(log n) lookups.
-- -----------------------------------------------------------------------------

CREATE UNIQUE INDEX IF NOT EXISTS processed_emails_message_id_uniq
    ON public.processed_emails (message_id);

CREATE INDEX IF NOT EXISTS processed_emails_uid_idx
    ON public.processed_emails (uid);

-- DESC matches the `list_recent()` query which orders by processed_at DESC
-- — this gives us an index-only scan for the audit endpoint.
CREATE INDEX IF NOT EXISTS processed_emails_processed_at_desc_idx
    ON public.processed_emails (processed_at DESC);


-- -----------------------------------------------------------------------------
-- Column comments — readable via `\d+ processed_emails` in psql and in the
-- Supabase table editor. Keep terse and operator-focused.
-- -----------------------------------------------------------------------------

COMMENT ON TABLE  public.processed_emails                       IS 'emAI dedup + audit trail: one row per email that reached a terminal state.';
COMMENT ON COLUMN public.processed_emails.message_id            IS 'RFC-822 Message-ID — stable dedup key across IMAP servers.';
COMMENT ON COLUMN public.processed_emails.uid                   IS 'IMAP UID at processing time. NOT unique — can drift after folder moves.';
COMMENT ON COLUMN public.processed_emails.relevance             IS 'Classifier verdict. FALSE ⇒ summarizer was never called (token-saver gate).';
COMMENT ON COLUMN public.processed_emails.priority              IS 'low | medium | high — drives digest sort order in WhatsApp report.';
COMMENT ON COLUMN public.processed_emails.delivery_status       IS 'Terminal state: delivered (sent to WhatsApp) or skipped_irrelevant.';
COMMENT ON COLUMN public.processed_emails.twilio_sids           IS 'Comma-separated Twilio message SIDs; one row points at ALL chunks that carried this email.';
COMMENT ON COLUMN public.processed_emails.processed_at          IS 'UTC timestamp when the row was written (just before IMAP \Seen).';


-- =============================================================================
-- Row-Level Security
-- =============================================================================
-- Threat model
-- -----------------------------------------------------------------------------
-- • The backend writes and reads this table using a PRIVILEGED connection:
--     - On Supabase:   the `service_role` API key, which bypasses RLS.
--     - On bare metal: a dedicated Postgres role (`emai_app`) granted
--       INSERT/SELECT below, used exclusively by the emAI process.
-- • Anyone who queries via the PUBLIC Supabase URL with the `anon` or
--   `authenticated` keys MUST NOT see these rows — email subjects and
--   summaries are confidential.
-- • RLS is enabled and we DELIBERATELY CREATE NO PERMISSIVE POLICIES for
--   the `anon` / `authenticated` roles. PostgREST therefore returns zero
--   rows for those callers, which is exactly what we want.
-- =============================================================================

ALTER TABLE public.processed_emails ENABLE ROW LEVEL SECURITY;

-- `FORCE` makes RLS apply even to the table owner. Without it, a migration
-- or a psql session running as the owner role would silently bypass RLS and
-- leak data during ad-hoc investigation. We'd rather force explicit grants.
ALTER TABLE public.processed_emails FORCE ROW LEVEL SECURITY;

-- Explicit deny-all policies. These are technically redundant (RLS defaults
-- to deny when no policies match) but they document intent AND they survive
-- someone later adding a permissive policy by mistake: a RESTRICTIVE policy
-- is AND-ed with every permissive one, so adding a loose SELECT policy
-- would still be blocked by the restrictive below.
--
-- If/when we add a multi-tenant owner_id column, REPLACE these restrictive
-- policies with permissive `USING (owner_id = auth.uid())` policies.

CREATE POLICY processed_emails_anon_deny
    ON public.processed_emails
    AS RESTRICTIVE
    FOR ALL
    TO anon
    USING  (FALSE)
    WITH CHECK (FALSE);

CREATE POLICY processed_emails_authenticated_deny
    ON public.processed_emails
    AS RESTRICTIVE
    FOR ALL
    TO authenticated
    USING  (FALSE)
    WITH CHECK (FALSE);


-- =============================================================================
-- Role grants
-- =============================================================================
-- The `service_role` key bypasses RLS entirely — no grants needed on
-- Supabase. For a self-hosted Postgres you'd run the emAI process with a
-- dedicated role; the block below is the blueprint. We wrap it in DO so the
-- migration stays idempotent and non-fatal when the role doesn't exist
-- (e.g. pure Supabase deployments).
-- =============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'emai_app') THEN
        GRANT USAGE  ON SCHEMA public              TO emai_app;
        GRANT SELECT, INSERT ON public.processed_emails TO emai_app;
        GRANT USAGE  ON SEQUENCE public.processed_emails_id_seq TO emai_app;
    END IF;
END
$$;

-- Revoke everything from the default `public` role — belt + braces against
-- a future grant that might accidentally widen access.
REVOKE ALL ON public.processed_emails FROM PUBLIC;


COMMIT;

-- =============================================================================
-- Rollback (for operator reference — do NOT run automatically)
-- =============================================================================
--  BEGIN;
--    DROP TABLE IF EXISTS public.processed_emails CASCADE;
--  COMMIT;
-- =============================================================================

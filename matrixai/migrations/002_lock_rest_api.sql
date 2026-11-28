-- Supabase exposes every public table through its REST API. The app talks to
-- Postgres directly as the table owner (which bypasses row-level security),
-- so turning RLS on with no policies closes the REST door without changing
-- anything for the app.
ALTER TABLE users               ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiments         ENABLE ROW LEVEL SECURITY;
ALTER TABLE suggestion_history  ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log           ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiment_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys            ENABLE ROW LEVEL SECURITY;

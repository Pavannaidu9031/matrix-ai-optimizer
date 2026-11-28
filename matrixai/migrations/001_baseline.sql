-- Baseline schema. Safe on both the existing live database (every statement
-- is IF NOT EXISTS / conditional) and on a brand-new empty one.

CREATE TABLE IF NOT EXISTS users (
    email       VARCHAR PRIMARY KEY,
    name        VARCHAR,
    picture     VARCHAR,
    is_approved BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiments (
    id          BIGSERIAL PRIMARY KEY,
    user_email  VARCHAR NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Old tables called the intensity column wavelength_shift. Rename it only
-- when the new name doesn't exist yet (the live table has both).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = 'experiments'
                  AND column_name = 'wavelength_shift')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = 'experiments'
                  AND column_name = 'light_intensity_uw')
    THEN
        ALTER TABLE experiments RENAME COLUMN wavelength_shift TO light_intensity_uw;
    END IF;
END $$;

ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS target_material           VARCHAR DEFAULT 'Generic',
    ADD COLUMN IF NOT EXISTS branch_name               VARCHAR DEFAULT 'main',
    ADD COLUMN IF NOT EXISTS sputter_mode              VARCHAR DEFAULT 'RF',
    ADD COLUMN IF NOT EXISTS rf_power                  FLOAT,
    ADD COLUMN IF NOT EXISTS working_pressure          FLOAT,
    ADD COLUMN IF NOT EXISTS ar_flow                   FLOAT,
    ADD COLUMN IF NOT EXISTS o2_flow                   FLOAT,
    ADD COLUMN IF NOT EXISTS substrate_temp            FLOAT,
    ADD COLUMN IF NOT EXISTS target_distance           FLOAT,
    ADD COLUMN IF NOT EXISTS sputter_time_s            FLOAT,
    ADD COLUMN IF NOT EXISTS rotation_speed            FLOAT,
    ADD COLUMN IF NOT EXISTS substrate_type            VARCHAR DEFAULT 'Si Wafer',
    ADD COLUMN IF NOT EXISTS core_diameter_um          FLOAT,
    ADD COLUMN IF NOT EXISTS pd_thickness              FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS pd_dc_power               FLOAT,
    ADD COLUMN IF NOT EXISTS pd_distance_cm            FLOAT,
    ADD COLUMN IF NOT EXISTS pd_sputter_time_s         FLOAT,
    ADD COLUMN IF NOT EXISTS pd_pressure_mtorr         FLOAT,
    ADD COLUMN IF NOT EXISTS pd_ar_flow                FLOAT,
    ADD COLUMN IF NOT EXISTS pd_o2_flow                FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS film_thickness            FLOAT,
    ADD COLUMN IF NOT EXISTS xrd_phase                 VARCHAR,
    ADD COLUMN IF NOT EXISTS grain_size                FLOAT,
    ADD COLUMN IF NOT EXISTS light_intensity_uw        FLOAT,
    ADD COLUMN IF NOT EXISTS h2_response_time          FLOAT,
    ADD COLUMN IF NOT EXISTS recovery_time_s           FLOAT,
    ADD COLUMN IF NOT EXISTS response_curve            TEXT,
    ADD COLUMN IF NOT EXISTS batch_notes               TEXT,
    ADD COLUMN IF NOT EXISTS suggestion_id             BIGINT,
    ADD COLUMN IF NOT EXISTS suggestion_option         VARCHAR,
    ADD COLUMN IF NOT EXISTS predicted_intensity_uw    FLOAT,
    ADD COLUMN IF NOT EXISTS predicted_intensity_sigma FLOAT,
    ADD COLUMN IF NOT EXISTS predicted_phase           VARCHAR,
    ADD COLUMN IF NOT EXISTS followed_suggestion       BOOLEAN,
    ADD COLUMN IF NOT EXISTS quality_score             FLOAT;

-- Measurements can be genuinely unmeasured; older tables may have forced
-- a value, which is how placeholder numbers ended up in the data.
ALTER TABLE experiments
    ALTER COLUMN film_thickness     DROP NOT NULL,
    ALTER COLUMN grain_size         DROP NOT NULL,
    ALTER COLUMN light_intensity_uw DROP NOT NULL,
    ALTER COLUMN h2_response_time   DROP NOT NULL,
    ALTER COLUMN recovery_time_s    DROP NOT NULL,
    ALTER COLUMN quality_score      DROP NOT NULL;

CREATE INDEX IF NOT EXISTS experiments_user_branch_idx
    ON experiments (user_email, branch_name, created_at DESC);

CREATE TABLE IF NOT EXISTS suggestion_history (
    id         BIGSERIAL PRIMARY KEY,
    user_id    VARCHAR,
    user_email VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE suggestion_history
    ADD COLUMN IF NOT EXISTS suggested_rf_power     FLOAT,
    ADD COLUMN IF NOT EXISTS suggested_pressure     FLOAT,
    ADD COLUMN IF NOT EXISTS suggested_distance     FLOAT,
    ADD COLUMN IF NOT EXISTS suggested_thickness    FLOAT,
    ADD COLUMN IF NOT EXISTS suggested_rotation     FLOAT,
    ADD COLUMN IF NOT EXISTS suggested_ar_flow      FLOAT,
    ADD COLUMN IF NOT EXISTS suggested_pd_thickness FLOAT,
    ADD COLUMN IF NOT EXISTS predicted_xrd_score    FLOAT,
    ADD COLUMN IF NOT EXISTS predicted_wavelength   FLOAT,
    ADD COLUMN IF NOT EXISTS confidence_score       FLOAT,
    ADD COLUMN IF NOT EXISTS converged              BOOLEAN,
    ADD COLUMN IF NOT EXISTS kappa_used             FLOAT;

CREATE INDEX IF NOT EXISTS suggestion_history_user_idx ON suggestion_history (user_email, id DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id            BIGSERIAL PRIMARY KEY,
    action        VARCHAR NOT NULL,
    experiment_id BIGINT,
    user_email    VARCHAR,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_comments (
    id            BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT,
    user_email    VARCHAR,
    comment_text  TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    api_key_hash VARCHAR PRIMARY KEY,
    user_email   VARCHAR NOT NULL,
    usage_count  INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- ──────────────────────────────────────────────────────────────
-- Table 1: training_flights_base
-- Holds the original 365-day MUC backfill with settled actual
-- times. Populated once via import_base_training_data.py.
-- Never overwritten after initial import.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_flights_base (
    id                   BIGSERIAL PRIMARY KEY,
    movement             TEXT,
    number_raw           TEXT,
    call_sign            TEXT,
    status               TEXT,
    codeshare_status     TEXT,
    is_cargo             BOOLEAN,
    airline_iata         TEXT,
    airline_icao         TEXT,
    airline_name         TEXT,
    aircraft_model       TEXT,
    aircraft_modeS       TEXT,
    aircraft_reg         TEXT,
    other_airport_iata   TEXT,
    other_airport_icao   TEXT,
    dep_sched_utc        TIMESTAMPTZ,
    dep_rev_utc          TIMESTAMPTZ,
    dep_pred_utc         TIMESTAMPTZ,
    dep_runway_utc       TIMESTAMPTZ,
    arr_sched_utc        TIMESTAMPTZ,
    arr_rev_utc          TIMESTAMPTZ,
    arr_pred_utc         TIMESTAMPTZ,
    arr_runway_utc       TIMESTAMPTZ,
    dep_best_utc         TIMESTAMPTZ,
    arr_best_utc         TIMESTAMPTZ,
    dep_delay_min        DOUBLE PRECISION,
    arr_delay_min        DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_tfb_dep_sched
    ON training_flights_base (dep_sched_utc);
CREATE INDEX IF NOT EXISTS idx_tfb_arr_sched
    ON training_flights_base (arr_sched_utc);
CREATE INDEX IF NOT EXISTS idx_tfb_number_dep
    ON training_flights_base (number_raw, dep_sched_utc, movement);


-- ──────────────────────────────────────────────────────────────
-- Table 2: training_flights_backfill
-- One row per flight per retrain cycle covering new months.
-- Tagged with job_id so each job's contribution is traceable.
-- Accumulates across retrain runs — never truncated.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_flights_backfill (
    id                   BIGSERIAL PRIMARY KEY,
    job_id               INTEGER,
    flight_date          DATE,
    movement             TEXT,
    number_raw           TEXT,
    call_sign            TEXT,
    status               TEXT,
    codeshare_status     TEXT,
    is_cargo             BOOLEAN,
    airline_iata         TEXT,
    airline_icao         TEXT,
    airline_name         TEXT,
    aircraft_model       TEXT,
    aircraft_modeS       TEXT,
    aircraft_reg         TEXT,
    other_airport_iata   TEXT,
    other_airport_icao   TEXT,
    dep_sched_utc        TIMESTAMPTZ,
    dep_rev_utc          TIMESTAMPTZ,
    dep_pred_utc         TIMESTAMPTZ,
    dep_runway_utc       TIMESTAMPTZ,
    arr_sched_utc        TIMESTAMPTZ,
    arr_rev_utc          TIMESTAMPTZ,
    arr_pred_utc         TIMESTAMPTZ,
    arr_runway_utc       TIMESTAMPTZ,
    dep_best_utc         TIMESTAMPTZ,
    arr_best_utc         TIMESTAMPTZ,
    dep_delay_min        DOUBLE PRECISION,
    arr_delay_min        DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_tfbf_job_id
    ON training_flights_backfill (job_id);
CREATE INDEX IF NOT EXISTS idx_tfbf_flight_date
    ON training_flights_backfill (flight_date);
CREATE INDEX IF NOT EXISTS idx_tfbf_number_dep
    ON training_flights_backfill (number_raw, dep_sched_utc, movement);


-- ──────────────────────────────────────────────────────────────
-- Table 3: retrain_jobs
-- One row per retraining run. Updated at every step by the
-- pipeline so the admin UI can show live progress.
-- status values:  queued | running | completed | failed
-- outcome values: promoted | rejected | failed | NULL (running)
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrain_jobs (
    id               SERIAL PRIMARY KEY,
    triggered_by     TEXT             NOT NULL,
    triggered_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    status           TEXT             NOT NULL DEFAULT 'queued',
    current_step     TEXT,
    step_detail      TEXT,
    outcome          TEXT,
    error_message    TEXT,
    backfill_start   DATE,
    backfill_end     DATE,
    api_calls_made   INTEGER          NOT NULL DEFAULT 0,
    new_auc15        DOUBLE PRECISION,
    new_prauc15      DOUBLE PRECISION,
    new_auc30        DOUBLE PRECISION,
    new_prauc30      DOUBLE PRECISION,
    new_mae_reg      DOUBLE PRECISION
);


-- ──────────────────────────────────────────────────────────────
-- Table 4: retrain_api_calls
-- Logs every AeroDataBox FIDS call made during a retrain
-- backfill. Used to enforce and audit monthly quota limits.
-- window values: 'T00:00-T11:59' or 'T12:00-T23:59'
-- status values: 'ok' or 'failed'
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrain_api_calls (
    id            SERIAL PRIMARY KEY,
    job_id        INTEGER     NOT NULL
                      REFERENCES retrain_jobs(id) ON DELETE CASCADE,
    called_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    day_str       DATE        NOT NULL,
    "window"        TEXT        NOT NULL,
    status        TEXT        NOT NULL,
    rows_fetched  INTEGER,
    error_detail  TEXT
);

CREATE INDEX IF NOT EXISTS idx_rac_job_id
    ON retrain_api_calls (job_id);
CREATE INDEX IF NOT EXISTS idx_rac_called_at
    ON retrain_api_calls (called_at);


-- ──────────────────────────────────────────────────────────────
-- Table 5: model_versions
-- Immutable record of every model version ever trained.
-- Only one row may have is_current = TRUE at any time.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    version_tag     TEXT             NOT NULL UNIQUE,
    trained_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    job_id          INTEGER          REFERENCES retrain_jobs(id),
    data_date_from  DATE,
    data_date_to    DATE,
    auc15_test      DOUBLE PRECISION,
    prauc15_test    DOUBLE PRECISION,
    auc30_test      DOUBLE PRECISION,
    prauc30_test    DOUBLE PRECISION,
    mae_reg_test    DOUBLE PRECISION,
    threshold_bin15 DOUBLE PRECISION,
    threshold_bin30 DOUBLE PRECISION,
    is_current      BOOLEAN          NOT NULL DEFAULT FALSE,
    artifact_dir    TEXT,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_mv_is_current
    ON model_versions (is_current);


-- ──────────────────────────────────────────────────────────────
-- Seed row: v3_final (the current production model baseline).
-- data_date_from and data_date_to are left NULL here — the
-- developer fills these in manually after checking the parquet
-- date range before running the migration.
-- Skipped automatically if v3_final already exists.
-- ──────────────────────────────────────────────────────────────
INSERT INTO model_versions (
    version_tag,
    trained_at,
    job_id,
    data_date_from,
    data_date_to,
    auc15_test,
    prauc15_test,
    auc30_test,
    prauc30_test,
    mae_reg_test,
    threshold_bin15,
    threshold_bin30,
    is_current,
    artifact_dir,
    notes
)
SELECT
    'v3_final',
    NOW(),
    NULL,
    '2025-02-22',   -- data_date_from
    '2026-02-21',   -- data_date_to
    0.7619,
    0.5656,
    0.7419,
    0.3539,
    16.55,
    0.30,
    0.40,
    TRUE,
    'models/v3_final',
    'Original clean training, leakage-free. 365-day MUC backfill. v4 discarded due to leakage.'
WHERE NOT EXISTS (
    SELECT 1 FROM model_versions WHERE version_tag = 'v3_final'
);

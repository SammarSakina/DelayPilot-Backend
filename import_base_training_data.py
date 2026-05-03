"""
import_base_training_data.py
─────────────────────────────────────────────────────────────
One-time script to import the original 365-day MUC backfill
parquet into the training_flights_base database table.

Run this once after Phase 2 migrations are complete.
Never needs to be run again — the table is the permanent base.

Usage (from inside the pipeline venv):
    python import_base_training_data.py --parquet <path_to_parquet>

Example:
    python import_base_training_data.py \
        --parquet "C:/Users/zeeni/Documents/xtra/FYP Notebooks/data/api_backfill/aerodatabox/muc_365d_fids/muc_fids_365d_full.parquet"
─────────────────────────────────────────────────────────────
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Columns expected in the parquet (from the backfill notebook) ──────────────
# These are the only columns written to training_flights_base.
# Any extra columns in the parquet are silently ignored.
EXPECTED_COLS = [
    "movement",
    "number_raw",
    "call_sign",
    "status",
    "codeshare_status",
    "is_cargo",
    "airline_iata",
    "airline_icao",
    "airline_name",
    "aircraft_model",
    "aircraft_modeS",
    "aircraft_reg",
    "other_airport_iata",
    "other_airport_icao",
    "dep_sched_utc",
    "dep_rev_utc",
    "dep_pred_utc",
    "dep_runway_utc",
    "arr_sched_utc",
    "arr_rev_utc",
    "arr_pred_utc",
    "arr_runway_utc",
    "dep_best_utc",
    "arr_best_utc",
    "dep_delay_min",
    "arr_delay_min",
]

TIMESTAMP_COLS = [
    "dep_sched_utc", "dep_rev_utc", "dep_pred_utc", "dep_runway_utc",
    "arr_sched_utc", "arr_rev_utc", "arr_pred_utc", "arr_runway_utc",
    "dep_best_utc", "arr_best_utc",
]


def get_connection_string() -> str:
    import os
    from dotenv import load_dotenv
    load_dotenv()

    url = os.getenv("DATABASE_URL")
    if url:
        logger.info("Using DATABASE_URL from environment.")
        return url

    pg_host = os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_user = os.getenv("POSTGRES_USER")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "")
    pg_db   = os.getenv("POSTGRES_DB")

    if pg_host and pg_user and pg_db:
        logger.info(
            "Using POSTGRES_* vars: host=%s port=%s db=%s user=%s",
            pg_host, pg_port, pg_db, pg_user,
        )
        return f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

    logger.warning(
        "POSTGRES_* vars not found — falling back to DB_* vars."
    )
    user     = os.getenv("DB_USER", "postgres")
    host     = os.getenv("DB_HOST", "localhost")
    name     = os.getenv("DB_NAME", "delaypilot")
    password = os.getenv("DB_PASSWORD", "")
    port     = os.getenv("DB_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def load_parquet(parquet_path: Path) -> pd.DataFrame:
    logger.info("Reading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Parquet loaded: %d rows, %d columns", len(df), len(df.columns))

    # Keep only the columns the DB table expects
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet is missing expected columns: {missing}\n"
            f"Columns present: {df.columns.tolist()}"
        )

    df = df[EXPECTED_COLS].copy()

    # Ensure all timestamp columns are UTC-aware datetime (before logging)
    for col in TIMESTAMP_COLS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Log stats while columns are still proper datetime/float types
    logger.info(
        "Date range: dep_sched_utc %s → %s",
        df["dep_sched_utc"].min(),
        df["dep_sched_utc"].max(),
    )
    logger.info(
        "dep_delay_min known rate: %.1f%%",
        df["dep_delay_min"].notna().mean() * 100,
    )
    logger.info(
        "arr_delay_min known rate: %.1f%%",
        df["arr_delay_min"].notna().mean() * 100,
    )

    # NOW convert NaT/NaN to None for psycopg2 — after all logging is done
    for col in TIMESTAMP_COLS:
        df[col] = df[col].astype(object).where(df[col].notna(), other=None)

    float_cols = ["dep_delay_min", "arr_delay_min"]
    for col in float_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), other=None)

    str_cols = [c for c in EXPECTED_COLS if c not in TIMESTAMP_COLS + float_cols]
    for col in str_cols:
        df[col] = df[col].where(df[col].notna(), other=None)

    return df


def check_already_imported(cur) -> int:
    cur.execute("SELECT COUNT(*) FROM training_flights_base")
    return cur.fetchone()[0]


def insert_rows(conn, df: pd.DataFrame, batch_size: int = 2000):
    cols = EXPECTED_COLS
    insert_sql = f"""
        INSERT INTO training_flights_base ({", ".join(cols)})
        VALUES %s
        ON CONFLICT DO NOTHING
    """

    rows = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
    total = len(rows)
    inserted = 0

    with conn.cursor() as cur:
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            psycopg2.extras.execute_values(cur, insert_sql, batch)
            inserted += len(batch)
            logger.info(
                "Inserted %d / %d rows ...", inserted, total
            )
        conn.commit()

    return inserted


def run(parquet_path: Path):
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = load_parquet(parquet_path)

    conn_string = get_connection_string()
    logger.info("Connecting to database ...")
    conn = psycopg2.connect(conn_string)

    try:
        with conn.cursor() as cur:
            existing = check_already_imported(cur)

        if existing > 0:
            logger.warning(
                "training_flights_base already contains %d rows. "
                "This script is meant to be run only once. "
                "Aborting to prevent duplicates. "
                "If you genuinely need to re-import, truncate the table "
                "manually first: TRUNCATE training_flights_base;",
                existing,
            )
            return

        logger.info(
            "Table is empty — proceeding with import of %d rows.", len(df)
        )
        inserted = insert_rows(conn, df)
        logger.info("Import complete. Total rows inserted: %d", inserted)

        # Final verification
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM training_flights_base")
            count = cur.fetchone()[0]
        logger.info(
            "Verification: training_flights_base now contains %d rows.", count
        )

    finally:
        conn.close()

    logger.info(
        "Phase 1 complete. training_flights_base is ready. "
        "You may now proceed to Phase 3 (pipeline retraining code)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import 365-day MUC backfill parquet into training_flights_base."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Full path to muc_fids_365d_full.parquet",
    )
    args = parser.parse_args()
    run(Path(args.parquet))

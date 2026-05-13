import logging
import os

import pandas as pd
from sqlalchemy import create_engine

try:
    # Fallback to live flights if historical parquet path is missing
    from ingest_flights_live import ingest_live_muc_window
except ImportError:  # pragma: no cover
    ingest_live_muc_window = None  # type: ignore[misc]


logger = logging.getLogger(__name__)


def get_engine() -> "create_engine":
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "delaypilot2026")
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_db = os.getenv("PG_DB", "delaypilot_db")

    url = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    return create_engine(url)


def load_flight_parquet_to_db(max_rows: int = 50000) -> None:
    """
    Load historical flight data from Parquet into flights_raw table in Postgres.

    Only up to `max_rows` rows will be inserted to respect the 50k requirement.
    The source path may contain non-parquet files (e.g. raw JSON); we only read
    .parquet files.
    """
    
    # (TO BE ADJUSTED LATER )
    flights_path = r"F:\FYP Notebooks\data\api_backfill\aerodatabox\muc_365d_fids"

    if not os.path.exists(flights_path):
        logger.warning("Flight parquet path not found: %s", flights_path)
        if ingest_live_muc_window is not None:
            logger.info("Fetching live Flights data from Aerodatabox ingestion instead.")
            ingest_live_muc_window()
            return
        else:
            logger.error("No live flights fallback available; cannot populate flights_raw.")
            raise FileNotFoundError(f"Flight parquet path not found: {flights_path}")

    # Collect parquet files
    parquet_files: list[str] = []
    if os.path.isdir(flights_path):
        for root, _dirs, files in os.walk(flights_path):
            for name in files:
                if name.lower().endswith(".parquet"):
                    parquet_files.append(os.path.join(root, name))
    else:
        if flights_path.lower().endswith(".parquet"):
            parquet_files.append(flights_path)

    if not parquet_files:
        logger.error("No .parquet files found under: %s", flights_path)
        raise FileNotFoundError(f"No .parquet files found under: {flights_path}")

    logger.info("Reading flight parquet files:")
    for f in parquet_files:
        logger.info(" - %s", f)

    # Read and concatenate all parquet files
    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    logger.info("Total rows in parquet: %d", len(df))
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        logger.info("Sampled down to %d rows for ingestion.", len(df))

    engine = get_engine()
    logger.info("Writing data to Postgres table: flights_raw")

    df.to_sql(
        "flights_raw",
        engine,
        if_exists="replace",
        index=False,
    )
    logger.info("Inserted %d rows into flights_raw.", len(df))


if __name__ == "__main__":
    load_flight_parquet_to_db()


"""
import_base_to_supabase.py
Copies training_flights_base from local PostgreSQL to Supabase.
Zero API calls.
"""
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def get_local_engine():
    return create_engine(
        f"postgresql+psycopg2://"
        f"{os.getenv('PG_USER','postgres')}:{os.getenv('PG_PASSWORD','delaypilot2026')}"
        f"@{os.getenv('PG_HOST','localhost')}:{os.getenv('PG_PORT','5432')}"
        f"/{os.getenv('PG_DB','delaypilot_db')}"
    )

def get_supabase_engine():
    url = os.getenv("DATABASE_URL")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if "sslmode" not in url:
        url += "?sslmode=require"
    return create_engine(
        url.replace("postgresql://", "postgresql+psycopg2://", 1)
    )

def run():
    logger.info("Reading training_flights_base from local PostgreSQL...")
    local_engine = get_local_engine()
    df = pd.read_sql("SELECT * FROM training_flights_base", local_engine)
    logger.info("Read %d rows from local training_flights_base", len(df))

    if len(df) == 0:
        raise ValueError(
            "training_flights_base is empty in local PostgreSQL. "
            "Run import_base_training_data.py first."
        )

    logger.info("Writing to Supabase training_flights_base...")
    supa_engine = get_supabase_engine()

    # Check if already has rows
    existing = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM training_flights_base",
        supa_engine
    ).iloc[0]["cnt"]

    if existing > 0:
        logger.info(
            "Supabase training_flights_base already has %d rows — "
            "skipping import, data already present.", existing
        )
        return

    # Write in chunks to avoid timeout
    chunk_size = 5000
    total = len(df)
    for i in range(0, total, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk.to_sql(
            "training_flights_base", supa_engine,
            if_exists="append", index=False
        )
        logger.info("Inserted %d / %d rows...", min(i+chunk_size, total), total)

    logger.info("Done. Supabase training_flights_base now has %d rows.", total)

if __name__ == "__main__":
    run()
"""
DelayPilot data pipeline runner

Runs end-to-end:
  1) Ingest weather data -> weather_hourly
  2) Ingest flight data -> flights_raw
  3) Build feature table -> training_features_v3
  4) Clean features -> training_features_v3_clean

Usage:
  python run_pipeline.py              # run all steps
  python run_pipeline.py --no-clean   # skip final cleaning
"""

import argparse
import logging

from ingest_weather import load_weather_csv_to_db
from ingest_flights import load_flight_parquet_to_db
from build_featured_muc_rxn_wx3 import build_featured_muc_rxn_wx3
from build_featured_muc_rxn_wx3_fe import build_featured_muc_rxn_wx3_fe
from build_training_features import build_features
from clean_training_features import clean_training_features
from run_batch_predictions import run_batch_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run():
    parser = argparse.ArgumentParser(description="DelayPilot ETL pipeline (PostgreSQL)")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip final cleaning step (only build training_features_v3).",
    )
    args = parser.parse_args()

    logger.info("Starting DelayPilot pipeline (PostgreSQL)")

    logger.info("Step 1/8: Ingest weather data -> weather_hourly")
    load_weather_csv_to_db()

    logger.info("Step 2/8: Ingest flight data -> flights_raw")
    load_flight_parquet_to_db()

    logger.info("Step 3/8: Build featured_muc_rxn_wx3 (reactionary + weather joins)")
    build_featured_muc_rxn_wx3()

    logger.info("Step 4/8: Build featured_muc_rxn_wx3_fe (congestion + history + flags)")
    build_featured_muc_rxn_wx3_fe()

    logger.info("Step 5/8: Build joined features -> training_features_v3")
    build_features()

    if not args.no_clean:
        logger.info("Step 6/8: Clean features -> training_features_v3_clean")
        clean_training_features()
    else:
        logger.info("Skipping cleaning step (per --no-clean).")

    logger.info("Step 7/8: Run batch ML predictions -> flight_predictions")
    try:
        n = run_batch_predictions()
        logger.info("Step 7/8: Wrote %d predictions to flight_predictions.", n)
    except Exception as e:
        logger.warning("Step 7/8: Batch predictions failed (non-fatal): %s", e)

    logger.info("Step 8/8: Snapshot delay analytics -> flight_delay_snapshots")
    try:
        from snapshot_delay_analytics import snapshot_delay_analytics
        snapshot_delay_analytics()
        logger.info("Step 8/8: Delay analytics snapshot written.")
    except Exception as e:
        logger.warning("Step 8/8: Delay analytics snapshot failed (non-fatal): %s", e)

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    run()
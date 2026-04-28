"""
start_delaypilot.py
────────────────────────────────────────────────────────────────────────────
Launcher for the DelayPilot project.

Usage:
    python start_delaypilot.py

What it does:
  1) Run the historical ETL pipeline (run_pipeline.py)
     — populates flights_raw, weather_hourly, all feature tables from
       historical parquet/CSV files so the API has data immediately.
  2) Start FastAPI on http://localhost:8000
     — background_refresh.py takes over from here:
       • FIDS refresh fires immediately in background, then every 2h
       • Flight Status fires after FIDS completes, then every 15 min
       • Server is responsive the entire time
────────────────────────────────────────────────────────────────────────────
"""

import logging
import subprocess
import sys
from pathlib import Path

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def run_pipeline() -> None:
    logger.info("═══ Step 1: Real-Time ETL pipeline ═══")
    subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py")],
        check=True
    )
    logger.info("Real-Time pipeline finished — starting API server.")


def start_api() -> None:
    logger.info("═══ Step 2: Starting API on http://localhost:8000 ═══")
    logger.info("Background scheduler will handle live FIDS + Flight Status refreshes.")
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=False)


def main() -> None:
    run_pipeline()
    start_api()


if __name__ == "__main__":
    main()
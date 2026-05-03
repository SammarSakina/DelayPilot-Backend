"""
fetch_historical_weather.py
─────────────────────────────────────────────────────────────
Fetches historical hourly weather from Open-Meteo archive API
for EDDM, EDDF, EGLL and writes to weather_hourly in Supabase.

No API key required. Free endpoint. No quota limit.

Usage:
    python fetch_historical_weather.py \
        --start 2025-02-22 \
        --end   2026-05-01

Run this before triggering a retrain to ensure weather_hourly
covers the full training date range.
─────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import time
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# EDDM, EDDF, EGLL — same coordinates as ingest_weather_live.py
AIRPORTS = [
    {"icao": "EDDM", "lat": 48.3538, "lon": 11.7861},
    {"icao": "EDDF", "lat": 50.0267, "lon":  8.5584},
    {"icao": "EGLL", "lat": 51.4707, "lon": -0.4599},
]

# Same variables as ingest_weather_live.py — identical columns
HOURLY_VARS = ",".join([
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "snowfall", "snow_depth", "rain", "weather_code",
    "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "visibility", "vapour_pressure_deficit",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "is_day", "dew_point_2m", "wet_bulb_temperature_2m",
    "boundary_layer_height", "sunshine_duration",
])


def fetch_one_airport(icao: str, lat: float, lon: float,
                      start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  start,
        "end_date":    end,
        "hourly":      HOURLY_VARS,
        "timezone":    "GMT",
    }
    logger.info("Fetching historical weather for %s: %s → %s", icao, start, end)
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["hour_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.drop(columns=["time"])
    df["airport_icao"] = icao

    # Reorder columns: keys first
    cols = ["airport_icao", "hour_utc"] + [
        c for c in df.columns
        if c not in ("airport_icao", "hour_utc")
    ]
    df = df[cols]
    logger.info("%s: fetched %d rows", icao, len(df))
    return df


def get_engine():
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set in .env")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if "sslmode" not in url:
        url += "?sslmode=require"
    return create_engine(
        url.replace("postgresql://", "postgresql+psycopg2://", 1)
    )


def run(start: str, end: str):
    frames = []
    for apt in AIRPORTS:
        df = fetch_one_airport(
            apt["icao"], apt["lat"], apt["lon"], start, end
        )
        frames.append(df)
        time.sleep(1)  # be polite to the free API

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined: %d rows for %d airports",
        len(combined), len(AIRPORTS)
    )
    logger.info(
        "Date range in data: %s → %s",
        combined["hour_utc"].min(),
        combined["hour_utc"].max(),
    )

    engine = get_engine()
    combined.to_sql(
        "weather_hourly", engine,
        if_exists="replace",
        index=False,
    )
    logger.info(
        "Written %d rows to Supabase weather_hourly. Done.",
        len(combined)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical weather and write to weather_hourly."
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--end", required=True,
        help="End date YYYY-MM-DD (inclusive)",
    )
    args = parser.parse_args()
    run(args.start, args.end)
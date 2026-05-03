"""
retrain_pipeline.py
─────────────────────────────────────────────────────────────
Background retraining job for DelayPilot.
Called by POST /retrain/start in api_main.py.
Runs in a daemon thread — never blocks the API.

Mirrors the exact methodology of:
  SCHED_01_aerodatabox_backfill_labels_muc_365d.ipynb

Key design decisions:
- Each day is committed to DB immediately on success
  (equivalent of daily parquet files in the notebook)
- Failed days are retried in a second pass after the main
  loop (mirrors the notebook's retry cell)
- Days already present in training_flights_backfill for
  this job are skipped so the job is resumable
- Coverage stats logged to retrain_jobs after the backfill
- No parquet files written — DB rows replace them
─────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RAPIDAPI_HOST      = "aerodatabox.p.rapidapi.com"
FIDS_UNITS_PER_CALL = 2   # Tier 2 endpoint
from dotenv import load_dotenv as _load_env
_load_env()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")

LEAK_OR_LABEL_COLS = [
    "y_delay_min","y_bin15","y_bin30","dep_delay_min","arr_delay_min",
    "dep_best_utc","arr_best_utc","dep_rev_utc","arr_rev_utc",
    "dep_runway_utc","arr_runway_utc","best_utc",
    "arr_delay_bucket_5_15_30",
    "status","call_sign","number_raw","codeshare_status",
    "aircraft_modeS","aircraft_reg","airline_name","airport_iata",
    "dep_sched_utc","arr_sched_utc",
]
TIME_KEYS = [
    "sched_utc","ref_ts_utc","ref_hour_utc","ref_date","sched_hour_utc",
]

CLF_PARAMS = dict(
    iterations=1200, depth=8, learning_rate=0.05,
    loss_function="Logloss", eval_metric="AUC",
    random_seed=42, verbose=200,
)
REG_PARAMS = dict(
    iterations=2500, depth=8, learning_rate=0.05,
    loss_function="MAE", eval_metric="MAE",
    random_seed=42, verbose=200,
)

PRODUCTION_METRICS = {
    "auc15": 0.7619, "prauc15": 0.5656,
    "auc30": 0.7419, "prauc30": 0.3539,
}


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — prefer DATABASE_URL (Supabase) over PG_* (local)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _get_engine():
    from dotenv import load_dotenv
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        database_url = _normalise_db_url(database_url)
        if "sslmode" not in database_url:
            sep = "&" if "?" in database_url else "?"
            database_url += f"{sep}sslmode=require"
        sa_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return create_engine(sa_url)
    url = (
        f"postgresql+psycopg2://"
        f"{os.getenv('PG_USER','postgres')}:{os.getenv('PG_PASSWORD','delaypilot2026')}"
        f"@{os.getenv('PG_HOST','localhost')}:{os.getenv('PG_PORT','5432')}"
        f"/{os.getenv('PG_DB','delaypilot_db')}"
    )
    return create_engine(url)


def _get_conn():
    from dotenv import load_dotenv
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        database_url = _normalise_db_url(database_url)
        if "sslmode" not in database_url:
            sep = "&" if "?" in database_url else "?"
            database_url += f"{sep}sslmode=require"
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        host=os.getenv("PG_HOST","localhost"),
        port=int(os.getenv("PG_PORT","5432")),
        dbname=os.getenv("PG_DB","delaypilot_db"),
        user=os.getenv("PG_USER","postgres"),
        password=os.getenv("PG_PASSWORD","delaypilot2026"),
    )


def _update_job(conn, job_id, step, detail="", status="running"):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE retrain_jobs SET current_step=%s,step_detail=%s,status=%s WHERE id=%s",
            (step, detail[:2000], status, job_id),
        )
    conn.commit()
    logger.info("[job=%d] %s | %s", job_id, step, detail[:120])


def _fail_job(conn, job_id, step, error):
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE retrain_jobs SET status='failed',outcome='failed',
               current_step=%s,error_message=%s,finished_at=NOW() WHERE id=%s""",
            (step, str(error)[:2000], job_id),
        )
    conn.commit()
    logger.error("[job=%d] FAILED at %s: %s", job_id, step, error)


def _finish_job(conn, job_id, outcome, metrics=None, error=None):
    m = metrics or {}
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE retrain_jobs SET status='completed',outcome=%s,
               finished_at=NOW(),error_message=%s,
               new_auc15=%s,new_prauc15=%s,new_auc30=%s,new_prauc30=%s,new_mae_reg=%s
               WHERE id=%s""",
            (outcome, error,
             m.get("auc15"), m.get("prauc15"),
             m.get("auc30"), m.get("prauc30"),
             m.get("mae_reg"), job_id),
        )
    conn.commit()


def _log_api_call(conn, job_id, day_str, window, status,
                  rows_fetched=0, error_detail=""):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO retrain_api_calls
               (job_id,day_str,"window",status,rows_fetched,error_detail)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (job_id, day_str, window, status,
             rows_fetched, str(error_detail)[:500]),
        )
        cur.execute(
            "UPDATE retrain_jobs SET api_calls_made=api_calls_made+1 WHERE id=%s",
            (job_id,),
        )
    conn.commit()


def _is_cancelled(conn, job_id: int) -> bool:
    """
    Check whether the admin has cancelled this job via the UI.
    Called at the start of each major step so the thread exits
    cleanly without completing unnecessary work.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT outcome FROM retrain_jobs WHERE id=%s",
            (job_id,),
        )
        row = cur.fetchone()
    return row is not None and row[0] == "cancelled"


# ─────────────────────────────────────────────────────────────────────────────
# FIDS helpers — mirrors notebook exactly
# ─────────────────────────────────────────────────────────────────────────────

def _parse_utc(s):
    if s is None:
        return pd.NaT
    s = str(s).strip()
    if " " in s and s.endswith("Z") and "T" not in s:
        s = s.replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return pd.to_datetime(s, errors="coerce", utc=True)


def _get_utc(obj, field):
    if not obj:
        return pd.NaT
    return _parse_utc((obj.get(field, {}) or {}).get("utc"))


def _fetch_fids(from_local: str, to_local: str,
                api_key: str,
                direction: str = "Both",
                with_leg: bool = True,
                with_cancelled: bool = True,
                with_codeshared: bool = True,
                with_cargo: bool = True,
                with_private: bool = True,
                with_location: bool = False) -> dict:
    """
    Mirrors fetch_fids_muc() in the notebook exactly,
    including str(value).lower() param serialisation.
    """
    url = f"https://{RAPIDAPI_HOST}/flights/airports/iata/MUC/{from_local}/{to_local}"
    r = requests.get(
        url,
        headers={
            "Accept":           "application/json",
            "X-RapidAPI-Key":   RAPIDAPI_KEY,
            "X-RapidAPI-Host":  RAPIDAPI_HOST,
        },
        params={
            "withLeg":        str(with_leg).lower(),
            "direction":      direction,
            "withCancelled":  str(with_cancelled).lower(),
            "withCodeshared": str(with_codeshared).lower(),
            "withCargo":      str(with_cargo).lower(),
            "withPrivate":    str(with_private).lower(),
            "withLocation":   str(with_location).lower(),
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _normalize_fids(data: dict, airport_iata: str = "MUC") -> pd.DataFrame:
    """
    Mirrors normalize_fids() in the notebook exactly.
    Includes airport_iata, other_airport_name, other_airport_tz,
    and arr_delay_bucket_5_15_30 — all present in the original parquet.
    """
    rows = []
    for mv, items in [("departure", data.get("departures",[])),
                      ("arrival",   data.get("arrivals",[]))]:
        for it in items:
            dep = it.get("departure",{}) or {}
            arr = it.get("arrival",  {}) or {}
            al  = it.get("airline",  {}) or {}
            ac  = it.get("aircraft", {}) or {}
            oth = ((arr.get("airport",{}) or {}) if mv=="departure"
                   else (dep.get("airport",{}) or {}))
            rows.append({
                "movement":            mv,
                "airport_iata":        airport_iata,
                "number_raw":          it.get("number"),
                "call_sign":           it.get("callSign"),
                "status":              it.get("status"),
                "codeshare_status":    it.get("codeshareStatus"),
                "is_cargo":            it.get("isCargo"),

                "airline_name":        al.get("name"),
                "airline_iata":        al.get("iata"),
                "airline_icao":        al.get("icao"),

                "aircraft_model":      ac.get("model"),
                "aircraft_reg":        ac.get("reg"),
                "aircraft_modeS":      ac.get("modeS"),

                "other_airport_iata":  oth.get("iata"),
                "other_airport_icao":  oth.get("icao"),
                "other_airport_name":  oth.get("name"),
                "other_airport_tz":    oth.get("timeZone"),

                "dep_sched_utc":       _get_utc(dep,"scheduledTime"),
                "dep_rev_utc":         _get_utc(dep,"revisedTime"),
                "dep_pred_utc":        _get_utc(dep,"predictedTime"),
                "dep_runway_utc":      _get_utc(dep,"runwayTime"),
                
                "arr_sched_utc":       _get_utc(arr,"scheduledTime"),
                "arr_rev_utc":         _get_utc(arr,"revisedTime"),
                "arr_pred_utc":        _get_utc(arr,"predictedTime"),
                "arr_runway_utc":      _get_utc(arr,"runwayTime"),
            })
    return pd.DataFrame(rows)


def _process_day_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors the dtype-fix + best-time + label computation block
    inside the notebook's backfill loop exactly.
    """
    time_cols = [
        "dep_sched_utc","dep_rev_utc","dep_pred_utc","dep_runway_utc",
        "arr_sched_utc","arr_rev_utc","arr_pred_utc","arr_runway_utc",
    ]
    for c in time_cols:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    # best_utc: runway → rev → pred  (exact notebook fillna chain)
    df["dep_best_utc"] = (df["dep_runway_utc"]
                          .fillna(df["dep_rev_utc"])
                          .fillna(df["dep_pred_utc"]))
    df["arr_best_utc"] = (df["arr_runway_utc"]
                          .fillna(df["arr_rev_utc"])
                          .fillna(df["arr_pred_utc"]))
    df["dep_best_utc"] = pd.to_datetime(df["dep_best_utc"], utc=True, errors="coerce")
    df["arr_best_utc"] = pd.to_datetime(df["arr_best_utc"], utc=True, errors="coerce")

    # delay in minutes
    df["dep_delay_min"] = (
        (df["dep_best_utc"] - df["dep_sched_utc"]).dt.total_seconds() / 60
    )
    df["arr_delay_min"] = (
        (df["arr_best_utc"] - df["arr_sched_utc"]).dt.total_seconds() / 60
    )

    # arrival delay bucket (exact notebook bucket function)
    def _bucket(x):
        if pd.isna(x): return np.nan
        if x < 5:  return 0
        if x < 15: return 1
        if x < 30: return 2
        return 3

    df["arr_delay_bucket_5_15_30"] = df["arr_delay_min"].apply(_bucket)
    return df


def _day_stats(df: pd.DataFrame) -> dict:
    """Compute the same coverage stats the notebook tracked per day."""
    return {
        "rows":             len(df),
        "dep_known_rate":   float(df["dep_delay_min"].notna().mean()),
        "arr_known_rate":   float(df["arr_delay_min"].notna().mean()),
        "dep_runway_rate":  float(df["dep_runway_utc"].notna().mean()),
        "arr_runway_rate":  float(df["arr_runway_utc"].notna().mean()),
    }


# Columns written to training_flights_backfill
# (superset of training_flights_base to include notebook extras)
_INSERT_COLS = [
    "job_id","flight_date","movement","airport_iata","number_raw",
    "call_sign","status","codeshare_status","is_cargo",
    "airline_iata","airline_icao","airline_name",
    "aircraft_model","aircraft_modeS","aircraft_reg",
    "other_airport_iata","other_airport_icao",
    "dep_sched_utc","dep_rev_utc","dep_pred_utc","dep_runway_utc",
    "arr_sched_utc","arr_rev_utc","arr_pred_utc","arr_runway_utc",
    "dep_best_utc","arr_best_utc","dep_delay_min","arr_delay_min",
]
_TS_COLS = [c for c in _INSERT_COLS if "utc" in c]


def _insert_day(conn, df: pd.DataFrame, job_id: int, flight_date: date):
    """Insert one day's rows into training_flights_backfill."""
    df = df.copy()
    df["job_id"]      = job_id
    df["flight_date"] = flight_date

    # Ensure all expected columns exist
    for c in _INSERT_COLS:
        if c not in df.columns:
            df[c] = None

    # Convert timestamps to None-safe objects for psycopg2
    for c in _TS_COLS:
        df[c] = df[c].astype(object).where(df[c].notna(), other=None)
    for c in ["dep_delay_min","arr_delay_min"]:
        df[c] = df[c].astype(object).where(df[c].notna(), other=None)
    for c in [c for c in _INSERT_COLS
              if c not in _TS_COLS + ["dep_delay_min","arr_delay_min",
                                      "job_id","flight_date"]]:
        if c in df.columns:
            df[c] = df[c].where(pd.notnull(df[c]), other=None)

    rows = [tuple(r) for r in df[_INSERT_COLS].itertuples(index=False, name=None)]
    sql  = (f"INSERT INTO training_flights_backfill "
            f"({', '.join(_INSERT_COLS)}) VALUES %s ON CONFLICT DO NOTHING")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
    conn.commit()


def _already_ingested_dates(conn, job_id: int) -> set:
    """
    Return dates already successfully inserted for ANY job,
    not just the current one.

    This prevents re-fetching API data for days that were
    successfully ingested by a previous job that later failed
    at a different step (e.g. feature engineering).

    The current job_id is accepted as a parameter for
    consistency with callers but is not used in the query.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT flight_date FROM training_flights_backfill"
        )
        return {row[0] for row in cur.fetchall()}


def _failed_windows(conn, job_id: int) -> list:
    """
    Return list of (day_str, window) pairs that failed in the main loop.
    Used by the retry pass — mirrors notebook retry cell logic.
    """
    with conn.cursor() as cur:
        cur.execute(
            """SELECT day_str, "window" FROM retrain_api_calls
               WHERE job_id=%s AND status='failed'""",
            (job_id,),
        )
        return [(str(row[0]), row[1]) for row in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — mirrors existing pipeline files exactly
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_ansperf(data_dir: Path) -> dict:
    """
    Mirrors prepare_ansperf.py logic exactly.
    Reads all airport_traffic_YYYY.csv and apt_dly_YYYY.csv.bz2
    files from data_dir, builds clean parquet files, and writes
    them to data_dir/traffic_munich_daily.parquet and
    data_dir/atfm_delay_munich_daily.parquet.

    Returns a dict with keys:
      traffic_rows, atfm_rows, traffic_date_range, atfm_date_range
    or raises an exception if no source files are found.
    """
    MUC = "EDDM"

    # ── Traffic ──────────────────────────────────────────────
    traffic_files = sorted(data_dir.glob("airport_traffic_*.csv"))
    if not traffic_files:
        raise FileNotFoundError(
            f"No airport_traffic_*.csv files found in {data_dir}. "
            "Download from Eurocontrol ANSPerformance and place in "
            "the pipeline data/ folder."
        )

    traffic_parts = []
    for path in traffic_files:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        sample = str(df["FLT_DATE"].dropna().iloc[0]) if len(df) > 0 else ""
        df["date_dt"] = pd.to_datetime(
            df["FLT_DATE"],
            dayfirst="/" in sample,
            errors="coerce",
        )
        traffic_parts.append(df)
        logger.info("_prepare_ansperf: loaded %s (%d rows)", path.name, len(df))

    traffic_all = pd.concat(traffic_parts, ignore_index=True)
    traffic_all["APT_ICAO"] = (
        traffic_all["APT_ICAO"].astype(str).str.upper().str.strip()
    )
    traffic_muc = traffic_all[
        (traffic_all["APT_ICAO"] == MUC) &
        (traffic_all["date_dt"].notna())
    ].copy()
    traffic_muc["date"] = traffic_muc["date_dt"].dt.date
    traffic_daily = traffic_muc.rename(columns={
        "APT_ICAO":      "airport",
        "FLT_DEP_1":     "dep_cnt",
        "FLT_ARR_1":     "arr_cnt",
        "FLT_TOT_1":     "tot_cnt",
        "FLT_DEP_IFR_2": "dep_ifr_cnt",
        "FLT_ARR_IFR_2": "arr_ifr_cnt",
        "FLT_TOT_IFR_2": "tot_ifr_cnt",
    })[[
        "date", "airport",
        "dep_cnt", "arr_cnt", "tot_cnt",
        "dep_ifr_cnt", "arr_ifr_cnt", "tot_ifr_cnt",
    ]]

    # ── ATFM ─────────────────────────────────────────────────
    atfm_files = sorted(data_dir.glob("apt_dly_*.csv.bz2"))
    if not atfm_files:
        raise FileNotFoundError(
            f"No apt_dly_*.csv.bz2 files found in {data_dir}. "
            "Download from Eurocontrol ANSPerformance and place in "
            "the pipeline data/ folder."
        )

    atfm_parts = []
    for path in atfm_files:
        df = pd.read_csv(path, compression="bz2", encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        atfm_parts.append(df)
        logger.info("_prepare_ansperf: loaded %s (%d rows)", path.name, len(df))

    atfm_all = pd.concat(atfm_parts, ignore_index=True)
    atfm_all["date"] = pd.to_datetime(
        atfm_all["FLT_DATE"], errors="coerce", utc=True
    ).dt.date
    atfm_all["APT_ICAO"] = (
        atfm_all["APT_ICAO"].astype(str).str.upper().str.strip()
    )
    atfm_muc = atfm_all[atfm_all["APT_ICAO"] == MUC].copy()
    atfm_daily = atfm_muc[[
        "date", "APT_ICAO",
        "FLT_ARR_1", "DLY_APT_ARR_1",
        "FLT_ARR_1_DLY", "FLT_ARR_1_DLY_15", "ATFM_VERSION",
    ]].copy().rename(columns={
        "APT_ICAO":         "airport",
        "FLT_ARR_1":        "arrivals_cnt",
        "DLY_APT_ARR_1":    "atfm_arr_delay_min_total",
        "FLT_ARR_1_DLY":    "arrivals_delayed_cnt",
        "FLT_ARR_1_DLY_15": "arrivals_delayed15_cnt",
    })
    safe = atfm_daily["arrivals_cnt"].replace({0: pd.NA})
    atfm_daily["atfm_arr_delay_min_per_arrival"] = (
        atfm_daily["atfm_arr_delay_min_total"] / safe
    )
    atfm_daily["arrivals_delayed_rate"]   = (
        atfm_daily["arrivals_delayed_cnt"] / safe
    )
    atfm_daily["arrivals_delayed15_rate"] = (
        atfm_daily["arrivals_delayed15_cnt"] / safe
    )

    # ── Save parquets ─────────────────────────────────────────
    traffic_out = data_dir / "traffic_munich_daily.parquet"
    atfm_out    = data_dir / "atfm_delay_munich_daily.parquet"
    traffic_daily.to_parquet(traffic_out, index=False)
    atfm_daily.to_parquet(atfm_out, index=False)
    logger.info(
        "_prepare_ansperf: saved traffic (%d rows) and ATFM (%d rows)",
        len(traffic_daily), len(atfm_daily),
    )

    return {
        "traffic_rows":       len(traffic_daily),
        "atfm_rows":          len(atfm_daily),
        "traffic_date_range": f"{traffic_daily['date'].min()} → "
                              f"{traffic_daily['date'].max()}",
        "atfm_date_range":    f"{atfm_daily['date'].min()} → "
                              f"{atfm_daily['date'].max()}",
    }


def _fetch_weather_gap(engine, backfill_start: date,
                       backfill_end: date) -> dict:
    """
    Checks weather_hourly coverage in Supabase and fetches any
    missing historical data from Open-Meteo archive API (free).

    The training dataset spans from training_flights_base start
    (2025-02-22) through backfill_end. Weather must cover the
    same range. Any gap is filled here before feature engineering.

    Returns a dict with coverage info.
    """
    # Airports and coordinates — same as ingest_weather_live.py
    AIRPORTS = [
        {"icao": "EDDM", "lat": 48.3538, "lon": 11.7861},
        {"icao": "EDDF", "lat": 50.0267, "lon":  8.5584},
        {"icao": "EGLL", "lat": 51.4707, "lon": -0.4599},
    ]
    HOURLY_VARS = ",".join([
        "temperature_2m", "relative_humidity_2m",
        "apparent_temperature", "precipitation", "snowfall",
        "snow_depth", "rain", "weather_code", "surface_pressure",
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "visibility", "vapour_pressure_deficit",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "is_day", "dew_point_2m", "wet_bulb_temperature_2m",
        "boundary_layer_height", "sunshine_duration",
    ])

    # Check current coverage for EDDM
    check = pd.read_sql(
        """SELECT MIN(hour_utc) as min_dt,
                  MAX(hour_utc) as max_dt,
                  COUNT(*)      as row_count
             FROM weather_hourly
            WHERE airport_icao = 'EDDM'""",
        engine,
    )
    wx_min  = check.iloc[0]["min_dt"]
    wx_max  = check.iloc[0]["max_dt"]
    wx_rows = int(check.iloc[0]["row_count"])

    # Determine required range
    # Base data starts 2025-02-22 — always need weather from there
    required_start = date(2025, 2, 22)
    required_end   = backfill_end

    need_fetch = False
    fetch_start = required_start
    fetch_end   = required_end

    if wx_rows == 0:
        need_fetch = True
        logger.info(
            "_fetch_weather_gap: weather_hourly is empty — "
            "fetching full range %s → %s", fetch_start, fetch_end
        )
    else:
        wx_max_date = pd.Timestamp(wx_max).date() if wx_max else None
        wx_min_date = pd.Timestamp(wx_min).date() if wx_min else None

        if wx_max_date is None or wx_max_date < required_end:
            need_fetch = True
            # Only fetch the missing portion at the end
            fetch_start = (wx_max_date + timedelta(days=1)
                           if wx_max_date else required_start)
            fetch_end   = required_end
            logger.info(
                "_fetch_weather_gap: gap detected — fetching %s → %s",
                fetch_start, fetch_end,
            )
        else:
            logger.info(
                "_fetch_weather_gap: weather OK — %d rows, "
                "%s → %s. No fetch needed.",
                wx_rows, wx_min_date, wx_max_date,
            )
            return {
                "fetched": False,
                "rows":    wx_rows,
                "min_dt":  str(wx_min_date),
                "max_dt":  str(wx_max_date),
            }

    if not need_fetch:
        return {"fetched": False, "rows": wx_rows}

    # Fetch from Open-Meteo archive (free, no quota)
    frames = []
    for apt in AIRPORTS:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   apt["lat"],
            "longitude":  apt["lon"],
            "start_date": str(fetch_start),
            "end_date":   str(fetch_end),
            "hourly":     HOURLY_VARS,
            "timezone":   "GMT",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data.get("hourly", {}))
        df["hour_utc"]     = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["airport_icao"] = apt["icao"]
        df = df.drop(columns=["time"])
        cols = ["airport_icao", "hour_utc"] + [
            c for c in df.columns if c not in ("airport_icao", "hour_utc")
        ]
        df = df[cols]
        frames.append(df)
        logger.info(
            "_fetch_weather_gap: fetched %s (%d rows)", apt["icao"], len(df)
        )
        time.sleep(1)

    combined = pd.concat(frames, ignore_index=True)

    # If we only fetched a gap (not full range), append to existing
    if wx_rows > 0 and fetch_start > required_start:
        combined.to_sql(
            "weather_hourly", engine,
            if_exists="append", index=False,
        )
        logger.info(
            "_fetch_weather_gap: appended %d new rows to weather_hourly",
            len(combined),
        )
    else:
        # Replace entirely (empty table or full re-fetch)
        combined.to_sql(
            "weather_hourly", engine,
            if_exists="replace", index=False,
        )
        logger.info(
            "_fetch_weather_gap: replaced weather_hourly "
            "with %d rows", len(combined),
        )

    return {
        "fetched":      True,
        "rows_added":   len(combined),
        "fetch_start":  str(fetch_start),
        "fetch_end":    str(fetch_end),
    }


def _stage1(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors notebook 2 exactly:
      - Exact duplicate removal
      - Drop dep_pred_utc / arr_pred_utc (100% missing in training data)
      - Weather preprocessing (notebook 1 cleaning applied in-memory)
      - sched_utc, ref_ts_utc, ref_hour_utc
      - Labels: y_delay_min, y_bin15, y_bin30
      - best_utc
      - Reactionary features (leakage-safe)
      - Weather joins: wx_muc_* and wx_other_*
    """

    # ── Step 2: Drop columns that are 100% missing ───────────────────────
    # Notebook 2 first audited missingness, then dropped dep_pred_utc and
    # arr_pred_utc because they were 100% missing in the training dataset.
    # We only drop them here if they are still >= 99% missing in the
    # current dataset. If future API data populates them, they are kept.
    PRED_UTC_COLS = ["dep_pred_utc", "arr_pred_utc"]
    for c in PRED_UTC_COLS:
        if c in flights.columns:
            missing_rate = flights[c].isna().mean()
            if missing_rate >= 0.99:
                flights = flights.drop(columns=[c])
                logger.info(
                    "_stage1: dropped %s (%.1f%% missing — "
                    "treating as uninformative, same as notebook training run)",
                    c, missing_rate * 100,
                )
            else:
                logger.info(
                    "_stage1: keeping %s (%.1f%% missing — "
                    "has enough signal to use in best_utc chain)",
                    c, missing_rate * 100,
                )

    # ── Step 3: Normalize movement ────────────────────────────────────────
    flights["movement"] = (
        flights["movement"].astype(str).str.lower().str.strip()
        .replace({"dep": "departure", "arr": "arrival"})
    )

    # ── Step 4: sched_utc, ref_ts_utc, ref_hour_utc ───────────────────────
    for c in ["dep_sched_utc", "arr_sched_utc"]:
        flights[c] = pd.to_datetime(flights.get(c), utc=True, errors="coerce")

    flights["sched_utc"] = pd.to_datetime(
        pd.Series(
            np.where(flights["movement"].eq("departure"),
                     flights["dep_sched_utc"], flights["arr_sched_utc"]),
            index=flights.index,
        ),
        utc=True, errors="coerce",
    )
    flights["ref_ts_utc"]   = flights["sched_utc"] - pd.Timedelta(hours=2)
    flights["ref_hour_utc"] = flights["ref_ts_utc"].dt.floor("h")

    # ── Step 5: Labels ────────────────────────────────────────────────────
    flights["dep_delay_min"] = pd.to_numeric(
        flights["dep_delay_min"]
        if "dep_delay_min" in flights.columns
        else pd.Series(float("nan"), index=flights.index),
        errors="coerce")
    flights["arr_delay_min"] = pd.to_numeric(
        flights["arr_delay_min"]
        if "arr_delay_min" in flights.columns
        else pd.Series(float("nan"), index=flights.index),
        errors="coerce")
    flights["y_delay_min"] = pd.to_numeric(
        pd.Series(
            np.where(flights["movement"].eq("departure"),
                     flights["dep_delay_min"], flights["arr_delay_min"]),
            index=flights.index,
        ),
        errors="coerce",
    ).clip(lower=-300, upper=720)
    flights["y_bin15"] = (flights["y_delay_min"] >= 15).astype("int8")
    flights["y_bin30"] = (flights["y_delay_min"] >= 30).astype("int8")

    # ── Step 6: best_utc ──────────────────────────────────────────────────
    # dep_pred_utc and arr_pred_utc were dropped in Step 2 so the chain
    # is runway → rev only, exactly as in the notebook training run.
    for c in ["dep_best_utc", "arr_best_utc"]:
        flights[c] = pd.to_datetime(
            flights[c] if c in flights.columns
            else pd.Series(pd.NaT, index=flights.index),
            utc=True, errors="coerce")
    flights["best_utc"] = pd.to_datetime(
        pd.Series(
            np.where(flights["movement"].eq("departure"),
                     flights["dep_best_utc"], flights["arr_best_utc"]),
            index=flights.index,
        ),
        utc=True, errors="coerce",
    )

    # ── Step 7: Reactionary features (leakage-safe) ───────────────────────
    # Notebook 2: sort by [aircraft_modeS, sched_utc], groupby modeS,
    # shift(1), apply leakage guard: prev_best_utc <= ref_ts_utc
    flights["aircraft_modeS"] = (
        flights["aircraft_modeS"].astype(str).str.upper().str.strip()
        if "aircraft_modeS" in flights.columns
        else pd.Series("", index=flights.index)
    )
    flights = flights.sort_values(
        ["aircraft_modeS", "sched_utc"]
    ).reset_index(drop=True)
    g = flights.groupby("aircraft_modeS", sort=False)
    flights["prev_best_utc"]  = g["best_utc"].shift(1)
    flights["prev_delay_min"] = g["y_delay_min"].shift(1)
    knowable = (
        flights["prev_best_utc"].notna() &
        (flights["prev_best_utc"] <= flights["ref_ts_utc"])
    )
    flights["prev_delay_min_safe"] = np.where(
        knowable, flights["prev_delay_min"], np.nan)
    flights["prev_late15_safe"] = np.where(
        knowable, (flights["prev_delay_min"] >= 15).astype("int8"), np.nan)
    flights["prev_late30_safe"] = np.where(
        knowable, (flights["prev_delay_min"] >= 30).astype("int8"), np.nan)

    # ── Step 8: Weather preprocessing (notebook 1 cleaning in-memory) ─────
    # Applied to the weather DataFrame before joining — mirrors notebook 1
    # preprocessing exactly so joined columns match training column state.
    weather = weather.copy()
    weather["hour_utc"]     = pd.to_datetime(
        weather["hour_utc"], utc=True, errors="coerce")
    weather["airport_icao"] = (
        weather["airport_icao"].astype(str).str.upper().str.strip())

    # Notebook 1 Fix 1: drop snow_depth only if 100% missing
    # In the original training data it was 100% missing so it
    # was dropped. Only drop it now if still fully empty.
    if "snow_depth" in weather.columns:
        missing_rate = weather["snow_depth"].isna().mean()
        if missing_rate >= 0.99:
            weather = weather.drop(columns=["snow_depth"])
            logger.info(
                "_stage1: dropped snow_depth from weather "
                "(%.1f%% missing — same as notebook training run)",
                missing_rate * 100,
            )
        else:
            logger.info(
                "_stage1: keeping snow_depth in weather "
                "(%.1f%% missing — has usable values, "
                "will join as wx_muc_snow_depth / wx_other_snow_depth)",
                missing_rate * 100,
            )

    # Notebook 1 Fix 2: fill NaN→0 only for columns where
    # NaN genuinely means "none happened" (not instrument fault).
    # Only apply the fill if the column exists and has NaN values —
    # do not fill if the column is already clean.
    zero_fill_candidates = [
        "precipitation", "rain", "snowfall", "sunshine_duration"
    ]
    for c in zero_fill_candidates:
        if c in weather.columns:
            nan_count = weather[c].isna().sum()
            if nan_count > 0:
                weather[c] = weather[c].fillna(0)
                logger.info(
                    "_stage1: filled %d NaN→0 in weather column %s "
                    "(mirrors notebook 1 fix)",
                    nan_count, c,
                )

    # ── Step 9: Weather joins ──────────────────────────────────────────────
    # Notebook 2: always join EDDM as wx_muc_*, join EDDF/EGLL as
    # wx_other_* only when other_airport_icao matches.
    wx_muc = (
        weather[weather["airport_icao"] == "EDDM"]
        .drop(columns=["airport_icao"], errors="ignore").copy()
    )
    wx_muc = wx_muc.add_prefix("wx_muc_").rename(
        columns={"wx_muc_hour_utc": "ref_hour_utc"})

    wx_other = weather[
        weather["airport_icao"].isin(["EDDF", "EGLL"])
    ].copy()
    wx_other = wx_other.add_prefix("wx_other_").rename(
        columns={"wx_other_airport_icao": "other_airport_icao",
                 "wx_other_hour_utc":     "ref_hour_utc"})

    flights["other_airport_icao"] = (
        flights["other_airport_icao"].astype(str).str.upper().str.strip()
        if "other_airport_icao" in flights.columns
        else pd.Series("", index=flights.index)
    )

    before = len(flights)
    flights = flights.merge(wx_muc, on="ref_hour_utc", how="left")
    logger.info(
        "_stage1: Munich wx joined — rows=%d (expected=%d), "
        "wx_muc missing mean=%.3f%%",
        len(flights), before,
        flights.filter(like="wx_muc_").isna().mean().mean() * 100
        if any(c.startswith("wx_muc_") for c in flights.columns) else 0.0,
    )

    before = len(flights)
    flights = flights.merge(
        wx_other, on=["other_airport_icao", "ref_hour_utc"], how="left")
    other_pct = (
        flights.filter(like="wx_other_").notna().any(axis=1).mean() * 100
        if any(c.startswith("wx_other_") for c in flights.columns) else 0.0
    )
    logger.info(
        "_stage1: other-airport wx joined — rows=%d (expected=%d), "
        "wx_other present=%.2f%%",
        len(flights), before, other_pct,
    )

    return flights


def _stage2(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors build_featured_muc_rxn_wx3_fe.py exactly."""
    df["sched_utc"] = pd.to_datetime(
        df["sched_utc"] if "sched_utc" in df.columns
        else pd.Series(pd.NaT, index=df.index),
        utc=True, errors="coerce")
    df["movement"]           = df["movement"].astype(str).str.lower().str.strip()
    df["airline_icao"] = (
        df["airline_icao"].astype(str).str.upper().str.strip()
        if "airline_icao" in df.columns
        else pd.Series("", index=df.index)
    )
    df["other_airport_icao"] = (
        df["other_airport_icao"].astype(str).str.upper().str.strip()
        if "other_airport_icao" in df.columns
        else pd.Series("", index=df.index)
    )
    df["sched_hour_utc"]     = df["sched_utc"].dt.floor("h")

    hc = (df.groupby(["sched_hour_utc","movement"]).size()
            .rename("cnt").reset_index().sort_values("sched_hour_utc"))
    hc = hc.pivot(index="sched_hour_utc", columns="movement", values="cnt").fillna(0)
    for col in ["departure","arrival"]:
        if col not in hc.columns: hc[col] = 0
    hc = hc.sort_index()
    hc["muc_dep_cnt_pm1h"] = hc["departure"].rolling(3,center=True,min_periods=1).sum()
    hc["muc_arr_cnt_pm1h"] = hc["arrival"].rolling(3,  center=True,min_periods=1).sum()
    hc["muc_dep_cnt_pm2h"] = hc["departure"].rolling(5,center=True,min_periods=1).sum()
    hc["muc_arr_cnt_pm2h"] = hc["arrival"].rolling(5,  center=True,min_periods=1).sum()
    hc = hc.reset_index()[["sched_hour_utc",
                            "muc_dep_cnt_pm1h","muc_arr_cnt_pm1h",
                            "muc_dep_cnt_pm2h","muc_arr_cnt_pm2h"]]
    df = df.merge(hc, on="sched_hour_utc", how="left")

    air_h = (df.groupby(["airline_icao","sched_hour_utc","movement"]).size()
               .rename("cnt").reset_index()
               .sort_values(["airline_icao","sched_hour_utc"]))
    out_rows = []
    for mv in ["departure","arrival"]:
        sub = air_h[air_h["movement"]==mv].copy()
        if sub.empty: continue
        sub = sub.sort_values(["airline_icao","sched_hour_utc"])
        sub[f"air_{mv}_cnt_pm1h"] = (
            sub.groupby("airline_icao")["cnt"]
            .rolling(3,center=True,min_periods=1).sum()
            .reset_index(level=0,drop=True))
        sub[f"air_{mv}_cnt_pm2h"] = (
            sub.groupby("airline_icao")["cnt"]
            .rolling(5,center=True,min_periods=1).sum()
            .reset_index(level=0,drop=True))
        out_rows.append(sub[["airline_icao","sched_hour_utc",
                              f"air_{mv}_cnt_pm1h",f"air_{mv}_cnt_pm2h"]])
    if out_rows:
        air_feat = out_rows[0]
        for extra in out_rows[1:]:
            air_feat = air_feat.merge(extra, on=["airline_icao","sched_hour_utc"], how="outer")
        df = df.merge(air_feat, on=["airline_icao","sched_hour_utc"], how="left")
    for c in ["air_departure_cnt_pm1h","air_departure_cnt_pm2h",
              "air_arrival_cnt_pm1h","air_arrival_cnt_pm2h"]:
        if c in df.columns: df[c] = df[c].fillna(0)

    df = df.sort_values("sched_utc").reset_index(drop=True)
    rk = ["movement","airline_icao","other_airport_icao"]
    df["route_mean_delay_past"] = (
        df.groupby(rk, group_keys=False)["y_delay_min"]
        .apply(lambda s: s.shift(1).expanding(min_periods=10).mean()))
    df["route_rate15_past"] = (
        df.groupby(rk, group_keys=False)["y_bin15"]
        .apply(lambda s: s.shift(1).expanding(min_periods=10).mean()))
    df["air_mean_delay_past"] = (
        df.groupby(["movement","airline_icao"], group_keys=False)["y_delay_min"]
        .apply(lambda s: s.shift(1).expanding(min_periods=20).mean()))

    gm  = pd.to_numeric(df["y_delay_min"],errors="coerce").mean()
    gr15 = pd.to_numeric(df["y_bin15"],   errors="coerce").mean()
    df["route_mean_delay_past"] = df["route_mean_delay_past"].fillna(gm)
    df["route_rate15_past"]     = df["route_rate15_past"].fillna(gr15)
    df["air_mean_delay_past"]   = df["air_mean_delay_past"].fillna(gm)

    df["muc_wind_strong"]  = (
        (df["wx_muc_wind_speed_10m"] if "wx_muc_wind_speed_10m" in df.columns
         else pd.Series(0, index=df.index)) >= 25
    ).astype("int8")
    df["muc_gust_strong"]  = (
        (df["wx_muc_wind_gusts_10m"] if "wx_muc_wind_gusts_10m" in df.columns
         else pd.Series(0, index=df.index)) >= 40
    ).astype("int8")
    df["muc_precip_any"]   = (
        (df["wx_muc_precipitation"] if "wx_muc_precipitation" in df.columns
         else pd.Series(0, index=df.index)) >  0
    ).astype("int8")
    df["muc_snow_any"]     = (
        (df["wx_muc_snowfall"] if "wx_muc_snowfall" in df.columns
         else pd.Series(0, index=df.index)) >  0
    ).astype("int8")
    df["other_wind_strong"] = (
        ((df["wx_other_wind_speed_10m"] if "wx_other_wind_speed_10m" in df.columns
          else pd.Series(np.nan, index=df.index)) >= 25)
        .fillna(0).astype("int8")
    )
    df["other_gust_strong"] = (
        ((df["wx_other_wind_gusts_10m"] if "wx_other_wind_gusts_10m" in df.columns
          else pd.Series(np.nan, index=df.index)) >= 40)
        .fillna(0).astype("int8")
    )
    df["other_precip_any"]  = (
        ((df["wx_other_precipitation"] if "wx_other_precipitation" in df.columns
          else pd.Series(np.nan, index=df.index)) >  0)
        .fillna(0).astype("int8")
    )
    df["other_snow_any"]    = (
        ((df["wx_other_snowfall"] if "wx_other_snowfall" in df.columns
          else pd.Series(np.nan, index=df.index)) >  0)
        .fillna(0).astype("int8")
    )

    data_dir     = Path(__file__).parent / "data"
    traffic_path = data_dir / "traffic_munich_daily.parquet"
    atfm_path    = data_dir / "atfm_delay_munich_daily.parquet"
    if traffic_path.exists() and atfm_path.exists():
        traffic = pd.read_parquet(traffic_path)
        atfm    = pd.read_parquet(atfm_path)
        df["ref_date"] = pd.to_datetime(
            df["ref_ts_utc"] if "ref_ts_utc" in df.columns
            else pd.Series(pd.NaT, index=df.index),
            utc=True, errors="coerce").dt.date
        traffic["date"] = pd.to_datetime(traffic["date"],errors="coerce").dt.date
        atfm["date"]    = pd.to_datetime(atfm["date"],errors="coerce").dt.date
        traffic = traffic[traffic["airport"].astype(str).str.upper().eq("EDDM")].copy()
        atfm    = atfm[atfm["airport"].astype(str).str.upper().eq("EDDM")].copy()
        tf = (traffic.drop(columns=["airport"],errors="ignore")
              .add_prefix("ans_traffic_")
              .rename(columns={"ans_traffic_date":"ref_date"}))
        af = (atfm.drop(columns=["airport","ATFM_VERSION"],errors="ignore")
              .add_prefix("ans_atfm_")
              .rename(columns={"ans_atfm_date":"ref_date"}))
        df = df.merge(tf, on="ref_date", how="left")
        df = df.merge(af, on="ref_date", how="left")
        logger.info("ANSPerf joined.")
    else:
        logger.info("ANSPerf parquet files not found — skipping ans_* features.")

    df["ref_year"] = pd.to_datetime(
        df["ref_ts_utc"] if "ref_ts_utc" in df.columns
        else pd.Series(pd.NaT, index=df.index),
        utc=True, errors="coerce").dt.year
    if "prev_delay_min_safe" in df.columns:
        df["prev_delay_min_safe"] = pd.to_numeric(
            df["prev_delay_min_safe"],errors="coerce").clip(lower=-180,upper=600)
    df["sched_hour_utc"] = (pd.to_datetime(df["sched_utc"],utc=True).dt.hour
                            if "sched_utc" in df.columns else -1)
    return df


def _cause_group(f: str) -> str:
    n = f.lower()
    if n.startswith("wx_muc_") or n.startswith("wx_other_") or \
       n.startswith("muc_") or n.startswith("other_"): return "Weather"
    if "prev_delay" in n or "prev_late" in n: return "Reactionary"
    if "muc_dep_cnt" in n or "muc_arr_cnt" in n or \
       re.search(r"\bair_(departure|arrival)_cnt\b",n): return "Congestion"
    if n.startswith("ans_atfm_"):   return "ATFM constraints"
    if n.startswith("ans_traffic_"): return "Airport traffic"
    if n.startswith("route_") or n.startswith("air_mean_delay"): return "Historical patterns"
    if n.startswith("sched_") or n in ["is_weekend"]: return "Time/seasonality"
    if n in ["airline_icao","airline_iata","other_airport_icao",
             "other_airport_iata","movement","is_cargo","aircraft_model"]:
        return "Airline/route identity"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_retrain(job_id: int) -> None:
    conn = _get_conn()
    try:
        _execute(conn, job_id)
    except Exception as exc:
        logger.exception("[job=%d] Unhandled exception", job_id)
        try:
            _fail_job(conn, job_id, "unexpected_error", str(exc))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _execute(conn, job_id: int) -> None:  # noqa: C901
    engine = _get_engine()

    # Mark job as started immediately
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE retrain_jobs
                  SET status='running', started_at=NOW()
                WHERE id=%s""",
            (job_id,),
        )
    conn.commit()
    logger.info("[job=%d] Job marked as running", job_id)

    if _is_cancelled(conn, job_id):
        logger.info("[job=%d] Job cancelled before starting", job_id)
        return

    # ── STEP 1: Determine date range ─────────────────────────────────────────
    _update_job(conn, job_id, "determining_date_range",
                "Reading last model training end date")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT data_date_to, version_tag FROM model_versions WHERE is_current=TRUE"
        )
        row = cur.fetchone()
    if row is None:
        _fail_job(conn, job_id, "determining_date_range",
                  "No current model found in model_versions."); return

    last_end: date = row[0]
    current_tag   = row[1]
    backfill_start = last_end + timedelta(days=1)
    backfill_end   = date.today() - timedelta(days=1)

    if backfill_start > backfill_end:
        _fail_job(conn, job_id, "determining_date_range",
                  f"No new data. Model '{current_tag}' covers to {last_end}. "
                  f"Yesterday is {backfill_end}."); return

    days = [backfill_start + timedelta(days=i)
            for i in range((backfill_end - backfill_start).days + 1)]
    calls_needed = len(days) * 2
    units_needed = calls_needed * FIDS_UNITS_PER_CALL

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE retrain_jobs SET backfill_start=%s,backfill_end=%s WHERE id=%s",
            (backfill_start, backfill_end, job_id),
        )
    conn.commit()
    _update_job(conn, job_id, "determining_date_range",
                f"Range: {backfill_start} → {backfill_end} "
                f"({len(days)} days, {calls_needed} API calls needed)")

    if _is_cancelled(conn, job_id): return

    # ── STEP 2: Quota check ───────────────────────────────────────────────────
    _update_job(conn, job_id, "quota_check",
                "Checking monthly API quota before any call")

    monthly_req_cap   = int(os.getenv("RAPIDAPI_MONTHLY_REQUESTS", "40000"))
    monthly_units_cap = int(os.getenv("RAPIDAPI_MONTHLY_UNITS",    "60000"))
    month_start       = date.today().replace(day=1)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM retrain_api_calls WHERE called_at>=%s AND status='ok'",
            (month_start,),
        )
        calls_used = cur.fetchone()[0]

    req_remaining   = monthly_req_cap   - calls_used
    units_remaining = monthly_units_cap - (calls_used * FIDS_UNITS_PER_CALL)

    if calls_needed > req_remaining:
        _fail_job(conn, job_id, "quota_check",
                  f"Insufficient request quota. Need {calls_needed}, "
                  f"{req_remaining} remaining (cap={monthly_req_cap}, used={calls_used}).")
        return
    if units_needed > units_remaining:
        _fail_job(conn, job_id, "quota_check",
                  f"Insufficient unit quota. Need {units_needed} units, "
                  f"{units_remaining} remaining.")
        return

    _update_job(conn, job_id, "quota_check",
                f"Quota OK — {req_remaining} requests / {units_remaining} units remaining. "
                f"This job needs {calls_needed} requests / {units_needed} units.")

    if _is_cancelled(conn, job_id): return

    # ── STEP A: Prepare ANSPerformance data ──────────────────────────────────
    _update_job(conn, job_id, "preparing_ansperf",
                "Checking for ANSPerformance CSV files in data/ folder")

    if _is_cancelled(conn, job_id): return

    data_dir = Path(__file__).parent / "data"
    try:
        ansperf_result = _prepare_ansperf(data_dir)
        _update_job(conn, job_id, "preparing_ansperf",
                    f"ANSPerf ready — traffic: {ansperf_result['traffic_rows']} rows "
                    f"({ansperf_result['traffic_date_range']}) | "
                    f"ATFM: {ansperf_result['atfm_rows']} rows "
                    f"({ansperf_result['atfm_date_range']})")
    except FileNotFoundError as exc:
        # ANSPerf files not found — non-fatal, training proceeds
        # without ANSPerf features (CatBoost handles NaN natively)
        logger.warning(
            "[job=%d] ANSPerf files not found (non-fatal): %s",
            job_id, exc,
        )
        _update_job(conn, job_id, "preparing_ansperf",
                    f"WARNING: {exc} — training will proceed without "
                    f"ANSPerf features (NaN handled by CatBoost).")
    except Exception as exc:
        logger.warning(
            "[job=%d] ANSPerf preparation failed (non-fatal): %s",
            job_id, exc,
        )
        _update_job(conn, job_id, "preparing_ansperf",
                    f"WARNING: ANSPerf prep failed: {exc} — "
                    f"proceeding without ANSPerf features.")

    if _is_cancelled(conn, job_id): return

    # ── STEP B: Fetch historical weather gap ─────────────────────────────────
    _update_job(conn, job_id, "fetching_weather",
                "Checking weather_hourly coverage and fetching any gap "
                "from Open-Meteo archive API (free, no quota impact)")

    try:
        wx_result = _fetch_weather_gap(engine, backfill_start, backfill_end)
        if wx_result.get("fetched"):
            _update_job(conn, job_id, "fetching_weather",
                        f"Weather gap filled — fetched {wx_result['rows_added']:,} rows "
                        f"({wx_result['fetch_start']} → {wx_result['fetch_end']})")
        else:
            _update_job(conn, job_id, "fetching_weather",
                        f"Weather already covers full range — "
                        f"{wx_result.get('rows', 0):,} EDDM rows, "
                        f"{wx_result.get('min_dt')} → {wx_result.get('max_dt')}. "
                        f"No fetch needed.")
    except Exception as exc:
        logger.warning(
            "[job=%d] Weather fetch failed (non-fatal): %s", job_id, exc
        )
        _update_job(conn, job_id, "fetching_weather",
                    f"WARNING: Weather fetch failed: {exc} — "
                    f"weather features may be NaN.")

    if _is_cancelled(conn, job_id): return

    # ── STEP C: Pre-flight verification ──────────────────────────────────────
    _update_job(conn, job_id, "pre_flight_verification",
                "Verifying data integrity before training")

    errors = []

    # Check base rows
    base_count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM training_flights_base",
        engine,
    ).iloc[0]["cnt"]
    if base_count == 0:
        errors.append(
            "training_flights_base is empty — run "
            "import_base_to_supabase.py to import the 365-day parquet."
        )

    # Check backfill rows for target date range
    bf_count = pd.read_sql(
        """SELECT COUNT(*) as cnt FROM training_flights_backfill
            WHERE flight_date >= %(s)s AND flight_date <= %(e)s""",
        engine,
        params={"s": backfill_start, "e": backfill_end},
    ).iloc[0]["cnt"]
    if bf_count == 0:
        errors.append(
            f"training_flights_backfill has no rows for "
            f"{backfill_start} → {backfill_end}."
        )

    # Check weather coverage
    wx_count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM weather_hourly WHERE airport_icao='EDDM'",
        engine,
    ).iloc[0]["cnt"]
    if wx_count == 0:
        errors.append(
            "weather_hourly is empty — weather features will be NaN."
        )

    if errors:
        # Only abort if base flights are missing — that is critical.
        # Weather and backfill warnings are logged but non-fatal.
        critical = [e for e in errors if "training_flights_base" in e]
        if critical:
            _fail_job(conn, job_id, "pre_flight_verification",
                      " | ".join(critical))
            return
        else:
            _update_job(conn, job_id, "pre_flight_verification",
                        "Warnings: " + " | ".join(errors) +
                        " — proceeding with available data.")
    else:
        _update_job(conn, job_id, "pre_flight_verification",
                    f"All checks passed — "
                    f"base: {base_count:,} rows | "
                    f"backfill: {bf_count:,} rows | "
                    f"weather EDDM: {wx_count:,} rows.")

    if _is_cancelled(conn, job_id): return

    # ── STEP 3: FIDS backfill — main loop ────────────────────────────────────
    _update_job(conn, job_id, "fids_backfill",
                f"Starting main backfill loop: {len(days)} days")

    if not RAPIDAPI_KEY:
        _fail_job(conn, job_id, "fids_backfill",
                  "RAPIDAPI_KEY not set in environment."); return

    # Check which dates are already in DB (resumability)
    already_done = _already_ingested_dates(conn, job_id)
    skipped = len([d for d in days if d in already_done])
    if skipped:
        logger.info("[job=%d] Skipping %d already-ingested dates", job_id, skipped)

    day_stats = {}   # date → stats dict (mirrors notebook's stats list)

    for idx, d in enumerate(days, 1):
        if d in already_done:
            continue

        day_str   = d.isoformat()
        all_parts = []
        day_ok    = True

        windows = [
            ("T00:00-T11:59", f"{day_str}T00:00", f"{day_str}T11:59"),
            ("T12:00-T23:59", f"{day_str}T12:00", f"{day_str}T23:59"),
        ]

        for window_label, from_local, to_local in windows:
            try:
                data = _fetch_fids(from_local, to_local, RAPIDAPI_KEY)
                part = _normalize_fids(data, "MUC")
                all_parts.append(part)
                _log_api_call(conn, job_id, d, window_label, "ok",
                              rows_fetched=len(part))
            except Exception as exc:
                _log_api_call(conn, job_id, d, window_label, "failed",
                              error_detail=str(exc))
                logger.warning("[job=%d] Window failed %s %s: %s",
                               job_id, day_str, window_label, exc)
                day_ok = False
            time.sleep(0.25)   # same rate-limit sleep as notebook

        if all_parts:
            df_day = pd.concat(all_parts, ignore_index=True)
            df_day = _process_day_df(df_day)
            _insert_day(conn, df_day, job_id, d)
            day_stats[d] = _day_stats(df_day)
        else:
            day_stats[d] = {"ok": False, "error": "both windows failed"}

        if idx % 10 == 0:
            ok_so_far = sum(1 for s in day_stats.values()
                            if isinstance(s, dict) and s.get("ok", True)
                            and "error" not in s)
            _update_job(conn, job_id, "fids_backfill",
                        f"{idx}/{len(days)} days processed | "
                        f"{ok_so_far} successful so far")

    # ── STEP 3b: Retry failed days — mirrors notebook retry cell ─────────────
    failed_windows = _failed_windows(conn, job_id)

    # Group failed windows by day
    failed_days_map: dict[str, list] = {}
    for day_str, window_label in failed_windows:
        failed_days_map.setdefault(day_str, []).append(window_label)

    if failed_days_map:
        _update_job(conn, job_id, "fids_backfill_retry",
                    f"Retrying {len(failed_days_map)} days that had failed windows")
        logger.info("[job=%d] Retry pass: %d days", job_id, len(failed_days_map))

        for day_str, failed_wlabels in failed_days_map.items():
            d = date.fromisoformat(day_str)
            all_parts = []

            windows = [
                ("T00:00-T11:59", f"{day_str}T00:00", f"{day_str}T11:59"),
                ("T12:00-T23:59", f"{day_str}T12:00", f"{day_str}T23:59"),
            ]
            for window_label, from_local, to_local in windows:
                if window_label not in failed_wlabels:
                    continue  # only retry the ones that actually failed
                try:
                    data = _fetch_fids(from_local, to_local, RAPIDAPI_KEY)
                    part = _normalize_fids(data, "MUC")
                    all_parts.append(part)
                    _log_api_call(conn, job_id, d, window_label, "ok",
                                  rows_fetched=len(part))
                    logger.info("[job=%d] Retried OK: %s %s rows=%d",
                                job_id, day_str, window_label, len(part))
                except Exception as exc:
                    _log_api_call(conn, job_id, d, window_label, "failed",
                                  error_detail=f"retry: {exc}")
                    logger.warning("[job=%d] Retry still failing %s %s: %s",
                                   job_id, day_str, window_label, exc)
                time.sleep(0.35)   # slightly longer sleep on retry, same as notebook

            if all_parts:
                df_retry = pd.concat(all_parts, ignore_index=True)
                df_retry = _process_day_df(df_retry)
                _insert_day(conn, df_retry, job_id, d)

    # ── STEP 3c: Coverage summary ─────────────────────────────────────────
    # Count rows for the target date range across ALL jobs —
    # data from previous job runs is valid and must not be ignored.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(DISTINCT flight_date)
               FROM training_flights_backfill
               WHERE flight_date >= %s AND flight_date <= %s""",
            (backfill_start, backfill_end),
        )
        ok_days = cur.fetchone()[0]

        cur.execute(
            """SELECT COUNT(*)
               FROM training_flights_backfill
               WHERE flight_date >= %s AND flight_date <= %s""",
            (backfill_start, backfill_end),
        )
        total_rows = cur.fetchone()[0]

        cur.execute(
            """SELECT COUNT(*)
               FROM retrain_api_calls
               WHERE job_id=%s AND status='failed'""",
            (job_id,),
        )
        still_failing_calls = cur.fetchone()[0]

    coverage_summary = (
        f"Backfill complete: {ok_days}/{len(days)} days OK | "
        f"{total_rows:,} rows available in date range | "
        f"{still_failing_calls} API calls still failing after retry"
    )
    logger.info("[job=%d] %s", job_id, coverage_summary)
    _update_job(conn, job_id, "fids_backfill_complete", coverage_summary)

    if _is_cancelled(conn, job_id): return

    if total_rows == 0:
        _fail_job(conn, job_id, "fids_backfill_complete",
                  "Zero rows in training_flights_backfill for the "
                  "target date range — no flight data available.")
        return

    # ── STEP 4: Weather coverage check ───────────────────────────────────
    # Check whether weather_hourly already covers the training date range.
    # If it does, skip the live fetch entirely — ingest_weather_live uses
    # the forecast endpoint which only returns ~7 days and would OVERWRITE
    # the historical data with useless forecast rows.
    # Historical weather must be loaded separately via
    # fetch_historical_weather.py before triggering a retrain.
    _update_job(conn, job_id, "weather_gap_fill",
                "Checking weather_hourly coverage for training date range")

    try:
        weather_check = pd.read_sql(
            """SELECT MIN(hour_utc) as min_dt, MAX(hour_utc) as max_dt,
                      COUNT(*) as row_count
               FROM weather_hourly
               WHERE airport_icao = 'EDDM'""",
            engine,
        )
        wx_min  = weather_check.iloc[0]["min_dt"]
        wx_max  = weather_check.iloc[0]["max_dt"]
        wx_rows = int(weather_check.iloc[0]["row_count"])

        be_aware = pd.Timestamp(backfill_end).tz_localize("UTC")

        if wx_rows == 0:
            _update_job(conn, job_id, "weather_gap_fill",
                        "WARNING: weather_hourly is empty. Weather features "
                        "will be NaN. Run fetch_historical_weather.py before "
                        "next retrain.")
            logger.warning(
                "[job=%d] weather_hourly is empty — weather features will "
                "be NaN. Run: python fetch_historical_weather.py "
                "--start 2025-02-22 --end %s",
                job_id, backfill_end,
            )
        elif wx_max is not None and pd.Timestamp(wx_max) < be_aware:
            gap_days = (be_aware - pd.Timestamp(wx_max)).days
            _update_job(conn, job_id, "weather_gap_fill",
                        f"WARNING: weather_hourly covers up to "
                        f"{wx_max} but training needs up to "
                        f"{backfill_end} ({gap_days} days gap). "
                        f"Run fetch_historical_weather.py to fill gap.")
            logger.warning(
                "[job=%d] Weather gap of %d days. Run: "
                "python fetch_historical_weather.py "
                "--start %s --end %s",
                job_id, gap_days, wx_max, backfill_end,
            )
        else:
            _update_job(conn, job_id, "weather_gap_fill",
                        f"Weather OK — {wx_rows:,} EDDM rows covering "
                        f"{wx_min} → {wx_max}. No fetch needed.")
            logger.info(
                "[job=%d] Weather coverage OK: %d rows, %s → %s",
                job_id, wx_rows, wx_min, wx_max,
            )

    except Exception as exc:
        logger.warning(
            "[job=%d] Weather coverage check failed (non-fatal): %s",
            job_id, exc,
        )
        _update_job(conn, job_id, "weather_gap_fill",
                    f"Weather check failed (non-fatal): {exc}")

    # ── STEP 5: Build union dataset ───────────────────────────────────────────
    _update_job(conn, job_id, "building_union_dataset",
                "Loading training_flights_base + training_flights_backfill")

    base_df = pd.read_sql("SELECT * FROM training_flights_base", engine)
    bf_df   = pd.read_sql("SELECT * FROM training_flights_backfill", engine)
    logger.info("[job=%d] Base: %d rows | Backfill: %d rows",
                job_id, len(base_df), len(bf_df))

    bf_df   = bf_df.drop(columns=["id","job_id","flight_date"], errors="ignore")
    base_df = base_df.drop(columns=["id"], errors="ignore")

    union_df = pd.concat([base_df, bf_df], ignore_index=True)

    # Step A: exact-row dedup (mirrors notebook 2 exactly)
    # Removes rows identical across all columns — handles
    # overlapping 12h FIDS windows from the API.
    before_exact = len(union_df)
    union_df = union_df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)
    logger.info(
        "[job=%d] Exact-row dedup: %d → %d rows (dropped %d)",
        job_id, before_exact, len(union_df),
        before_exact - len(union_df),
    )

    # Step B: key-based dedup (base vs backfill overlap)
    # When the same flight appears in both training_flights_base
    # and training_flights_backfill, keep the row with the most
    # non-null actual time fields (the more complete record).
    dedup_key = ["number_raw","dep_sched_utc","movement"]
    actual_ts = ["dep_runway_utc","dep_rev_utc","arr_runway_utc","arr_rev_utc"]
    union_df["_actual_count"] = union_df[actual_ts].notna().sum(axis=1)
    before_key = len(union_df)
    union_df = (union_df
                .sort_values("_actual_count", ascending=False)
                .drop_duplicates(subset=dedup_key, keep="first")
                .drop(columns=["_actual_count"])
                .reset_index(drop=True))
    logger.info(
        "[job=%d] Key-based dedup: %d → %d rows (dropped %d)",
        job_id, before_key, len(union_df),
        before_key - len(union_df),
    )

    _update_job(conn, job_id, "building_union_dataset",
                f"Union dataset: {len(union_df):,} rows after deduplication")

    if _is_cancelled(conn, job_id): return

    # ── STEP 6: Feature engineering stage 1 ──────────────────────────────────
    _update_job(conn, job_id, "feature_engineering_stage1",
                "Reactionary features + weather joins")
    if _is_cancelled(conn, job_id): return
    weather_df = pd.read_sql("SELECT * FROM weather_hourly", engine)
    union_df   = _stage1(union_df, weather_df)

    # ── STEP 7: Feature engineering stage 2 ──────────────────────────────────
    _update_job(conn, job_id, "feature_engineering_stage2",
                "Congestion, history, weather flags, ANSPerf join")
    if _is_cancelled(conn, job_id): return
    union_df = _stage2(union_df)

    # ── STEP 8: Time features + drop leakage ─────────────────────────────────
    _update_job(conn, job_id, "preparing_feature_matrix",
                "Adding time features, dropping leakage columns")
    if _is_cancelled(conn, job_id): return

    union_df["sched_utc"] = pd.to_datetime(
        union_df["sched_utc"], utc=True, errors="coerce")
    union_df["sched_hour"]  = union_df["sched_utc"].dt.hour.astype("int16")
    union_df["sched_dow"]   = union_df["sched_utc"].dt.dayofweek.astype("int16")
    union_df["sched_month"] = union_df["sched_utc"].dt.month.astype("int16")
    union_df["is_weekend"]  = union_df["sched_dow"].isin([5,6]).astype("int8")

    y15  = union_df["y_bin15"].astype(int)
    y30  = union_df["y_bin30"].astype(int)
    y_reg = pd.to_numeric(union_df["y_delay_min"], errors="coerce").clip(upper=720)

    drop_cols = set(
        [c for c in LEAK_OR_LABEL_COLS if c in union_df.columns] +
        [c for c in TIME_KEYS          if c in union_df.columns]
    )
    X = union_df.drop(columns=list(drop_cols), errors="ignore").copy()
    logger.info("[job=%d] Feature matrix X: %s", job_id, X.shape)

    # ── STEP 9: Time-based split 75/15/10 ────────────────────────────────────
    _update_job(conn, job_id, "splitting_data",
                "Time-based split 75/15/10 by sched_utc")
    if _is_cancelled(conn, job_id): return

    order   = union_df["sched_utc"].sort_values().index
    n       = len(order)
    n_train = int(n * 0.75)
    n_valid = int(n * 0.15)

    train_idx = order[:n_train]
    valid_idx = order[n_train:n_train+n_valid]
    test_idx  = order[n_train+n_valid:]

    X_train = X.loc[train_idx]; X_valid = X.loc[valid_idx]; X_test = X.loc[test_idx]
    y15_train = y15.loc[train_idx]; y15_valid = y15.loc[valid_idx]; y15_test = y15.loc[test_idx]
    y30_train = y30.loc[train_idx]; y30_valid = y30.loc[valid_idx]; y30_test = y30.loc[test_idx]
    yr_train  = y_reg.loc[train_idx]; yr_valid = y_reg.loc[valid_idx]; yr_test = y_reg.loc[test_idx]

    # ── STEP 10: Prepare CatBoost matrices ───────────────────────────────────
    _update_job(conn, job_id, "preparing_catboost_matrices",
                "Encoding categoricals, imputing numerics")
    if _is_cancelled(conn, job_id): return

    cat_cols = [c for c in X_train.columns
                if X_train[c].dtype=="object" or
                   str(X_train[c].dtype)=="category" or
                   X_train[c].dtype=="bool"]

    X_tr = X_train.copy(); X_va = X_valid.copy(); X_te = X_test.copy()

    for c in cat_cols:
        for df_ in [X_tr, X_va, X_te]:
            df_[c] = df_[c].astype("object").fillna("__MISSING__")

    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    for c in num_cols:
        for df_ in [X_tr, X_va, X_te]:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")
    for df_ in [X_tr, X_va, X_te]:
        df_.replace([np.inf,-np.inf], np.nan, inplace=True)

    num_medians = X_tr[num_cols].median(numeric_only=True).to_dict()
    for c in num_cols:
        for df_ in [X_tr, X_va, X_te]:
            df_[c] = df_[c].fillna(num_medians.get(c,0)).fillna(0).astype("float64")

    # ── STEP 11: Train all three models ──────────────────────────────────────
    _update_job(conn, job_id, "training_clf15",
                "Training CatBoostClassifier for delay >= 15 min")
    pos15 = float(y15_train.mean())
    clf15 = CatBoostClassifier(**CLF_PARAMS,
                               class_weights=[1.0, float((1-pos15)/pos15)])
    clf15.fit(X_tr, y15_train, cat_features=cat_cols,
              eval_set=(X_va, y15_valid), use_best_model=True)

    _update_job(conn, job_id, "training_clf30",
                "Training CatBoostClassifier for delay >= 30 min")
    pos30 = float(y30_train.mean())
    clf30 = CatBoostClassifier(**CLF_PARAMS,
                               class_weights=[1.0, float((1-pos30)/pos30)])
    clf30.fit(X_tr, y30_train, cat_features=cat_cols,
              eval_set=(X_va, y30_valid), use_best_model=True)

    _update_job(conn, job_id, "training_reg2",
                "Training CatBoostRegressor for delay minutes (>= 5 min)")
    mtr = yr_train.notna() & (yr_train >= 5)
    mva = yr_valid.notna() & (yr_valid >= 5)
    mte = yr_test.notna()  & (yr_test  >= 5)
    reg2 = CatBoostRegressor(**REG_PARAMS)
    reg2.fit(X_tr.loc[mtr], yr_train.loc[mtr], cat_features=cat_cols,
             eval_set=(X_va.loc[mva], yr_valid.loc[mva]), use_best_model=True)

    # ── STEP 12: Threshold tuning ─────────────────────────────────────────────
    _update_job(conn, job_id, "threshold_tuning",
                "Grid search thresholds on validation set (F1 criterion)")

    def _best_threshold(clf, Xv, yv):
        probs = clf.predict_proba(Xv)[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.round(np.linspace(0.1, 0.9, 17), 2):
            yhat = (probs >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                yv, yhat, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    best_t15 = _best_threshold(clf15, X_va, y15_valid)
    best_t30 = _best_threshold(clf30, X_va, y30_valid)

    # ── STEP 13: Evaluate on test set ─────────────────────────────────────────
    _update_job(conn, job_id, "evaluating",
                "Computing AUC, PR-AUC, MAE on held-out test set")

    p15_test = clf15.predict_proba(X_te)[:, 1]
    p30_test = clf30.predict_proba(X_te)[:, 1]
    auc15    = float(roc_auc_score(y15_test, p15_test))
    prauc15  = float(average_precision_score(y15_test, p15_test))
    auc30    = float(roc_auc_score(y30_test, p30_test))
    prauc30  = float(average_precision_score(y30_test, p30_test))
    mae_reg  = float(mean_absolute_error(yr_test.loc[mte],
                                         reg2.predict(X_te.loc[mte])))

    candidate_metrics = dict(auc15=auc15, prauc15=prauc15,
                             auc30=auc30, prauc30=prauc30, mae_reg=mae_reg)
    _update_job(conn, job_id, "evaluating",
                f"AUC15={auc15:.4f} PR-AUC15={prauc15:.4f} | "
                f"AUC30={auc30:.4f} PR-AUC30={prauc30:.4f} | MAE={mae_reg:.2f}")

    # ── STEP 14: Compare against production ───────────────────────────────
    _update_job(conn, job_id, "comparing_metrics",
                "Comparing candidate vs current production model")

    with conn.cursor() as cur:
        cur.execute(
            """SELECT auc15_test, prauc15_test, auc30_test, prauc30_test,
                      version_tag, artifact_dir,
                      threshold_bin15, threshold_bin30
               FROM model_versions WHERE is_current=TRUE"""
        )
        prod_row = cur.fetchone()

    if prod_row:
        prod = dict(
            auc15=prod_row[0], prauc15=prod_row[1],
            auc30=prod_row[2], prauc30=prod_row[3],
        )
        prod_tag      = prod_row[4]
        prod_art_dir  = prod_row[5]
        prod_thr15    = prod_row[6]
        prod_thr30    = prod_row[7]
    else:
        prod = PRODUCTION_METRICS
        prod_tag     = "v3_final"
        prod_art_dir = str(Path(__file__).parent / "models" / "v3_final")
        prod_thr15   = 0.30
        prod_thr30   = 0.40

    # Determine which individual models beat production
    clf15_better = (auc15   >= prod["auc15"]   and
                    prauc15 >= prod["prauc15"])
    clf30_better = (auc30   >= prod["auc30"]   and
                    prauc30 >= prod["prauc30"])

    comparison_detail = (
        f"clf15: AUC {auc15:.4f} vs {prod['auc15']:.4f}, "
        f"PR-AUC {prauc15:.4f} vs {prod['prauc15']:.4f} "
        f"→ {'BETTER' if clf15_better else 'WORSE'} | "
        f"clf30: AUC {auc30:.4f} vs {prod['auc30']:.4f}, "
        f"PR-AUC {prauc30:.4f} vs {prod['prauc30']:.4f} "
        f"→ {'BETTER' if clf30_better else 'WORSE'} | "
        f"MAE reg2: {mae_reg:.2f} min"
    )
    logger.info("[job=%d] Comparison: %s", job_id, comparison_detail)
    _update_job(conn, job_id, "comparing_metrics", comparison_detail)

    if not clf15_better and not clf30_better:
        # Neither model improved — reject entirely
        rejection = (
            f"Neither clf15 nor clf30 beat '{prod_tag}'. "
            f"{comparison_detail}"
        )
        _finish_job(conn, job_id, "rejected",
                    metrics=candidate_metrics, error=rejection)
        logger.info("[job=%d] REJECTED — no improvement on any model", job_id)
        return

    # ── STEP 15: Partial or full promotion ────────────────────────────────
    _update_job(conn, job_id, "promoting",
                f"Promoting: clf15={'NEW' if clf15_better else 'KEEP v3'} | "
                f"clf30={'NEW' if clf30_better else 'KEEP v3'} | "
                f"reg2=NEW (always updated with new training data)")

    now_str = datetime.now(timezone.utc).strftime("%Y_%m")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM model_versions")
        version_num = cur.fetchone()[0] + 1
    new_tag       = f"v{version_num}_retrain_{now_str}"
    models_root   = Path(__file__).parent / "models"
    candidate_dir = models_root / f"candidate_job{job_id}"
    final_dir     = models_root / new_tag
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Read actual start date of base training data from DB
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(dep_sched_utc)::date FROM training_flights_base"
        )
        base_start_row = cur.fetchone()
    data_date_from = (
        base_start_row[0]
        if base_start_row and base_start_row[0]
        else backfill_start - timedelta(days=365)
    )

    # ── Decide which .cbm files to use for each model ─────────────────────
    # If the new model is better, save and use the new one.
    # If not, copy the production model file so the new version
    # folder is self-contained and the metadata paths are consistent.

    bin15_path = candidate_dir / f"cb_bin15_{new_tag}.cbm"
    bin30_path = candidate_dir / f"cb_bin30_{new_tag}.cbm"
    reg_path   = candidate_dir / f"cb_reg_delay_ge5_{new_tag}.cbm"

    if clf15_better:
        clf15.save_model(str(bin15_path))
        thr15_final = best_t15
        logger.info("[job=%d] clf15: using NEW model (AUC %.4f > %.4f)",
                    job_id, auc15, prod["auc15"])
    else:
        # Copy production clf15 into the new version folder
        prod_bin15 = Path(prod_art_dir) / next(
            (f for f in Path(prod_art_dir).iterdir()
             if f.name.startswith("cb_bin15_") and f.suffix == ".cbm"),
            Path(prod_art_dir) / "cb_bin15_v3_final.cbm",
        ).name if Path(prod_art_dir).exists() else None
        if prod_bin15 and prod_bin15.exists():
            import shutil as _shutil
            _shutil.copy2(str(prod_bin15), str(bin15_path))
        else:
            # fallback: save the new one anyway
            clf15.save_model(str(bin15_path))
        thr15_final = prod_thr15
        logger.info("[job=%d] clf15: keeping PRODUCTION model (AUC %.4f <= %.4f)",
                    job_id, auc15, prod["auc15"])

    if clf30_better:
        clf30.save_model(str(bin30_path))
        thr30_final = best_t30
        logger.info("[job=%d] clf30: using NEW model (AUC %.4f > %.4f)",
                    job_id, auc30, prod["auc30"])
    else:
        prod_bin30 = Path(prod_art_dir) / next(
            (f for f in Path(prod_art_dir).iterdir()
             if f.name.startswith("cb_bin30_") and f.suffix == ".cbm"),
            Path(prod_art_dir) / "cb_bin30_v3_final.cbm",
        ).name if Path(prod_art_dir).exists() else None
        if prod_bin30 and prod_bin30.exists():
            import shutil as _shutil
            _shutil.copy2(str(prod_bin30), str(bin30_path))
        else:
            clf30.save_model(str(bin30_path))
        thr30_final = prod_thr30
        logger.info("[job=%d] clf30: keeping PRODUCTION model (AUC %.4f <= %.4f)",
                    job_id, auc30, prod["auc30"])

    # reg2 always uses the newly trained version — it benefits from
    # more training data regardless of classifier improvement
    reg2.save_model(str(reg_path))

    # ── Build metadata and cause groups ──────────────────────────────────
    feature_list = X_tr.columns.tolist()
    cause_groups = {f: _cause_group(f) for f in feature_list}

    grp_path  = candidate_dir / f"cause_groups_{new_tag}.json"
    meta_path = candidate_dir / f"metadata_{new_tag}.json"
    grp_path.write_text(json.dumps(cause_groups, indent=2))

    # Record which models are new vs carried over from production
    promotion_notes = (
        f"clf15={'new' if clf15_better else 'from_'+prod_tag} | "
        f"clf30={'new' if clf30_better else 'from_'+prod_tag} | "
        f"reg2=new"
    )

    meta_path.write_text(json.dumps({
        "version":             new_tag,
        "promotion_notes":     promotion_notes,
        "paths": {
            "bin15":          str(bin15_path),
            "bin30":          str(bin30_path),
            "reg2_delay_ge5": str(reg_path),
            "cause_groups":   str(grp_path),
        },
        "features":             feature_list,
        "categorical_features": cat_cols,
        "num_medians":          num_medians,
        "thresholds": {
            "bin15_best_valid":    thr15_final,
            "bin30_best_valid":    thr30_final,
            "reg_train_delay_min": 5,
        },
        "metrics": {
            "candidate": candidate_metrics,
            "production": prod,
        },
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "data_date_from": str(data_date_from),
        "data_date_to":   str(backfill_end),
    }, indent=2))

    shutil.move(str(candidate_dir), str(final_dir))

    # ── Record effective metrics in model_versions ────────────────────────
    # Store the EFFECTIVE metrics — new model's metrics if promoted,
    # production metrics if carried over. This ensures the next
    # retrain compares against the right baseline for each model.
    effective_auc15   = auc15   if clf15_better else prod["auc15"]
    effective_prauc15 = prauc15 if clf15_better else prod["prauc15"]
    effective_auc30   = auc30   if clf30_better else prod["auc30"]
    effective_prauc30 = prauc30 if clf30_better else prod["prauc30"]

    # ── Update DB ─────────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("UPDATE model_versions SET is_current=FALSE")
        cur.execute(
            """INSERT INTO model_versions
               (version_tag, trained_at, job_id,
                data_date_from, data_date_to,
                auc15_test, prauc15_test,
                auc30_test, prauc30_test,
                mae_reg_test,
                threshold_bin15, threshold_bin30,
                is_current, artifact_dir, notes)
               VALUES (%s, NOW(), %s, %s, %s,
                       %s, %s, %s, %s, %s,
                       %s, %s, TRUE, %s, %s)""",
            (new_tag, job_id,
             data_date_from, backfill_end,
             effective_auc15, effective_prauc15,
             effective_auc30, effective_prauc30,
             mae_reg,
             thr15_final, thr30_final,
             str(final_dir),
             f"Job {job_id}. {promotion_notes}. "
             f"Beats {prod_tag} on: "
             f"{'clf15 ' if clf15_better else ''}"
             f"{'clf30' if clf30_better else ''}".strip()
             or "neither (partial — reg2 only updated)"),
        )
    conn.commit()

    # ── Hot-reload ────────────────────────────────────────────────────────
    try:
        from model_service import V3FinalModelService
        import api_main
        api_main.model_service = V3FinalModelService(
            models_dir=str(final_dir),
            metadata_filename=f"metadata_{new_tag}.json",
        )
        logger.info("[job=%d] Model hot-reloaded from %s", job_id, final_dir)
    except Exception as exc:
        logger.warning(
            "[job=%d] Hot-reload failed (non-fatal): %s", job_id, exc)

    outcome = "promoted" if (clf15_better and clf30_better) else "partial_promotion"
    _finish_job(conn, job_id, outcome, metrics=candidate_metrics)
    logger.info(
        "[job=%d] %s as %s | %s",
        job_id,
        "PROMOTED" if outcome == "promoted" else "PARTIALLY PROMOTED",
        new_tag,
        promotion_notes,
    )

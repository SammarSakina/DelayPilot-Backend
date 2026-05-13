"""
prepare_ansperf.py
─────────────────────────────────────────────────────────────
Standalone script to clean and merge ANSPerformance CSV files
for all available years and save the two parquet files that
the retraining pipeline reads during feature engineering.

Mirrors the exact logic of notebook 03_ansperf_ingest_clean.

Usage:
    python prepare_ansperf.py --data-dir "C:/path/to/csv/folder"

The script reads all airport_traffic_YYYY.csv and
apt_dly_YYYY.csv.bz2 files it finds in --data-dir,
combines them, filters to EDDM, and writes two parquet
files to the pipeline's data/ directory:
    data/traffic_munich_daily.parquet
    data/atfm_delay_munich_daily.parquet
─────────────────────────────────────────────────────────────
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MUC = "EDDM"


def load_traffic_file(path: Path) -> pd.DataFrame:
    """
    Load one airport_traffic_YYYY.csv file.
    Handles two date formats:
      - 2024 format: DD/MM/YYYY  (dayfirst=True)
      - 2025+ format: YYYY-MM-DD (ISO)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["source_file"] = path.name

    # Detect date format from first non-null value
    sample = df["FLT_DATE"].dropna().iloc[0] if len(df) > 0 else ""
    if "/" in str(sample):
        # DD/MM/YYYY format (2024 file)
        df["date_dt"] = pd.to_datetime(
            df["FLT_DATE"], dayfirst=True, errors="coerce"
        )
        logger.info("%s: parsed dates as DD/MM/YYYY", path.name)
    else:
        # ISO format (2025, 2026 files)
        df["date_dt"] = pd.to_datetime(df["FLT_DATE"], errors="coerce")
        logger.info("%s: parsed dates as ISO", path.name)

    nat_count = df["date_dt"].isna().sum()
    if nat_count > 0:
        logger.warning("%s: %d NaT dates after parsing", path.name, nat_count)

    logger.info("%s: %d rows loaded", path.name, len(df))
    return df


def load_atfm_file(path: Path) -> pd.DataFrame:
    """Load one apt_dly_YYYY.csv.bz2 file."""
    df = pd.read_csv(path, compression="bz2", encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["source_file"] = path.name
    logger.info("%s: %d rows loaded", path.name, len(df))
    return df


def build_traffic_daily(data_dir: Path) -> pd.DataFrame:
    """
    Find all airport_traffic_YYYY.csv files, load, combine,
    filter to EDDM, and build the clean daily traffic table.
    Mirrors notebook Cell 5 exactly.
    """
    traffic_files = sorted(data_dir.glob("airport_traffic_*.csv"))
    if not traffic_files:
        raise FileNotFoundError(
            f"No airport_traffic_*.csv files found in {data_dir}"
        )
    logger.info(
        "Found traffic files: %s",
        [f.name for f in traffic_files],
    )

    parts = [load_traffic_file(f) for f in traffic_files]
    traffic_all = pd.concat(parts, ignore_index=True)

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

    logger.info(
        "Traffic EDDM: %d rows | date range: %s → %s | unique days: %d",
        len(traffic_daily),
        traffic_daily["date"].min(),
        traffic_daily["date"].max(),
        traffic_daily["date"].nunique(),
    )
    return traffic_daily


def build_atfm_daily(data_dir: Path) -> pd.DataFrame:
    """
    Find all apt_dly_YYYY.csv.bz2 files, load, combine,
    filter to EDDM, and build the clean daily ATFM table.
    Mirrors notebook Cells 7-8 exactly.
    """
    atfm_files = sorted(data_dir.glob("apt_dly_*.csv.bz2"))
    if not atfm_files:
        raise FileNotFoundError(
            f"No apt_dly_*.csv.bz2 files found in {data_dir}"
        )
    logger.info(
        "Found ATFM files: %s",
        [f.name for f in atfm_files],
    )

    parts = [load_atfm_file(f) for f in atfm_files]
    atfm_all = pd.concat(parts, ignore_index=True)
    logger.info("ATFM combined: %d rows", len(atfm_all))

    # Parse date — format is ISO with Z: 2024-01-01T00:00:00Z
    atfm_all["date"] = pd.to_datetime(
        atfm_all["FLT_DATE"], errors="coerce", utc=True
    ).dt.date

    atfm_all["APT_ICAO"] = (
        atfm_all["APT_ICAO"].astype(str).str.upper().str.strip()
    )
    atfm_muc = atfm_all[
        atfm_all["APT_ICAO"] == MUC
    ].copy()

    atfm_daily = atfm_muc[[
        "date", "APT_ICAO",
        "FLT_ARR_1",
        "DLY_APT_ARR_1",
        "FLT_ARR_1_DLY",
        "FLT_ARR_1_DLY_15",
        "ATFM_VERSION",
    ]].copy()

    atfm_daily = atfm_daily.rename(columns={
        "APT_ICAO":          "airport",
        "FLT_ARR_1":         "arrivals_cnt",
        "DLY_APT_ARR_1":     "atfm_arr_delay_min_total",
        "FLT_ARR_1_DLY":     "arrivals_delayed_cnt",
        "FLT_ARR_1_DLY_15":  "arrivals_delayed15_cnt",
    })

    # Derived columns — mirrors notebook Cell 8 exactly
    safe_arrivals = atfm_daily["arrivals_cnt"].replace({0: pd.NA})
    atfm_daily["atfm_arr_delay_min_per_arrival"] = (
        atfm_daily["atfm_arr_delay_min_total"] / safe_arrivals
    )
    atfm_daily["arrivals_delayed_rate"] = (
        atfm_daily["arrivals_delayed_cnt"] / safe_arrivals
    )
    atfm_daily["arrivals_delayed15_rate"] = (
        atfm_daily["arrivals_delayed15_cnt"] / safe_arrivals
    )

    logger.info(
        "ATFM EDDM: %d rows | date range: %s → %s | unique days: %d",
        len(atfm_daily),
        atfm_daily["date"].min(),
        atfm_daily["date"].max(),
        atfm_daily["date"].nunique(),
    )
    return atfm_daily


def run(data_dir: Path):
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Traffic ──────────────────────────────────────────────────────────
    traffic_daily = build_traffic_daily(data_dir)
    traffic_out = out_dir / "traffic_munich_daily.parquet"
    traffic_daily.to_parquet(traffic_out, index=False)
    logger.info("Saved: %s", traffic_out)

    # ── ATFM ─────────────────────────────────────────────────────────────
    atfm_daily = build_atfm_daily(data_dir)
    atfm_out = out_dir / "atfm_delay_munich_daily.parquet"
    atfm_daily.to_parquet(atfm_out, index=False)
    logger.info("Saved: %s", atfm_out)

    # ── Sanity check — mirrors notebook Cell 9 ───────────────────────────
    traffic_set = set(traffic_daily["date"])
    atfm_set    = set(atfm_daily["date"])
    logger.info(
        "Sanity check: traffic days=%d | atfm days=%d | "
        "in ATFM not traffic=%d | in traffic not ATFM=%d",
        len(traffic_set), len(atfm_set),
        len(atfm_set - traffic_set),
        len(traffic_set - atfm_set),
    )
    logger.info(
        "Done. Both parquet files written to %s", out_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean ANSPerformance CSVs and write parquet files."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing airport_traffic_YYYY.csv and "
             "apt_dly_YYYY.csv.bz2 files for all years.",
    )
    args = parser.parse_args()
    run(Path(args.data_dir))
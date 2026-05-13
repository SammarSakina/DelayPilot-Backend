import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


def get_engine():
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "delaypilot2026")
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_db = os.getenv("PG_DB", "delaypilot_db")
    url = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    return create_engine(url)


def build_featured_muc_rxn_wx3_fe(max_rows: Optional[int] = None) -> None:
    """
    DB equivalent of the `featured_muc_rxn_wx3_fe.parquet` notebook.

    Starting from:
      - featured_muc_rxn_wx3 (reactionary + weather joined)

    This function adds:
      - congestion features at MUC (muc_*_cnt_pm1h/pm2h)
      - airline congestion features (air_*_cnt_pm1h/pm2h)
      - historical aggregates (route_mean_delay_past, route_rate15_past, air_mean_delay_past)
      - simple weather flags (muc_*_any/strong, other_*_any/strong)
      - optional ANSPerf joins (ans_traffic_*, ans_atfm_*) if the source parquets exist

    Result is written to Postgres table: featured_muc_rxn_wx3_fe.
    """
    engine = get_engine()

    logger.info("Loading featured_muc_rxn_wx3 from Postgres")
    df = pd.read_sql("SELECT * FROM featured_muc_rxn_wx3", engine)
    if df.empty:
        logger.error("featured_muc_rxn_wx3 is empty; run build_featured_muc_rxn_wx3 first.")
        raise RuntimeError("featured_muc_rxn_wx3 is empty; run build_featured_muc_rxn_wx3 first.")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        logger.info("Sampled featured_muc_rxn_wx3 down to %d rows for fe build.", len(df))

    # --- Ensure sched_utc and basic keys are present ---

    df["sched_utc"] = pd.to_datetime(df.get("sched_utc"), utc=True, errors="coerce")
    if df["sched_utc"].isna().all():
        raise RuntimeError("sched_utc is entirely NaT; cannot build time-based features.")

    df["movement"] = df["movement"].astype(str).str.lower().str.strip()
    df["airline_icao"] = df.get("airline_icao", "").astype(str).str.upper().str.strip()
    df["other_airport_icao"] = df.get("other_airport_icao", "").astype(str).str.upper().str.strip()

    # --- A) Congestion features ---

    # Bucket to hour for congestion windows
    df["sched_hour_utc"] = df["sched_utc"].dt.floor("h")

    # Airport-level counts at MUC by movement + rolling windows (3h/5h centered)
    hour_counts = (
        df.groupby(["sched_hour_utc", "movement"])
        .size()
        .rename("cnt")
        .reset_index()
        .sort_values("sched_hour_utc")
    )

    hc = hour_counts.pivot(index="sched_hour_utc", columns="movement", values="cnt").fillna(0)
    for col in ["departure", "arrival"]:
        if col not in hc.columns:
            hc[col] = 0
    hc = hc.sort_index()

    hc["muc_dep_cnt_pm1h"] = hc["departure"].rolling(3, center=True, min_periods=1).sum()
    hc["muc_arr_cnt_pm1h"] = hc["arrival"].rolling(3, center=True, min_periods=1).sum()

    hc["muc_dep_cnt_pm2h"] = hc["departure"].rolling(5, center=True, min_periods=1).sum()
    hc["muc_arr_cnt_pm2h"] = hc["arrival"].rolling(5, center=True, min_periods=1).sum()

    hc = hc.reset_index()[
        ["sched_hour_utc", "muc_dep_cnt_pm1h", "muc_arr_cnt_pm1h", "muc_dep_cnt_pm2h", "muc_arr_cnt_pm2h"]
    ]
    df = df.merge(hc, on="sched_hour_utc", how="left")

    # Airline-level congestion (rolling counts per airline + movement)
    air_h = (
        df.groupby(["airline_icao", "sched_hour_utc", "movement"])
        .size()
        .rename("cnt")
        .reset_index()
        .sort_values(["airline_icao", "sched_hour_utc"])
    )

    out_rows = []
    for mv in ["departure", "arrival"]:
        sub = air_h[air_h["movement"] == mv].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["airline_icao", "sched_hour_utc"])
        sub["cnt_pm1h"] = (
            sub.groupby("airline_icao")["cnt"]
            .rolling(3, center=True, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        sub["cnt_pm2h"] = (
            sub.groupby("airline_icao")["cnt"]
            .rolling(5, center=True, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        sub = sub.rename(
            columns={
                "cnt_pm1h": f"air_{mv}_cnt_pm1h",
                "cnt_pm2h": f"air_{mv}_cnt_pm2h",
            }
        )
        out_rows.append(
            sub[["airline_icao", "sched_hour_utc", f"air_{mv}_cnt_pm1h", f"air_{mv}_cnt_pm2h"]]
        )

    if out_rows:
        air_feat = out_rows[0]
        for extra in out_rows[1:]:
            air_feat = air_feat.merge(
                extra,
                on=["airline_icao", "sched_hour_utc"],
                how="outer",
            )
        df = df.merge(air_feat, on=["airline_icao", "sched_hour_utc"], how="left")

    # Fix airline congestion NaNs -> 0
    for c in [
        "air_departure_cnt_pm1h",
        "air_departure_cnt_pm2h",
        "air_arrival_cnt_pm1h",
        "air_arrival_cnt_pm2h",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # --- B) Historical aggregates (past-only) ---

    df = df.sort_values("sched_utc").reset_index(drop=True)

    route_key = ["movement", "airline_icao", "other_airport_icao"]

    # Use lower minimums (10 route flights, 20 airline flights) to increase coverage
    df["route_mean_delay_past"] = (
        df.groupby(route_key)["y_delay_min"]
        .apply(lambda s: s.shift(1).expanding(min_periods=10).mean())
        .reset_index(level=route_key, drop=True)
    )

    df["route_rate15_past"] = (
        df.groupby(route_key)["y_bin15"]
        .apply(lambda s: s.shift(1).expanding(min_periods=10).mean())
        .reset_index(level=route_key, drop=True)
    )

    df["air_mean_delay_past"] = (
        df.groupby(["movement", "airline_icao"])["y_delay_min"]
        .apply(lambda s: s.shift(1).expanding(min_periods=20).mean())
        .reset_index(level=["movement", "airline_icao"], drop=True)
    )

    # Fill remaining NaNs with global baselines
    global_mean_delay = pd.to_numeric(df["y_delay_min"], errors="coerce").mean()
    global_rate15 = pd.to_numeric(df["y_bin15"], errors="coerce").mean()

    df["route_mean_delay_past"] = df["route_mean_delay_past"].fillna(global_mean_delay)
    df["route_rate15_past"] = df["route_rate15_past"].fillna(global_rate15)
    df["air_mean_delay_past"] = df["air_mean_delay_past"].fillna(global_mean_delay)

    # --- C) Weather flags ---

    # Simple event flags at Munich
    df["muc_wind_strong"] = (df.get("wx_muc_wind_speed_10m", 0) >= 25).astype("int8")
    df["muc_gust_strong"] = (df.get("wx_muc_wind_gusts_10m", 0) >= 40).astype("int8")
    df["muc_precip_any"] = (df.get("wx_muc_precipitation", 0) > 0).astype("int8")
    df["muc_snow_any"] = (df.get("wx_muc_snowfall", 0) > 0).astype("int8")

    # Other-airport flags (NaNs mean other-airport weather isn't available -> treat as 0)
    df["other_wind_strong"] = (
        (df.get("wx_other_wind_speed_10m", np.nan) >= 25).fillna(0).astype("int8")
    )
    df["other_gust_strong"] = (
        (df.get("wx_other_wind_gusts_10m", np.nan) >= 40).fillna(0).astype("int8")
    )
    df["other_precip_any"] = (
        (df.get("wx_other_precipitation", np.nan) > 0).fillna(0).astype("int8")
    )
    df["other_snow_any"] = (
        (df.get("wx_other_snowfall", np.nan) > 0).fillna(0).astype("int8")
    )

    # --- D) Optional ANSPerf joins (if parquet files exist on disk) ---

    data_dir = Path(__file__).parent / "data"
    traffic_path = data_dir / "traffic_munich_daily.parquet"
    atfm_path = data_dir / "atfm_delay_munich_daily.parquet"


    if traffic_path.exists() and atfm_path.exists():
        traffic = pd.read_parquet(traffic_path)
        atfm = pd.read_parquet(atfm_path)

        df["ref_date"] = pd.to_datetime(df.get("ref_ts_utc"), utc=True, errors="coerce").dt.date

        traffic["date"] = pd.to_datetime(traffic["date"], errors="coerce").dt.date
        atfm["date"] = pd.to_datetime(atfm["date"], errors="coerce").dt.date

        traffic = traffic[
            traffic["airport"].astype(str).str.upper().eq("EDDM")
        ].copy()
        atfm = atfm[
            atfm["airport"].astype(str).str.upper().eq("EDDM")
        ].copy()

        traffic_feat = traffic.drop(columns=["airport"], errors="ignore").add_prefix("ans_traffic_")
        traffic_feat = traffic_feat.rename(columns={"ans_traffic_date": "ref_date"})

        atfm_feat = atfm.drop(columns=["airport", "ATFM_VERSION"], errors="ignore").add_prefix(
            "ans_atfm_"
        )
        atfm_feat = atfm_feat.rename(columns={"ans_atfm_date": "ref_date"})

        before_rows = len(df)
        df = df.merge(traffic_feat, on="ref_date", how="left")
        df = df.merge(atfm_feat, on="ref_date", how="left")
        logger.info(
            "Joined ANSPerf daily features: rows kept=%d (expected=%d)",
            len(df),
            before_rows,
        )
    else:
        logger.info(
            "ANSPerf parquet files not found on disk; skipping ans_traffic_/ans_atfm_ feature joins."
        )

    # --- E) Final small tweaks ---

    # ref_year (for knowing which years have ANS coverage)
    df["ref_year"] = pd.to_datetime(df.get("ref_ts_utc"), utc=True, errors="coerce").dt.year

    # Clip reactionary delays to reduce outlier impact
    if "prev_delay_min_safe" in df.columns:
        df["prev_delay_min_safe"] = pd.to_numeric(
            df["prev_delay_min_safe"], errors="coerce"
        ).clip(lower=-180, upper=600)

    # Convert sched_hour_utc from datetime-floor (used for merging above) to
    # integer 0–23 so cause-scoring can use it as a time-of-day signal.
    if "sched_utc" in df.columns:
        df["sched_hour_utc"] = pd.to_datetime(df["sched_utc"], utc=True).dt.hour
    else:
        df["sched_hour_utc"] = -1

    # --- Persist to Postgres ---

    df.to_sql(
        "featured_muc_rxn_wx3_fe",
        engine,
        if_exists="replace",
        index=False,
    )
    logger.info(
        "Wrote featured_muc_rxn_wx3_fe with %d rows and %d columns.",
        len(df),
        len(df.columns),
    )


if __name__ == "__main__":
    build_featured_muc_rxn_wx3_fe()


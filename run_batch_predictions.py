"""
run_batch_predictions.py
────────────────────────────────────────────────────────────────────────────
Runs the three CatBoost models (bin15, bin30, reg2) across every row in
featured_muc_rxn_wx3_fe and writes results into a new table:

    flight_predictions  (PostgreSQL)

This script is called:
  - Once inside run_pipeline.py after step 6/6 (clean_training_features)
  - On every background refresh cycle (Step 5a, before Flight Status API)

TABLE: flight_predictions
  number_raw       VARCHAR   — FK to featured_muc_rxn_wx3_fe
  sched_utc        TIMESTAMPTZ
  p_delay_15       FLOAT     — probability of delay ≥ 15 min
  p_delay_30       FLOAT     — probability of delay ≥ 30 min
  pred_delay_15    INTEGER   — binary: 1 if p15 >= thr15
  pred_delay_30    INTEGER   — binary: 1 if p30 >= thr30
  minutes_pred     FLOAT     — raw regressor output (minutes)
  minutes_ui       FLOAT     — UI-adjusted minutes (0 / ≥5 / ≥30 guardrails)
  ml_cause         VARCHAR   — heuristic cause label from model outputs
  predicted_at     TIMESTAMPTZ — when this prediction was computed

The table is fully replaced on each run (DROP + CREATE + INSERT).
This is safe because:
  - featured_muc_rxn_wx3_fe itself is replaced on each FIDS refresh
  - Predictions are only valid for the current FIDS window anyway
  - No historical prediction archive is needed for the FYP UI

CAUSE DERIVATION (ml_cause)
────────────────────────────
The model does not output a cause label directly. We derive it from the
prediction outputs and feature values using the same heuristic as the
previous UI-side cause logic, but now it runs server-side with access to
all feature columns:

  Priority 1 — Weather signal from pipeline features:
    wx_muc_weather_code > 50  OR  wx_muc_precipitation > 0
    → "Weather"

  Priority 2 — High delay magnitude from model:
    minutes_ui >= 30  (model says major delay)
    → "Reactionary"  (knock-on from previous leg)

  Priority 3 — Moderate delay:
    minutes_ui >= 5
    → "Congestion"

  No delay predicted (minutes_ui == 0):
    → None (not shown in UI)
────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text

from config import get_connection_string
from model_service import V3FinalModelService

logger = logging.getLogger(__name__)


CAUSE_LABEL_MAP = {
    "Weather (MUC)":        "Weather",
    "En-Route Weather":     "Weather",
    "Reactionary":          "Reactionary",
    "ATC / Congestion":     "Congestion",
    "Airline / Turnaround": "Airline",
    "Historical Patterns":  "Historical",
}


def get_engine():
    return create_engine(get_connection_string())


# ── Cause categories (IATA-aligned, derived from available pipeline features) ──
#
# The CatBoost models predict DELAY PROBABILITY AND MAGNITUDE only.
# They do not output a cause label. Cause is derived from the feature values
# that were engineered into featured_muc_rxn_wx3_fe using domain-based rules.
#
# Five categories, in priority order (a flight can satisfy multiple):
#
#   1. Weather (MUC)     — precipitation, snow, strong wind/gust at Munich
#   2. En-Route Weather  — bad weather at origin/destination airport
#   3. Reactionary       — previous leg of same aircraft was late (prev_late*_safe)
#   4. ATC / Congestion  — high airport traffic count in ±1h/2h window
#   5. Airline / Turnaround — airline-level congestion + chronic route delay history
#
# Cause is only set when the model actually predicts a meaningful delay
# (pred_delay_15=True OR minutes_ui >= 5). A model predicting 2 min with
# pred_delay_15=False has no delay to attribute a cause to.
#
# cause_scores{} are also stored so the drawer can show contribution bars
# based on real feature signal strengths, not fabricated percentages.

def _derive_cause_scores(row: dict) -> dict[str, float]:
    """
    Score each cause category from 0.0–1.0 based on pipeline feature signals.
    Scores are proportional to signal strength — higher = stronger evidence.
    These are normalised and returned as contribution percentages in the API.
    """
    def _f(key, default=0.0):
        v = row.get(key)
        return float(v) if v is not None else default

    scores = {}

    # ── 1. Weather at MUC ────────────────────────────────────────────────────
    wx_score = 0.0
    wx_score += min(_f("wx_muc_precipitation") / 5.0,  1.0) * 0.35   # rain intensity
    wx_score += min(_f("wx_muc_snowfall")      / 2.0,  1.0) * 0.40   # snow (higher weight)
    wx_score += _f("muc_precip_any")                           * 0.10
    wx_score += _f("muc_snow_any")                             * 0.10
    wx_score += min(_f("wx_muc_wind_speed_10m") / 40.0, 1.0)  * 0.25  # strong wind
    wx_score += _f("muc_wind_strong")                          * 0.15
    wx_score += _f("muc_gust_strong")                          * 0.15
    # Weather code > 50 = moderate/heavy precip (WMO scale)
    wx_code = _f("wx_muc_weather_code")
    if wx_code > 70:   wx_score += 0.40   # heavy snow/rain
    elif wx_code > 50: wx_score += 0.20   # moderate precip
    elif wx_code > 20: wx_score += 0.10   # light precip / fog
    scores["Weather (MUC)"] = min(wx_score, 1.0)

    # ── 2. En-Route / Origin Weather ─────────────────────────────────────────
    enr_score = 0.0
    enr_score += _f("other_precip_any")  * 0.35
    enr_score += _f("other_snow_any")    * 0.40
    enr_score += _f("other_wind_strong") * 0.30
    enr_score += _f("other_gust_strong") * 0.25
    enr_score += min(_f("wx_other_precipitation") / 5.0, 1.0) * 0.20
    scores["En-Route Weather"] = min(enr_score, 1.0)

    # ── 3. Reactionary ───────────────────────────────────────────────────────
    rxn_score = 0.0
    prev_late15 = _f("prev_late15_safe")
    prev_late30 = _f("prev_late30_safe")
    prev_delay  = _f("prev_delay_min_safe")
    y_delay     = _f("y_delay_min")   # FIDS observed delay for this flight

    rxn_score += prev_late30 * 0.35
    rxn_score += prev_late15 * 0.20
    rxn_score += min(prev_delay / 45.0, 1.0) * 0.50

    # Live flights often have no prev-leg data; use FIDS observed delay as proxy.
    # A flight already delayed in FIDS most likely has a late inbound aircraft.
    if prev_delay == 0 and prev_late15 == 0 and y_delay >= 15:
        rxn_score += min(y_delay / 60.0, 1.0) * 0.35

    scores["Reactionary"] = min(rxn_score, 1.0)

    # ── 4. ATC / Airport Congestion ──────────────────────────────────────────
    dep_pm1h   = _f("muc_dep_cnt_pm1h")
    arr_pm1h   = _f("muc_arr_cnt_pm1h")
    dep_pm2h   = _f("muc_dep_cnt_pm2h")
    arr_pm2h   = _f("muc_arr_cnt_pm2h")
    total_pm1h = dep_pm1h + arr_pm1h
    total_pm2h = dep_pm2h + arr_pm2h

    # Subtract routine-traffic baseline before normalising so normal MUC
    # traffic (~60–80 movements per 3h window) doesn't always score ~0.50.
    # Only genuinely busy periods (>100 movements) score above ~0.43.
    above_baseline_pm1h = max(total_pm1h - 50, 0)
    above_baseline_pm2h = max(total_pm2h - 90, 0)
    congestion_score  = min(above_baseline_pm1h / 70.0,  1.0) * 0.55
    congestion_score += min(above_baseline_pm2h / 110.0, 1.0) * 0.35
    scores["ATC / Congestion"] = min(congestion_score, 1.0)

    # ── 5. Airline / Turnaround ───────────────────────────────────────────────
    air_dep    = _f("air_departure_cnt_pm1h")
    air_arr    = _f("air_arrival_cnt_pm1h")
    route_mean = _f("route_mean_delay_past")
    air_mean   = _f("air_mean_delay_past")

    airline_score = 0.0
    airline_score += min((air_dep + air_arr) / 20.0, 1.0) * 0.20
    airline_score += min(route_mean / 20.0, 1.0)          * 0.50
    airline_score += min(air_mean   / 15.0, 1.0)          * 0.30
    scores["Airline / Turnaround"] = min(airline_score, 1.0)

    # ── 6. Historical Patterns (time-of-day + chronic route delay) ───────────
    # Munich has well-known delay patterns by UTC hour:
    #   06–09: morning bank, reactionary from overnight positioning
    #   09–13: midday, generally lowest delays
    #   13–17: afternoon, congestion builds
    #   17–21: evening bank, turnaround delays accumulate
    # Used as a proxy when route/airline historical averages are unavailable
    # (common for live flights without sufficient historical rotation data).
    hist_score = 0.0
    sched_hour = _f("sched_hour_utc", default=-1)   # integer 0–23 UTC

    if route_mean > 2:
        hist_score += min(route_mean / 25.0, 1.0) * 0.60
    if air_mean > 2:
        hist_score += min(air_mean / 20.0, 1.0) * 0.30

    if 6 <= sched_hour < 9:
        hist_score += 0.30   # morning bank
    elif 17 <= sched_hour < 21:
        hist_score += 0.20   # evening bank
    elif 13 <= sched_hour < 17:
        hist_score += 0.10   # afternoon build-up

    scores["Historical Patterns"] = min(hist_score, 1.0)

    return scores


def _derive_ml_cause(row: dict, pred_delay_15: int, minutes_ui: float) -> tuple[str | None, dict]:
    """
    Derive primary cause label and per-cause contribution scores.

    Only assigns a cause when the model predicts an actual delay:
      pred_delay_15 == 1  (model binary classifier above threshold)
      OR minutes_ui >= 5  (model regressor predicts meaningful delay)

    Returns: (primary_cause, cause_scores_pct)
      primary_cause     — str label or None (no delay)
      cause_scores_pct  — dict {cause: pct} normalised to 100, or {}
    """
    is_delayed = (pred_delay_15 == 1) or (minutes_ui >= 5)
    if not is_delayed:
        return None, {}

    scores = _derive_cause_scores(row)

    # Normalise to percentages that sum to 100
    total = sum(scores.values())
    if total <= 0:
        # Fallback: all signals zero → label by magnitude alone
        if minutes_ui >= 30:
            return "Reactionary", {"Reactionary": 100}
        return "Congestion", {"Congestion": 100}

    pct = {k: round(v / total * 100) for k, v in scores.items()}

    # Adjust rounding so it sums to exactly 100
    diff = 100 - sum(pct.values())
    if diff != 0:
        top_key = max(pct, key=lambda k: scores[k])
        pct[top_key] += diff

    # Primary cause = highest scoring category
    primary = max(scores, key=lambda k: scores[k])
    primary_ui = CAUSE_LABEL_MAP.get(primary, primary)

    pct_mapped = {}
    for key, value in pct.items():
        mapped_key = CAUSE_LABEL_MAP.get(key, key)
        pct_mapped[mapped_key] = pct_mapped.get(mapped_key, 0) + value

    # Adjust rounding again after merging duplicate UI labels (for example, weather buckets).
    diff = 100 - sum(pct_mapped.values())
    if diff != 0:
        pct_mapped[primary_ui] = pct_mapped.get(primary_ui, 0) + diff

    return primary_ui, pct_mapped


def run_batch_predictions() -> int:
    """
    Run predictions for all flights in featured_muc_rxn_wx3_fe.
    Returns the number of rows written to flight_predictions.
    """
    engine = get_engine()

    # ── 1. Load model ────────────────────────────────────────────────────────
    logger.info("[batch_pred] Loading models...")
    try:
        svc = V3FinalModelService()
    except Exception as e:
        logger.error("[batch_pred] Could not load models: %s", e)
        raise

    # ── 2. Read all feature rows ─────────────────────────────────────────────
    logger.info("[batch_pred] Reading featured_muc_rxn_wx3_fe...")
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM featured_muc_rxn_wx3_fe", conn)

    if df.empty:
        logger.warning("[batch_pred] featured_muc_rxn_wx3_fe is empty — nothing to predict.")
        return 0

    logger.info("[batch_pred] Running predictions for %d flights...", len(df))

    # ── 3. Run predictions row-by-row ────────────────────────────────────────
    records = []
    errors  = 0

    for _, row in df.iterrows():
        try:
            result     = svc.predict_one(row)
            minutes_ui = result["minutes_ui"]
            ml_cause, cause_scores = _derive_ml_cause(row.to_dict(), result["pred_delay_15"], minutes_ui)

            records.append({
                "number_raw":    row["number_raw"],
                "sched_utc":     row["sched_utc"],
                "p_delay_15":    round(result["p_delay_15"],   4),
                "p_delay_30":    round(result["p_delay_30"],   4),
                "pred_delay_15": result["pred_delay_15"],
                "pred_delay_30": result["pred_delay_30"],
                "minutes_pred":  round(result["minutes_pred"], 2),
                "minutes_ui":    round(minutes_ui,             1),
                "ml_cause":      ml_cause,
                "cause_scores":  json.dumps(cause_scores),   # JSON string for Postgres TEXT
                "predicted_at":  datetime.now(timezone.utc),
            })
        except Exception as e:
            errors += 1
            logger.warning(
                "[batch_pred] Prediction failed for %s @ %s: %s",
                row.get("number_raw"), row.get("sched_utc"), e,
            )

    if not records:
        logger.error("[batch_pred] All predictions failed — not writing table.")
        return 0

    # ── 4. Write to flight_predictions ───────────────────────────────────────
    out_df = pd.DataFrame(records)

    with engine.connect() as conn:
        # Drop and recreate — predictions are only valid for the current FIDS window
        conn.execute(text("DROP TABLE IF EXISTS flight_predictions"))
        conn.commit()

    out_df.to_sql(
        "flight_predictions",
        engine,
        if_exists="replace",
        index=False,
    )

    logger.info(
        "[batch_pred] Wrote %d predictions to flight_predictions (%d errors).",
        len(records), errors,
    )
    return len(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_batch_predictions()

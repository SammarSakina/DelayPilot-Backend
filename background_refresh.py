"""
background_refresh.py
────────────────────────────────────────────────────────────────────────────
Runs a single combined refresh cycle on one interval:

  STEP 1 — Weather (Open-Meteo)
  STEP 2 — FIDS   (Aerodatabox live window) → flights_raw
  STEP 3 — Feature build pass 1            → featured_muc_rxn_wx3
  STEP 4 — Feature build pass 2            → featured_muc_rxn_wx3_fe
  STEP 5 — Batch ML Predictions            → flight_predictions   ← ADDED
  STEP 6 — Flight Status API               → flight_status_live
  STEP 7 — Delay Analytics Snapshot        → flight_delay_snapshots (upsert)

WHY STEP 5 EXISTS:
  The original cycle rebuilt feature tables (Steps 3–4) but never re-ran
  the CatBoost models afterwards. This meant flight_predictions stayed at
  startup values while featured_muc_rxn_wx3_fe was refreshed every 30 min —
  the API's /flights JOIN was serving stale ML scores against fresh features.

  Step 5 calls run_batch_predictions() immediately after the feature tables
  are ready and before Flight Status ingestion, so:
    - /predict/from-db always reads current-cycle scores
    - flight_delay_snapshots (Step 7) resolve against fresh ML minutes
    - confirmed_delay_min (Step 6) enriches the same flight rows the model
      just scored, maintaining a consistent per-cycle timestamp

Interval (configurable):
  REFRESH_INTERVAL_MINUTES   default: 30

Usage — called from api_main.py on_startup:
    from background_refresh import start_background_refresh
    @app.on_event("startup")
    def startup():
        start_background_refresh()
────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ── Single shared interval ────────────────────────────────────────────────────
_interval_cache_checked_at = 0.0
_interval_cache_value_sec: Optional[int] = None
_settings_engine = None
_pipeline_log = []
_log_lock = None

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "last_ran":    None,
    "running":     False,
    "last_error":  None,
    "fids_last_ran":        None,
    "predictions_last_ran": None,
    "status_last_ran":      None,
    "scheduler_enabled":    True,
    "next_run_at":          None,
}


def get_refresh_state() -> dict:
    """Return a snapshot of refresh state — safe to call from any thread."""
    return dict(_state)


def set_scheduler_enabled(enabled: bool) -> None:
    """
    Enable or disable the background scheduler without
    restarting the server. When disabled, the scheduler
    loop keeps running but skips all refresh cycles.
    Existing data in the DB and UI remain untouched.
    """
    _state["scheduler_enabled"] = bool(enabled)
    logger.info(
        "[background] Scheduler %s by admin.",
        "ENABLED" if enabled else "DISABLED",
    )
    append_pipeline_log({
        "event": f"Scheduler {'enabled' if enabled else 'disabled'} by admin",
        "status": "success" if enabled else "skipped",
        "timestamp": _now_iso(),
    })


def get_scheduler_status() -> dict:
    """
    Return scheduler enabled state and countdown to next run.
    Used by GET /scheduler/status endpoint.
    """
    next_run_at = _state.get("next_run_at")
    seconds_remaining = None
    if next_run_at is not None and _state["scheduler_enabled"]:
        delta = (next_run_at - datetime.now(timezone.utc)).total_seconds()
        seconds_remaining = max(0, int(delta))
    return {
        "enabled":           _state["scheduler_enabled"],
        "next_run_at":       next_run_at.isoformat() if next_run_at else None,
        "seconds_remaining": seconds_remaining,
        "interval_minutes":  get_refresh_interval_sec() // 60,
        "last_ran":          _state["last_ran"].isoformat()
                             if _state["last_ran"] else None,
        "running":           _state["running"],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_pipeline_log(entry: dict) -> None:
    global _log_lock
    if _log_lock is None:
        _log_lock = threading.Lock()
    with _log_lock:
        _pipeline_log.append(dict(entry))
        while len(_pipeline_log) > 50:
            _pipeline_log.pop(0)


def get_pipeline_log() -> list:
    global _log_lock
    if _log_lock is None:
        _log_lock = threading.Lock()
    with _log_lock:
        logs = [dict(entry) for entry in _pipeline_log]
    return sorted(logs, key=lambda e: e.get("timestamp", ""), reverse=True)


def _log_step_success(step_number: int, step_name: str) -> None:
    append_pipeline_log({
        "event": f"Step {step_number} completed: {step_name}",
        "status": "success",
        "timestamp": _now_iso(),
    })


def _log_step_failure(step_number: int, step_name: str, exc: Exception) -> None:
    append_pipeline_log({
        "event": f"Step {step_number} failed: {step_name}",
        "status": "error",
        "timestamp": _now_iso(),
        "error": str(exc)[:200],
    })


def _get_settings_engine():
    global _settings_engine
    if _settings_engine is None:
        pg_user     = os.getenv("PG_USER",     "postgres")
        pg_password = os.getenv("PG_PASSWORD", "delaypilot2026")
        pg_host     = os.getenv("PG_HOST",     "localhost")
        pg_port     = os.getenv("PG_PORT",     "5432")
        pg_db       = os.getenv("PG_DB",       "delaypilot_db")
        url = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        _settings_engine = create_engine(url)
    return _settings_engine


def _valid_refresh_minutes(value: Any) -> Optional[int]:
    try:
        minutes = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if 5 <= minutes <= 120:
        return minutes
    return None


def _fallback_refresh_interval_sec() -> int:
    minutes = _valid_refresh_minutes(os.getenv("REFRESH_INTERVAL_MINUTES", "30"))
    return (minutes or 30) * 60


def _read_refresh_interval_minutes_from_db() -> Optional[int]:
    queries = [
        ("SELECT value FROM system_settings WHERE key = :key LIMIT 1", "value"),
        ("SELECT setting_value FROM system_settings WHERE setting_key = :key LIMIT 1", "setting_value"),
        ("SELECT value FROM system_settings WHERE name = :key LIMIT 1", "value"),
        ("SELECT refresh_interval_minutes FROM system_settings LIMIT 1", "refresh_interval_minutes"),
    ]
    engine = _get_settings_engine()
    for sql, column_name in queries:
        try:
            with engine.connect() as conn:
                row = conn.execute(text(sql), {"key": "refresh_interval_minutes"}).mappings().first()
        except Exception as exc:
            logger.debug("[background] Refresh interval settings query failed: %s", exc)
            continue
        if row:
            minutes = _valid_refresh_minutes(row.get(column_name))
            if minutes is not None:
                return minutes
    return None


def get_refresh_interval_sec() -> int:
    global _interval_cache_checked_at, _interval_cache_value_sec
    now = time.monotonic()
    if (
        _interval_cache_value_sec is not None
        and now - _interval_cache_checked_at < 60
    ):
        return _interval_cache_value_sec
    minutes = _read_refresh_interval_minutes_from_db()
    _interval_cache_value_sec = minutes * 60 if minutes is not None else _fallback_refresh_interval_sec()
    _interval_cache_checked_at = now
    return _interval_cache_value_sec


# ── Combined refresh cycle ────────────────────────────────────────────────────

def _run_full_refresh():
    """
    Execute one complete data refresh cycle in order:

      1. Weather  — Open-Meteo hourly data
      2. FIDS     — Aerodatabox live window → flights_raw
      3. Features — build featured_muc_rxn_wx3
      4. Features — build featured_muc_rxn_wx3_fe  (what /flights reads)
      5. Predictions — run_batch_predictions()     ← NEW
                       reads featured_muc_rxn_wx3_fe, scores all rows,
                       writes flight_predictions (DROP + INSERT each cycle)
      6. Status   — Aerodatabox Flight Status API  → flight_status_live
      7. Snapshot — Resolve tier-priority delay + upsert analytics table

    Step 5 runs AFTER feature rebuild (so features are current) and BEFORE
    flight status ingestion (so the snapshot in Step 7 can combine fresh
    ML scores with confirmed delays from the same cycle).
    """
    if _state["running"]:
        logger.info("[background] Refresh already in progress — skipping this tick.")
        append_pipeline_log({
            "event": "Cycle skipped (already running)",
            "status": "skipped",
            "timestamp": _now_iso(),
        })
        return

    _state["running"]    = True
    _state["last_error"] = None
    cycle_start = datetime.now(timezone.utc)
    append_pipeline_log({
        "event": "Refresh cycle started",
        "status": "running",
        "timestamp": cycle_start.isoformat(),
    })
    logger.info("[background] ══ Full refresh cycle starting ══")

    # ── Step 1: Weather ───────────────────────────────────────────────────────
    try:
        from ingest_weather_live import update_weather_live
        update_weather_live()
        logger.info("[background] Step 1/7 ✓ Weather updated.")
        _log_step_success(1, "Weather")
    except Exception as e:
        logger.warning("[background] Step 1/7 ✗ Weather update failed: %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(1, "Weather", e)

    # ── Step 2: FIDS ─────────────────────────────────────────────────────────
    try:
        from ingest_flights_live import ingest_live_muc_window
        ingest_live_muc_window()
        _state["fids_last_ran"] = datetime.now(timezone.utc)
        logger.info("[background] Step 2/7 ✓ FIDS ingested.")
        _log_step_success(2, "FIDS")
    except Exception as e:
        logger.warning("[background] Step 2/7 ✗ FIDS ingest failed: %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(2, "FIDS", e)

    # ── Step 3: Feature build pass 1 ─────────────────────────────────────────
    try:
        from build_featured_muc_rxn_wx3 import build_featured_muc_rxn_wx3
        build_featured_muc_rxn_wx3()
        logger.info("[background] Step 3/7 ✓ featured_muc_rxn_wx3 rebuilt.")
        _log_step_success(3, "Feature pass 1")
    except Exception as e:
        logger.warning("[background] Step 3/7 ✗ Feature pass 1 failed: %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(3, "Feature pass 1", e)

    # ── Step 4: Feature build pass 2 ─────────────────────────────────────────
    try:
        from build_featured_muc_rxn_wx3_fe import build_featured_muc_rxn_wx3_fe
        build_featured_muc_rxn_wx3_fe()
        logger.info("[background] Step 4/7 ✓ featured_muc_rxn_wx3_fe rebuilt.")
        _log_step_success(4, "Feature pass 2")
    except Exception as e:
        logger.warning("[background] Step 4/7 ✗ Feature pass 2 failed: %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(4, "Feature pass 2", e)

    # ── Step 5: Batch ML Predictions ─────────────────────────────────────────
    # This is the step that was missing in the original cycle.
    # Without this, flight_predictions holds startup-time scores even after
    # the feature tables have been refreshed.  With it, every /flights response
    # and every /predict/from-db call reads scores computed from current-cycle
    # features.  The step is non-fatal: a model-load failure should not prevent
    # the dashboard from receiving updated flight status in Step 6.
    try:
        from run_batch_predictions import run_batch_predictions
        n_written = run_batch_predictions()
        _state["predictions_last_ran"] = datetime.now(timezone.utc)
        logger.info("[background] Step 5/7 ✓ Batch predictions written (%d rows).", n_written)
        _log_step_success(5, f"Batch predictions ({n_written} rows)")
    except Exception as e:
        logger.warning("[background] Step 5/7 ✗ Batch predictions failed (non-fatal): %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(5, "Batch predictions", e)

    # ── Step 6: Flight Status API ─────────────────────────────────────────────
    try:
        from update_flight_status import update_flight_status
        update_flight_status()
        _state["status_last_ran"] = datetime.now(timezone.utc)
        logger.info("[background] Step 6/7 ✓ Flight status updated.")
        _log_step_success(6, "Flight status")
    except Exception as e:
        logger.warning("[background] Step 6/7 ✗ Flight status update failed: %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(6, "Flight status", e)

    # ── Step 7: Delay Analytics Snapshot ─────────────────────────────────────
    try:
        from snapshot_delay_analytics import snapshot_delay_analytics
        snapshot_delay_analytics()
        logger.info("[background] Step 7/7 ✓ Delay analytics snapshot written.")
        _log_step_success(7, "Delay snapshot")
    except Exception as e:
        logger.warning("[background] Step 7/7 ✗ Delay analytics snapshot failed: %s", e)
        _state["last_error"] = str(e)
        _log_step_failure(7, "Delay snapshot", e)

    elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
    if _state["last_error"] is None:
        append_pipeline_log({
            "event": "Refresh cycle completed",
            "status": "success",
            "timestamp": _now_iso(),
            "duration_seconds": round(elapsed, 1),
        })
    _state["last_ran"] = datetime.now(timezone.utc)
    _state["running"]  = False
    logger.info("[background] ══ Full refresh cycle complete (%.1fs) ══", elapsed)


# ── Scheduler loop ────────────────────────────────────────────────────────────

def _scheduler_loop():
    """
    Daemon loop — defers the first cycle by refresh_interval_sec because
    start_delaypilot.py already runs run_pipeline.py (which includes batch
    predictions) synchronously before the API starts.  The first background
    cycle would be a redundant double-refresh at boot.
    """
    initial_interval_sec = get_refresh_interval_sec()
    # Set next_run_at immediately so the UI timer shows
    # on first poll rather than waiting for the first sleep
    _state["next_run_at"] = (
        datetime.now(timezone.utc) +
        timedelta(seconds=initial_interval_sec)
    )
    logger.info(
        "[background] Scheduler started — first refresh in %d min, then every %d min.",
        initial_interval_sec // 60,
        initial_interval_sec // 60,
    )

    last_ran = time.monotonic()   # defers first run by refresh_interval_sec

    while True:
        interval_sec = get_refresh_interval_sec()
        next_run = datetime.now(timezone.utc) + timedelta(seconds=interval_sec)
        _state["next_run_at"] = next_run

        # Sleep in small increments so we can respond to
        # disable/enable without waiting the full interval.
        elapsed = 0
        while elapsed < interval_sec:
            time.sleep(5)
            elapsed += 5
            # Re-read next_run_at in case interval changed
            if not _state["scheduler_enabled"]:
                # Keep updating next_run_at so timer resets
                # when re-enabled
                _state["next_run_at"] = (
                    datetime.now(timezone.utc) +
                    timedelta(
                        seconds=get_refresh_interval_sec()
                    )
                )

        if not _state["scheduler_enabled"]:
            logger.info(
                "[background] Scheduler disabled — skipping refresh cycle."
            )
            append_pipeline_log({
                "event": "Refresh cycle skipped (scheduler disabled by admin)",
                "status": "skipped",
                "timestamp": _now_iso(),
            })
            continue

        _run_full_refresh()


# ── Public entry point ────────────────────────────────────────────────────────

_scheduler_thread: threading.Thread = None


def start_background_refresh():
    """
    Start the background scheduler thread.
    Safe to call multiple times — only one thread will ever run.
    """
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        logger.info("[background] Scheduler already running — skipping start.")
        return

    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        name="delaypilot-scheduler",
        daemon=True,
    )
    _scheduler_thread.start()
    logger.info("[background] Scheduler thread started.")
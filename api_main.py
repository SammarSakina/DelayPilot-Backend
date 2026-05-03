from datetime import datetime
from typing import Any, Dict, Optional

import logging
import re
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from typing import Optional
from fastapi import Query

from model_service import V3FinalModelService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_engine():
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "delaypilot2026")
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_db = os.getenv("PG_DB", "delaypilot_db")
    url = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    return create_engine(url)


app = FastAPI(title="DelayPilot Prediction API", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Allow frontend (e.g. React/Vue) running on a different origin to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for FYP/demo; in production, restrict to known origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton model service (loaded once at startup)
model_service = V3FinalModelService(models_dir=str(os.path.join(os.path.dirname(__file__), "models")))
engine = get_engine()
# ── Background refresh scheduler ─────────────────────────────
from background_refresh import start_background_refresh, get_refresh_state


def _health_payload() -> Dict[str, Any]:
    state = get_refresh_state()
    return {
        "status":          "ok",
        "last_ran":        state["last_ran"].isoformat()        if state["last_ran"]        else None,
        "fids_last_ran":   state["fids_last_ran"].isoformat()   if state["fids_last_ran"]   else None,
        "status_last_ran": state["status_last_ran"].isoformat() if state["status_last_ran"] else None,
        "running":         state["running"],
    }


@app.on_event("startup")
def on_startup():
    # Ensure flight_status_live exists before any request tries to JOIN it
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS flight_status_live (
                number_raw          VARCHAR,
                flight_date         DATE,
                op_status           VARCHAR,
                etd_utc             TIMESTAMPTZ,
                atd_utc             TIMESTAMPTZ,
                eta_utc             TIMESTAMPTZ,
                ata_utc             TIMESTAMPTZ,
                confirmed_delay_min FLOAT,
                fetched_at          TIMESTAMPTZ,
                PRIMARY KEY (number_raw, flight_date)
            )
        """))
        conn.commit()
    import os as _sched_os
    from dotenv import load_dotenv as _sched_ldenv
    _sched_ldenv()
    if _sched_os.getenv("ENABLE_SCHEDULER", "true").lower() != "false":
        start_background_refresh()
    else:
        logger.info(
            "[background] Scheduler disabled via ENABLE_SCHEDULER=false"
        )

class DbPredictionRequest(BaseModel):
    """
    Request for prediction using a flight row stored in Postgres.

    For now we look up by:
      - number_raw (flight number string as in Aerodatabox, e.g. 'LH1234')
      - sched_utc (ISO-8601 string, same as stored in featured_muc_rxn_wx3_fe)

    In your real app, you may switch to an explicit flight_id key.
    """

    number_raw: str
    # Accept as string so we can flexibly parse formats from UI/DB
    sched_utc: str


def _parse_sched_utc(value: str) -> datetime:
    """
    Parse sched_utc from a few common string formats into a timezone-aware datetime.

    Accepts:
      - '2026-03-05T02:35:00+00:00'
      - '2026-03-05 02:35:00+00:00'
      - '2026-03-05 02:35:00+05' (will be normalized to +05:00)
    """
    s = (value or "").strip()
    if not s:
        raise ValueError("sched_utc is empty")

    # Allow space instead of 'T'
    if " " in s and "T" not in s:
        # only replace the first space between date and time
        s = s.replace(" ", "T", 1)

    # Normalize timezone like +05 or -03 to +05:00 / -03:00
    m = re.match(r"(.+)([+-]\d{2})$", s)
    if m:
        base, tz_hour = m.groups()
        s = f"{base}{tz_hour}:00"

    return datetime.fromisoformat(s)


# AFTER
class PredictionResponse(BaseModel):
    p_delay_15: float
    p_delay_30: float
    pred_delay_15: int
    pred_delay_30: int
    minutes_pred: float
    minutes_ui: float
    thresholds: Dict[str, Any]
    # Status fields from flight_status_live (None when table not yet populated)
    op_status:           Optional[str]   = None
    confirmed_delay_min: Optional[float] = None
    etd_utc:             Optional[str]   = None
    atd_utc:             Optional[str]   = None
    eta_utc:             Optional[str]   = None
    ata_utc:             Optional[str]   = None


class FlightListItem(BaseModel):
    """Lightweight flight representation for the frontend flights table."""

    number_raw: str
    airline_icao: str | None = None
    movement: str
    route: str
    sched_utc: datetime
    status: str
    predicted_delay_min: float


class LoginRequest(BaseModel):
    username: str
    password: str
    role: str


class LoginUser(BaseModel):
    username: str
    name: str
    role: str


class LoginResponse(BaseModel):
    success: bool
    token: str
    user: LoginUser
    message: Optional[str] = None


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    """
    Simple demo login endpoint for the DelayPilot frontend.

    For FYP purposes, we:
      - Accept any username/password that meet the minimum length checks
      - Echo back a fake JWT token and basic user info

    In a real system, this would be replaced with proper authentication.
    """
    logger.info("Login attempt for username=%s role=%s", req.username, req.role)

    if len(req.username) < 4 or len(req.password) < 8:
        logger.warning("Login failed due to validation for username=%s", req.username)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Fake token and user; good enough for UI routing and role-based nav
    token = f"demo-token-{req.username}"
    user = LoginUser(username=req.username, name=req.username, role=req.role)

    logger.info("Login succeeded for username=%s role=%s", req.username, req.role)
    return LoginResponse(success=True, token=token, user=user)


@app.get("/flights/current", response_model=list[FlightListItem])
def list_current_flights(limit: int = 200) -> list[FlightListItem]:
    """
    Return a list of recent flights from featured_muc_rxn_wx3_fe for the frontend.

    For simplicity we:
      - sort by sched_utc descending (most recent first)
      - compute a simple status based on y_delay_min
    """
    logger.info("Listing current flights (limit=%s)", limit)

    with engine.connect() as conn:
        query = text(
            """
            SELECT
                number_raw,
                COALESCE(airline_icao, airline_iata, '') AS airline_icao,
                movement,
                other_airport_icao,
                sched_utc,
                y_delay_min
            FROM featured_muc_rxn_wx3_fe
            WHERE number_raw IS NOT NULL
              AND sched_utc IS NOT NULL
            ORDER BY sched_utc DESC
            LIMIT :limit
            """
        )
        rows = conn.execute(query, {"limit": limit}).mappings().all()

    flights: list[FlightListItem] = []
    for r in rows:
        delay_min = float(r.get("y_delay_min") or 0.0)
        movement = (r.get("movement") or "").lower()
        other_airport = (r.get("other_airport_icao") or "").upper()

        # Simple route string for UI
        if movement == "departure":
            route = f"MUC → {other_airport}" if other_airport else "MUC → ?"
        else:
            route = f"{other_airport} → MUC" if other_airport else "? → MUC"

        # Simple status buckets for UI
        if delay_min <= 0:
            status = "On-Time"
        elif delay_min <= 15:
            status = "Minor Delay"
        elif delay_min <= 60:
            status = "Major Delay"
        else:
            status = "Major Delay"

        flights.append(
            FlightListItem(
                number_raw=r["number_raw"],
                airline_icao=r.get("airline_icao") or None,
                movement=movement,
                route=route,
                sched_utc=r["sched_utc"],
                status=status,
                predicted_delay_min=delay_min,
            )
        )

    return flights


@app.post("/predict/from-db", response_model=PredictionResponse)
def predict_from_db(req: DbPredictionRequest):
    """
    Predict using a row from `featured_muc_rxn_wx3_fe` in Postgres.

    This is mainly for integration testing with the historical pipeline.
    Later you can add a separate endpoint that builds features from live APIs.
    """
    logger.info(
        "Received prediction request number_raw=%s sched_utc=%s",
        req.number_raw,
        req.sched_utc,
    )

    try:
        # Parse sched_utc string into a proper datetime that matches DB storage
        try:
            sched_dt = _parse_sched_utc(req.sched_utc)
        except Exception as exc:
            logger.warning("Invalid sched_utc format '%s': %s", req.sched_utc, exc)
            raise HTTPException(status_code=400, detail=f"Invalid sched_utc format: {exc}")

        with engine.connect() as conn:
            query = text(
                """
                SELECT *
                FROM featured_muc_rxn_wx3_fe
                WHERE number_raw = :number_raw
                  AND sched_utc = :sched_utc
                LIMIT 1
                """
            )
            row = conn.execute(
                query,
                {
                    "number_raw": req.number_raw,
                    "sched_utc": sched_dt,
                },
            ).mappings().first()

        if row is None:
            logger.info(
                "No matching flight found in featured_muc_rxn_wx3_fe for number_raw=%s sched_utc=%s",
                req.number_raw,
                sched_dt.isoformat(),
            )
            raise HTTPException(
                status_code=404,
                detail="No matching flight found in featured_muc_rxn_wx3_fe for the given number_raw and sched_utc.",
            )

        # Convert row mapping to DataFrame with a single row
        row_dict = dict(row)
        df = pd.DataFrame([row_dict])

        logger.info(
            "Running model prediction for number_raw=%s sched_utc=%s",
            req.number_raw,
            sched_dt.isoformat(),
        )
        # AFTER
        result = model_service.predict_one(df)
        logger.info(
            "Prediction succeeded for number_raw=%s sched_utc=%s",
            req.number_raw,
            sched_dt.isoformat(),
        )

        # Patch C — enrich with confirmed delay, falling back to flights_raw
        try:
            status_q = text("""
                SELECT
                    COALESCE(
                        s.op_status,
                        CASE r.status
                            WHEN 'Landed'     THEN 'Landed'
                            WHEN 'EnRoute'    THEN 'EnRoute'
                            WHEN 'En Route'   THEN 'EnRoute'
                            WHEN 'Departed'   THEN 'EnRoute'
                            WHEN 'Cancelled'  THEN 'Cancelled'
                            WHEN 'Diverted'   THEN 'Diverted'
                            WHEN 'GateClosed' THEN 'EnRoute'
                            WHEN 'Boarding'   THEN 'Scheduled'
                            WHEN 'CheckIn'    THEN 'Scheduled'
                            WHEN 'Delayed'    THEN 'Scheduled'
                            WHEN 'Expected'   THEN 'Scheduled'
                            ELSE NULL
                        END,
                        'Scheduled'
                    ) AS op_status,
                    COALESCE(
                        s.confirmed_delay_min,
                        CASE
                            WHEN r.movement = 'departure' AND r.dep_best_utc IS NOT NULL
                                THEN ROUND(
                                    EXTRACT(EPOCH FROM (r.dep_best_utc - r.dep_sched_utc)) / 60.0
                                , 1)
                            WHEN r.movement = 'arrival' AND r.arr_best_utc IS NOT NULL
                                THEN ROUND(
                                    EXTRACT(EPOCH FROM (r.arr_best_utc - r.arr_sched_utc)) / 60.0
                                , 1)
                            ELSE NULL
                        END
                    ) AS confirmed_delay_min,
                    COALESCE(s.etd_utc, r.dep_rev_utc)   AS etd_utc,
                    COALESCE(s.atd_utc, r.dep_runway_utc) AS atd_utc,
                    COALESCE(s.eta_utc, r.arr_rev_utc)   AS eta_utc,
                    COALESCE(s.ata_utc, r.arr_runway_utc) AS ata_utc
                FROM flights_raw r
                LEFT JOIN flight_status_live s
                       ON s.number_raw  = r.number_raw
                      AND s.flight_date = DATE(
                              COALESCE(r.dep_sched_utc, r.arr_sched_utc)
                              AT TIME ZONE 'UTC'
                          )
                WHERE r.number_raw = :number_raw
                  AND COALESCE(r.dep_sched_utc, r.arr_sched_utc) IS NOT NULL
                ORDER BY
                    ABS(EXTRACT(EPOCH FROM (
                        COALESCE(r.dep_sched_utc, r.arr_sched_utc) - :sched_utc_ts
                    ))) ASC
                LIMIT 1
            """)
            with engine.connect() as conn:
                srow = conn.execute(status_q, {
                    "number_raw":   req.number_raw,
                    "sched_utc_ts": sched_dt,
                }).fetchone()

            if srow:
                result["op_status"]           = srow[0]
                result["confirmed_delay_min"] = float(srow[1]) if srow[1] is not None else None
                result["etd_utc"]             = str(srow[2])   if srow[2] is not None else None
                result["atd_utc"]             = str(srow[3])   if srow[3] is not None else None
                result["eta_utc"]             = str(srow[4])   if srow[4] is not None else None
                result["ata_utc"]             = str(srow[5])   if srow[5] is not None else None
            else:
                result["op_status"]           = "Scheduled"
                result["confirmed_delay_min"] = None
                result["etd_utc"]             = None
                result["atd_utc"]             = None
                result["eta_utc"]             = None
                result["ata_utc"]             = None
        except Exception as status_exc:
            # Non-fatal — flight_status_live may not exist yet
            logger.warning("Could not fetch flight status for prediction: %s", status_exc)
            result["op_status"] = None
            result["confirmed_delay_min"] = None

        return PredictionResponse(**result)

    except HTTPException:
        # Already a well-formed API error; just propagate.
        raise
    except Exception as exc:
        logger.exception(
            "Prediction failed due to unexpected error for number_raw=%s sched_utc=%s",
            req.number_raw,
            req.sched_utc,
        )
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


# Replace existing /health with this
@app.get("/health")
def health() -> Dict[str, Any]:
    return _health_payload()


@app.get("/pipeline-logs")
def pipeline_logs() -> Dict[str, Any]:
    from background_refresh import get_pipeline_log, get_refresh_state
    logs = get_pipeline_log()
    state = get_refresh_state()
    return {
        "logs": logs,
        "health": {
            "last_ran": str(state["last_ran"]) if state["last_ran"] else None,
            "fids_last_ran": str(state["fids_last_ran"]) if state["fids_last_ran"] else None,
            "status_last_ran": str(state["status_last_ran"]) if state["status_last_ran"] else None,
            "running": state["running"],
            "last_error": state.get("last_error"),
        },
    }

@app.get("/flights/propagation")
def get_flight_propagation(
    number_raw: str = Query(...),
    sched_utc: str = Query(...),
):
    """
    Return connected flights that share the same aircraft tail as the source flight
    and are scheduled later the same day, with propagated delay estimates.
    """
    def safe_val(v):
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        try:
            return v.isoformat()
        except AttributeError:
            pass
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    try:
        try:
            sched_dt = _parse_sched_utc(sched_utc)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid sched_utc: {exc}")

        # ── 1. Fetch source flight ────────────────────────────────────────────
        src_q = text("""
            SELECT
                f.aircraft_modeS,
                f.y_delay_min,
                f.movement,
                f.other_airport_icao,
                f.airline_iata,
                COALESCE(s.op_status, 'Scheduled') AS op_status,
                s.confirmed_delay_min,
                p.minutes_ui  AS ml_minutes_ui,
                p.p_delay_15
            FROM featured_muc_rxn_wx3_fe f
            LEFT JOIN flight_status_live s
                   ON s.number_raw  = f.number_raw
                  AND s.flight_date = DATE(f.sched_utc AT TIME ZONE 'UTC')
            LEFT JOIN flight_predictions p
                   ON p.number_raw = f.number_raw
                  AND p.sched_utc  = f.sched_utc
            WHERE f.number_raw = :number_raw
              AND f.sched_utc  = :sched_utc
            LIMIT 1
        """)
        with engine.connect() as conn:
            src = conn.execute(src_q, {"number_raw": number_raw, "sched_utc": sched_dt}).fetchone()

        if src is None:
            raise HTTPException(status_code=404, detail="Source flight not found.")

        aircraft_modeS = src[0] if src else None
        y_delay_min = src[1]
        confirmed_delay_min = src[6]
        ml_minutes_ui = src[7]
        p_delay_15 = src[8]

        if not aircraft_modeS:
            return {"propagation": [], "source": {}}

        # ── 2. Resolve source delay (3-tier priority) ─────────────────────────
        if confirmed_delay_min is not None:
            source_delay = float(confirmed_delay_min)
        elif y_delay_min is not None and (float(y_delay_min) >= 15 or float(y_delay_min) < 0):
            source_delay = float(y_delay_min)
        elif ml_minutes_ui is not None and float(ml_minutes_ui) >= 5:
            source_delay = float(ml_minutes_ui)
        else:
            source_delay = None

        # ── 3. Propagation estimate ───────────────────────────────────────────
        if source_delay is not None and p_delay_15 is not None:
            propagation_estimate = round(float(source_delay) * float(p_delay_15) * 0.6, 1)
        else:
            propagation_estimate = None

        # ── 4. Query connected flights ────────────────────────────────────────
        conn_q = text("""
            SELECT
                f.number_raw,
                f.airline_iata,
                f.movement,
                f.other_airport_icao,
                f.sched_utc,
                f.y_delay_min,
                COALESCE(s.op_status, 'Scheduled') AS op_status,
                s.confirmed_delay_min,
                p.minutes_ui AS ml_minutes_ui
            FROM featured_muc_rxn_wx3_fe f
            LEFT JOIN flight_status_live s
                   ON s.number_raw  = f.number_raw
                  AND s.flight_date = DATE(f.sched_utc AT TIME ZONE 'UTC')
            LEFT JOIN flight_predictions p
                   ON p.number_raw = f.number_raw
                  AND p.sched_utc  = f.sched_utc
            WHERE f.aircraft_modeS = :modeS
              AND f.sched_utc > :sched_utc
              AND DATE(f.sched_utc AT TIME ZONE 'UTC') = DATE(:sched_utc AT TIME ZONE 'UTC')
            ORDER BY f.sched_utc ASC
            LIMIT 2
        """)
        with engine.connect() as conn:
            conn_rows = conn.execute(conn_q, {"modeS": aircraft_modeS, "sched_utc": sched_dt}).fetchall()

        # ── 5. Build response ─────────────────────────────────────────────────
        def _resolve_delay(conf, fids, ml, prop_est):
            if conf is not None:
                return float(conf), "confirmed"
            if fids is not None and (float(fids) >= 15 or float(fids) < 0):
                return float(fids), "fids"
            if ml is not None and float(ml) >= 5:
                return float(ml), "model"
            if prop_est is not None:
                return prop_est, "model_propagation"
            return None, "none"

        def _delay_status(op_status, resolved_delay):
            if op_status in ("Landed", "Cancelled", "Diverted"):
                return op_status
            if resolved_delay is None:
                return "On Time"
            if resolved_delay >= 30:
                return "Major Delay"
            if resolved_delay >= 5:
                return "Minor Delay"
            if resolved_delay < 0:
                return "Early"
            return "On Time"

        source_payload = {
            "aircraft_modeS": safe_val(src[0]),
            "y_delay_min": safe_val(src[1]),
            "movement": safe_val(src[2]),
            "other_airport_icao": safe_val(src[3]),
            "airline_iata": safe_val(src[4]),
            "op_status": safe_val(src[5]),
            "confirmed_delay_min": safe_val(src[6]),
            "ml_minutes_ui": safe_val(src[7]),
            "p_delay_15": safe_val(src[8]),
            "resolved_source_delay_min": safe_val(source_delay),
            "propagation_estimate": safe_val(propagation_estimate),
        }

        connected = []
        for r in conn_rows:
            c_number_raw = r[0]
            c_airline_iata = r[1]
            c_movement = (r[2] or "").lower()
            c_other_airport = (r[3] or "").upper()
            c_sched_utc = r[4]
            c_y_delay_min = r[5]
            c_op_status = r[6]
            c_confirmed_delay = r[7]
            c_ml_minutes_ui = r[8]

            resolved_delay, delay_source = _resolve_delay(
                c_confirmed_delay, c_y_delay_min, c_ml_minutes_ui, propagation_estimate
            )

            status = _delay_status(c_op_status, resolved_delay)

            if c_movement == "departure":
                route = f"MUC → {c_other_airport}" if c_other_airport else "MUC → ?"
            else:
                route = f"{c_other_airport} → MUC" if c_other_airport else "? → MUC"

            connected.append({
                "number_raw": safe_val(c_number_raw),
                "airline_iata": safe_val(c_airline_iata),
                "route": safe_val(route),
                "sched_utc": safe_val(c_sched_utc),
                "resolved_delay_min": safe_val(resolved_delay),
                "delay_source": safe_val(delay_source),
                "delay_status": safe_val(status),
                "is_propagated": safe_val(delay_source == "model_propagation"),
            })

        return {"propagation": connected, "source": source_payload}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Propagation error for number_raw=%s sched_utc=%s: %s",
            number_raw,
            sched_utc,
            exc,
        )
        return {"propagation": [], "source": {}, "error": str(exc)}


# NEW ENDPOINTS ADDED
"""
api_main.py  — PATCH for /flights endpoint
───────────────────────────────────────────
Replace the existing @app.get("/flights") function in api_main.py
with the version below.

What changes:
  • JOINs flights_raw to get dep_best_utc / arr_best_utc
    (the "actual / estimated" times Aerodatabox provides).
  • Exposes them as actual_utc in the JSON response.
  • Keeps all existing fields intact.
"""

# ── REPLACE the existing /flights route with this ──────────────────

@app.get("/flights")
def get_flights(date: str = None):
    """
    Return all flights in featured_muc_rxn_wx3_fe,
    enriched with actual/revised times and op_status from flight_status_live.

    op_status values: Scheduled | EnRoute | Landed | Cancelled | Diverted | Unknown
    If no status record exists, op_status defaults to 'Scheduled'.
    """
    try:
        def _iso(value):
            return value.isoformat() if value is not None else None

        date_filter = ""
        params = {}
        if date:
            date_filter = "WHERE DATE(f.sched_utc AT TIME ZONE 'UTC') = :date"
            params["date"] = date

        q = text(f"""
            SELECT DISTINCT ON (f.number_raw, f.sched_utc)
                f.number_raw,
                f.sched_utc,
                f.movement,
                f.airline_iata,
                f.other_airport_icao,
                f.y_delay_min,
                f.y_bin15,
                f.y_bin30,
                -- Weather fields for cause derivation in UI
                f.wx_muc_weather_code,
                f.wx_muc_precipitation,
                -- Actual time from flights_raw (best estimate from FIDS)
                CASE
                    WHEN f.movement = 'departure' THEN r.dep_best_utc
                    ELSE r.arr_best_utc
                END AS actual_utc,
                -- Status: flight_status_live (authoritative) → FIDS r.status → 'Scheduled'
                COALESCE(
                    s.op_status,
                    CASE r.status
                        WHEN 'Landed'     THEN 'Landed'
                        WHEN 'EnRoute'    THEN 'EnRoute'
                        WHEN 'En Route'   THEN 'EnRoute'
                        WHEN 'Departed'   THEN 'EnRoute'
                        WHEN 'Cancelled'  THEN 'Cancelled'
                        WHEN 'Diverted'   THEN 'Diverted'
                        WHEN 'GateClosed' THEN 'EnRoute'
                        WHEN 'Boarding'   THEN 'Scheduled'
                        WHEN 'CheckIn'    THEN 'Scheduled'
                        WHEN 'Delayed'    THEN 'Scheduled'
                        WHEN 'Expected'   THEN 'Scheduled'
                        WHEN 'Scheduled'  THEN 'Scheduled'
                        ELSE NULL
                    END,
                    'Scheduled'
                ) AS op_status,
                s.etd_utc,
                s.atd_utc,
                s.eta_utc,
                s.ata_utc,
                -- confirmed_delay_min: flight_status_live (Tier 1) → FIDS best times (Tier 2)
                COALESCE(
                    s.confirmed_delay_min,
                    CASE
                        WHEN f.movement = 'departure' AND r.dep_best_utc IS NOT NULL
                            THEN ROUND(
                                EXTRACT(EPOCH FROM (r.dep_best_utc - r.dep_sched_utc)) / 60.0
                            , 1)
                        WHEN f.movement = 'arrival' AND r.arr_best_utc IS NOT NULL
                            THEN ROUND(
                                EXTRACT(EPOCH FROM (r.arr_best_utc - r.arr_sched_utc)) / 60.0
                            , 1)
                        ELSE NULL
                    END
                ) AS confirmed_delay_min,
                -- ML batch predictions (from flight_predictions, populated by run_batch_predictions.py)
                p.minutes_ui            AS ml_minutes_ui,
                p.p_delay_15            AS ml_p_delay_15,
                p.p_delay_30            AS ml_p_delay_30,
                p.pred_delay_15         AS ml_pred_delay_15,
                p.pred_delay_30         AS ml_pred_delay_30,
                p.ml_cause              AS ml_cause,
                p.cause_scores          AS cause_scores
            FROM featured_muc_rxn_wx3_fe f
            LEFT JOIN flights_raw r
                   ON r.number_raw = f.number_raw
                  AND (
                      (f.movement = 'departure' AND r.dep_sched_utc = f.sched_utc)
                   OR (f.movement = 'arrival'   AND r.arr_sched_utc = f.sched_utc)
                      )
            LEFT JOIN flight_status_live s
                   ON s.number_raw  = f.number_raw
                  AND s.flight_date = DATE(f.sched_utc AT TIME ZONE 'UTC')
            LEFT JOIN flight_predictions p
                   ON p.number_raw = f.number_raw
                  AND p.sched_utc  = f.sched_utc
            {date_filter}
            ORDER BY f.number_raw, f.sched_utc
        """)

        with engine.connect() as conn:
            rows = conn.execute(q, params).fetchall()

        return [
            {
                "number_raw":            r[0],
                "sched_utc":             _iso(r[1]),
                "movement":              r[2],
                "airline_iata":          r[3],
                "other_airport_icao":    r[4],
                "destination":           r[4],
                "delay_min":             float(r[5]) if r[5] is not None else None,
                "is_delayed_15":         bool(r[6])  if r[6] is not None else None,
                "is_delayed_30":         bool(r[7])  if r[7] is not None else None,
                # Weather for cause derivation
                "wx_muc_weather_code":   float(r[8]) if r[8] is not None else None,
                "wx_muc_precipitation":  float(r[9]) if r[9] is not None else None,
                "actual_utc":            _iso(r[10]),
                # Status fields
                "op_status":             r[11],
                "etd_utc":               _iso(r[12]),
                "atd_utc":               _iso(r[13]),
                "eta_utc":               _iso(r[14]),
                "ata_utc":               _iso(r[15]),
                "confirmed_delay_min":   float(r[16]) if r[16] is not None else None,
                # ML batch predictions
                "ml_minutes_ui":         float(r[17]) if r[17] is not None else None,
                "ml_p_delay_15":         float(r[18]) if r[18] is not None else None,
                "ml_p_delay_30":         float(r[19]) if r[19] is not None else None,
                "ml_pred_delay_15":      int(r[20])   if r[20] is not None else None,
                "ml_pred_delay_30":      int(r[21])   if r[21] is not None else None,
                "ml_cause":              str(r[22])   if r[22] is not None else None,
                "cause_scores":          str(r[23])   if r[23] is not None else None,
            }
            for r in rows
        ]

    except Exception as e:
        logger.error("/flights error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

"""
api_main.py  — ADD /simulate endpoint
──────────────────────────────────────────────────────────────────
How it works:
  1. Loads the real feature row for the flight from featured_muc_rxn_wx3_fe
  2. Overlays ONLY the parameters the user changed (overrides dict)
  3. Re-runs predict_one() with the modified row
  4. Returns baseline prediction + simulated prediction side-by-side
  5. Also returns a reactionary impact estimate for connected flights

The user only touches the parameters they care about.
Everything else stays true to the real pipeline data.
"""

class SimulateRequest(BaseModel):
    number_raw: str
    sched_utc:  str

    # ── Weather overrides (MUC) ─────────────────────────────────
    # These map directly to weather_hourly columns joined at ref_hour_utc
    wind_speed_10m:     Optional[float] = None   # km/h
    wind_gusts_10m:     Optional[float] = None   # km/h
    precipitation:      Optional[float] = None   # mm
    snowfall:           Optional[float] = None   # mm
    visibility:         Optional[float] = None   # metres (Open-Meteo unit)
    weather_code:       Optional[int]   = None   # WMO code

    # ── Reactionary override ────────────────────────────────────
    # prev_delay_min_safe: delay of previous rotation of the same aircraft
    prev_delay_min_safe: Optional[float] = None  # minutes

    # ── Congestion overrides ────────────────────────────────────
    # These are rolling counts of flights at MUC around sched_utc
    # muc_arr_1h / muc_dep_1h — arrivals/departures in ±1h window
    # These inputs override MUC rolling congestion features around sched_utc.
    # The underlying pm1h columns are 3-hour rolling sums, not single-hour counts.
    # Maps to muc_arr_cnt_pm1h, a 3-hour rolling sum (±1h window).
    muc_arr_1h: Optional[float] = None
    # Maps to muc_dep_cnt_pm1h, a 3-hour rolling sum (±1h window).
    muc_dep_1h: Optional[float] = None


@app.post("/simulate")
def simulate_flight(req: SimulateRequest):
    """
    Load a real flight's feature row, apply user overrides,
    run the model, and return baseline vs simulated predictions.
    """
    try:
        # ── 1. Parse sched_utc ──────────────────────────────────
        from dateutil import parser as dtparser
        import pytz, pandas as pd, numpy as np

        try:
            sched_dt = dtparser.parse(req.sched_utc)
            if sched_dt.tzinfo is None:
                sched_dt = pytz.utc.localize(sched_dt)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid sched_utc: {e}")

        # ── 2. Load feature row from DB ─────────────────────────
        q = text("""
            SELECT * FROM featured_muc_rxn_wx3_fe
            WHERE number_raw = :number_raw
              AND sched_utc  = :sched_utc
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = conn.execute(q, {"number_raw": req.number_raw, "sched_utc": sched_dt}).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Flight {req.number_raw} @ {req.sched_utc} not found.")

        # ── 3. Landed guard (uses flight_status_live for authoritative status)
        status_q = text("""
            SELECT op_status
            FROM flight_status_live
            WHERE number_raw  = :number_raw
              AND flight_date = DATE(:sched_utc AT TIME ZONE 'UTC')
            LIMIT 1
        """)
        with engine.connect() as conn:
            status_row = conn.execute(status_q, {
                "number_raw": req.number_raw,
                "sched_utc":  sched_dt,
            }).fetchone()

        # Block simulation only for definitively completed or cancelled flights.
        # Scheduled, EnRoute, Unknown, and missing → allow simulation.
        if status_row and status_row[0] in ("Landed", "Cancelled", "Diverted"):
            raise HTTPException(
                status_code=409,
                detail="FLIGHT_LANDED",   # frontend checks this exact string
            )

        # ── 4. Build baseline DataFrame ─────────────────────────
        columns = list(row._mapping.keys())
        values  = list(row._mapping.values())
        df_base = pd.DataFrame([values], columns=columns)

        # ── 5. Baseline prediction ──────────────────────────────
        baseline = model_service.predict_one(df_base)

        # ── 6. Build override DataFrame ─────────────────────────
        df_sim = df_base.copy()

        # Weather column name mapping (pipeline names vary slightly)
        # Open-Meteo uses these column names after normalization
        weather_map = {
            "wind_speed_10m":  ["wx_muc_wind_speed_10m",  "wind_speed_10m"],
            "wind_gusts_10m":  ["wx_muc_wind_gusts_10m",  "wind_gusts_10m"],
            "precipitation":   ["wx_muc_precipitation",   "precipitation"],
            "snowfall":        ["wx_muc_snowfall",         "snowfall"],
            "visibility":      ["wx_muc_visibility",       "visibility"],
            "weather_code":    ["wx_muc_weather_code",     "weather_code"],
        }

        def apply_override(df, candidates, value):
            """Try each candidate column name; apply to the first found.
            Also recomputes all derived binary flag columns that depend on the
            overridden value — using the EXACT same thresholds as
            build_featured_muc_rxn_wx3_fe.py.
            """
            for col in candidates:
                if col in df.columns:
                    df[col] = value
                    # ── Wind flags (thresholds from build_featured_muc_rxn_wx3_fe.py) ──
                    if "wind_speed" in col:
                        if "muc_wind_strong" in df.columns:
                            df["muc_wind_strong"] = int(value >= 25)   # ≥25 km/h
                    if "wind_gusts" in col:
                        if "muc_gust_strong" in df.columns:
                            df["muc_gust_strong"] = int(value >= 40)   # ≥40 km/h
                    # ── Precip/snow flags ───────────────────────────────────────────
                    if "precipitation" in col:
                        if "muc_precip_any" in df.columns:
                            df["muc_precip_any"] = int(value > 0)
                    if "snowfall" in col:
                        if "muc_snow_any" in df.columns:
                            df["muc_snow_any"] = int(value > 0)
                        if "muc_precip_any" in df.columns and value > 0:
                            df["muc_precip_any"] = 1   # snow counts as precipitation
                    return True
            return False

        override_applied = {}

        for field, candidates in weather_map.items():
            val = getattr(req, field)
            if val is not None:
                if apply_override(df_sim, candidates, val):
                    override_applied[field] = val

        # Reactionary override
        if req.prev_delay_min_safe is not None:
            val = req.prev_delay_min_safe
            for col in ["prev_delay_min_safe", "reactionary_delay_min"]:
                if col in df_sim.columns:
                    df_sim[col] = val
            if "prev_late15_safe" in df_sim.columns:
                df_sim["prev_late15_safe"] = int(val >= 15)
            if "prev_late30_safe" in df_sim.columns:
                df_sim["prev_late30_safe"] = int(val >= 30)
            override_applied["prev_delay_min_safe"] = val

        # Congestion overrides
        # Actual column names from build_featured_muc_rxn_wx3_fe.py:
        # pm1h columns are 3-hour rolling sums and pm2h columns are 5-hour rolling sums.
        # Airline-level congestion features are also updated proportionally when airport-level values change.
        #   muc_arr_cnt_pm1h / muc_dep_cnt_pm1h  (±1h rolling count)
        #   muc_arr_cnt_pm2h / muc_dep_cnt_pm2h  (±2h rolling count, scale proportionally)
        if req.muc_arr_1h is not None:
            for col in ["muc_arr_cnt_pm1h", "muc_arr_1h", "arr_1h"]:
                if col in df_sim.columns:
                    df_sim[col] = req.muc_arr_1h
                    override_applied["muc_arr_1h"] = req.muc_arr_1h
                    # Also scale ±2h count proportionally if present
                    pm2h = col.replace("pm1h", "pm2h")
                    if pm2h in df_sim.columns:
                        df_sim[pm2h] = req.muc_arr_1h * (5.0 / 3.0)
                    break
            airline_arr_estimate = req.muc_arr_1h * 0.15
            for col in ["air_arrival_cnt_pm1h"]:
                if col in df_sim.columns:
                    df_sim[col] = airline_arr_estimate
            for col in ["air_arrival_cnt_pm2h"]:
                if col in df_sim.columns:
                    df_sim[col] = airline_arr_estimate * (5.0 / 3.0)

        if req.muc_dep_1h is not None:
            for col in ["muc_dep_cnt_pm1h", "muc_dep_1h", "dep_1h"]:
                if col in df_sim.columns:
                    df_sim[col] = req.muc_dep_1h
                    override_applied["muc_dep_1h"] = req.muc_dep_1h
                    pm2h = col.replace("pm1h", "pm2h")
                    if pm2h in df_sim.columns:
                        df_sim[pm2h] = req.muc_dep_1h * (5.0 / 3.0)
                    break
            airline_dep_estimate = req.muc_dep_1h * 0.15
            for col in ["air_departure_cnt_pm1h"]:
                if col in df_sim.columns:
                    df_sim[col] = airline_dep_estimate
            for col in ["air_departure_cnt_pm2h"]:
                if col in df_sim.columns:
                    df_sim[col] = airline_dep_estimate * (5.0 / 3.0)

        # ── 7. Simulated prediction ─────────────────────────────
        simulated = model_service.predict_one(df_sim)

        # ── 8. Reactionary impact estimate ─────────────────────
        # Estimate how many same-airline flights scheduled within +3h
        # could be affected by this flight's simulated delay.
        # We query flights_raw for flights of same airline after sched_utc.
        airline_iata = str(df_base.get("airline_iata", pd.Series([""])).iloc[0])
        reactionary_q = text("""
            SELECT number_raw, dep_sched_utc AS sched
            FROM flights_raw
            WHERE airline_iata = :airline
              AND dep_sched_utc > :sched_utc
              AND dep_sched_utc < :sched_utc + INTERVAL '3 hours'
            ORDER BY dep_sched_utc
            LIMIT 5
        """)
        with engine.connect() as conn:
            rrows = conn.execute(reactionary_q, {
                "airline": airline_iata,
                "sched_utc": sched_dt
            }).fetchall()

        sim_delay = simulated.get("minutes_ui", 0) or 0
        reactionary_impacts = []
        for i, rr in enumerate(rrows):
            # Reactionary delay decays: each downstream flight gets ~60% of upstream
            decay   = 0.6 ** (i + 1)
            r_delay = round(sim_delay * decay)
            if r_delay >= 5:
                reactionary_impacts.append({
                    "flight":    rr[0],
                    "sched":     str(rr[1]),
                    "added_min": r_delay,
                    "severity":  "high" if r_delay >= 25 else "moderate" if r_delay >= 12 else "minor",
                })

        # ── 9. Return ───────────────────────────────────────────
        return {
            "flight":     req.number_raw,
            "sched_utc":  req.sched_utc,
            "overrides":  override_applied,
            "baseline":   baseline,
            "simulated":  simulated,
            "delta": {
                "p_delay_15":  round(simulated["p_delay_15"]  - baseline["p_delay_15"],  4),
                "p_delay_30":  round(simulated["p_delay_30"]  - baseline["p_delay_30"],  4),
                "minutes_ui":  round((simulated.get("minutes_ui") or 0) - (baseline.get("minutes_ui") or 0), 1),
            },
            "reactionary_impact": reactionary_impacts,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/simulate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/weather/current")
def current_weather():
    """
    Returns latest weather for all 3 airports: EDDM (MUC), EDDF (FRA), EGLL (LHR).
    MUC full detail; FRA and LHR summary. Extra MUC fields: wind_gusts, snowfall, cloud_cover, humidity.
    """
    try:
        q = text("""
            SELECT
                airport_icao, hour_utc, temperature_2m, wind_speed_10m, wind_gusts_10m,
                precipitation, snowfall, visibility, weather_code, cloud_cover, relative_humidity_2m
            FROM weather_hourly
            WHERE airport_icao IN ('EDDM', 'EDDF', 'EGLL')
              AND (airport_icao, hour_utc) IN (
                  SELECT airport_icao, MAX(hour_utc)
                  FROM weather_hourly
                  WHERE airport_icao IN ('EDDM', 'EDDF', 'EGLL')
                  GROUP BY airport_icao
              )
            ORDER BY airport_icao
        """)
        with engine.connect() as conn:
            rows = conn.execute(q).fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No weather data")

        airports = {}
        for row in rows:
            airports[row[0]] = {
                "timestamp": str(row[1]), "temperature": row[2], "wind_speed": row[3],
                "wind_gusts": row[4], "precipitation": row[5], "snowfall": row[6],
                "visibility": row[7], "weather_code": row[8],
                "cloud_cover": row[9], "relative_humidity": row[10],
            }

        muc = airports.get("EDDM", {})
        fra = airports.get("EDDF", {})
        lhr = airports.get("EGLL", {})

        return {
            "timestamp": muc.get("timestamp"), "temperature": muc.get("temperature"),
            "wind_speed": muc.get("wind_speed"), "wind_gusts": muc.get("wind_gusts"),
            "precipitation": muc.get("precipitation"), "snowfall": muc.get("snowfall"),
            "visibility": muc.get("visibility"), "weather_code": muc.get("weather_code"),
            "cloud_cover": muc.get("cloud_cover"), "relative_humidity": muc.get("relative_humidity"),
            "fra": {"temperature": fra.get("temperature"), "wind_speed": fra.get("wind_speed"),
                    "precipitation": fra.get("precipitation"), "weather_code": fra.get("weather_code"),
                    "visibility": fra.get("visibility")},
            "lhr": {"temperature": lhr.get("temperature"), "wind_speed": lhr.get("wind_speed"),
                    "precipitation": lhr.get("precipitation"), "weather_code": lhr.get("weather_code"),
                    "visibility": lhr.get("visibility")},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/weather/current error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch weather")



@app.get("/flights/analytics")
def get_delay_trends(date: str = None):
    """
    Returns aggregated delay trend + cause breakdown from flight_delay_snapshots.

    This table accumulates across refresh cycles (upsert, not replace), so the
    charts improve in accuracy over time as more Status API and ML data comes in.

    Query params:
      date: YYYY-MM-DD  (defaults to today UTC)

    Returns:
      trend  — list of {hour, total, delayed, delay_rate} for each UTC hour
      causes — list of {name, value} for cause breakdown (ML cause preferred,
               falls back to weather-code heuristic)
      meta   — {total_flights, delayed_flights, date, last_updated}
    """
    try:
        from datetime import date as dt_date
        target_date = date or dt_date.today().isoformat()

        with engine.connect() as conn:
            # Check table exists
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'flight_delay_snapshots'
                )
            """)).scalar()

        if not exists:
            return {"trend": [], "causes": [], "meta": {"error": "Snapshot table not yet created. Run one refresh cycle first."}}

        with engine.connect() as conn:
            # ── Hourly trend ─────────────────────────────────────────────────
            trend_rows = conn.execute(text("""
                SELECT
                    EXTRACT(HOUR FROM sched_utc AT TIME ZONE 'Europe/Berlin')::INT AS hour,
                    COUNT(*)                                              AS total,
                    COUNT(*) FILTER (
                        WHERE delay_status IN ('Minor Delay', 'Major Delay')
                    )                                                     AS delayed
                FROM flight_delay_snapshots
                WHERE flight_date = :date
                GROUP BY 1
                ORDER BY 1
            """), {"date": target_date}).fetchall()

            # ── Cause breakdown ───────────────────────────────────────────────
            # Priority: use ml_cause when available, else derive from weather/delay
            cause_rows = conn.execute(text("""
                SELECT
                    COALESCE(
                        NULLIF(ml_cause, ''),
                        CASE
                            WHEN wx_weather_code > 50 OR wx_precipitation > 0 THEN 'Weather'
                            WHEN resolved_delay_min >= 30                      THEN 'Reactionary'
                            WHEN resolved_delay_min >= 5                       THEN 'Congestion'
                            ELSE NULL
                        END
                    ) AS cause,
                    COUNT(*) AS cnt
                FROM flight_delay_snapshots
                WHERE flight_date = :date
                  AND delay_status IN ('Minor Delay', 'Major Delay')
                  AND resolved_delay_min IS NOT NULL
                GROUP BY 1
                ORDER BY 2 DESC
            """), {"date": target_date}).fetchall()

            # ── Meta ──────────────────────────────────────────────────────────
            meta_row = conn.execute(text("""
                SELECT
                    COUNT(*)                                              AS total,
                    COUNT(*) FILTER (
                        WHERE delay_status IN ('Minor Delay', 'Major Delay')
                    )                                                     AS delayed,
                    MAX(last_updated_utc)                                 AS last_updated
                FROM flight_delay_snapshots
                WHERE flight_date = :date
            """), {"date": target_date}).fetchone()

        # Build full 24-hour trend (fill missing hours with 0)
        trend_map = {r[0]: {"total": r[1], "delayed": r[2]} for r in trend_rows}
        trend = []
        for h in range(24):
            bucket = trend_map.get(h, {"total": 0, "delayed": 0})
            total   = int(bucket["total"])
            delayed = int(bucket["delayed"])
            trend.append({
                "hour":        f"{str(h).zfill(2)}:00",
                "total":       total,
                "delayed":     delayed,
                "delay_rate":  round(delayed / total * 100, 1) if total > 0 else 0,
            })

        causes = [
            {"name": row[0] or "Other", "value": int(row[1])}
            for row in cause_rows
            if row[0] is not None
        ]
        if not causes:
            causes = [{"name": "No delay data", "value": 1}]

        return {
            "trend":  trend,
            "causes": causes,
            "meta": {
                "total_flights":   int(meta_row[0]) if meta_row else 0,
                "delayed_flights": int(meta_row[1]) if meta_row else 0,
                "date":            target_date,
                "last_updated":    str(meta_row[2]) if meta_row and meta_row[2] else None,
            },
        }

    except Exception as e:
        logger.error("/flights/analytics error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Retraining endpoints ──────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel

class _RetrainStartRequest(_BaseModel):
    triggered_by: str = "admin"


@app.post("/retrain/start")
def retrain_start(req: _RetrainStartRequest):
    import psycopg2 as _pg
    import os as _os
    from dotenv import load_dotenv as _ldenv
    _ldenv()

    def _make_conn():
        _url = _os.getenv("DATABASE_URL")
        if _url:
            if _url.startswith("postgres://"):
                _url = _url.replace("postgres://", "postgresql://", 1)
            if "sslmode" not in _url:
                _url += ("&" if "?" in _url else "?") + "sslmode=require"
            return _pg.connect(_url)
        return _pg.connect(
            host=_os.getenv("PG_HOST", "localhost"),
            port=int(_os.getenv("PG_PORT", "5432")),
            dbname=_os.getenv("PG_DB", "delaypilot_db"),
            user=_os.getenv("PG_USER", "postgres"),
            password=_os.getenv("PG_PASSWORD", "delaypilot2026"),
        )

    _conn = _make_conn()
    try:
        with _conn.cursor() as _cur:
            _cur.execute(
                "INSERT INTO retrain_jobs (triggered_by, status) "
                "VALUES (%s, 'queued') RETURNING id",
                (req.triggered_by,),
            )
            _job_id = _cur.fetchone()[0]
        _conn.commit()
    finally:
        _conn.close()

    import threading as _th
    from retrain_pipeline import run_retrain
    _t = _th.Thread(target=run_retrain, args=(_job_id,), daemon=True)
    _t.start()

    logger.info("Retraining job %d started by %s", _job_id, req.triggered_by)
    return {"job_id": _job_id, "status": "queued"}


@app.get("/retrain/status/{job_id}")
def retrain_status(job_id: int):
    import psycopg2 as _pg
    import os as _os
    from dotenv import load_dotenv as _ldenv
    _ldenv()

    def _make_conn():
        _url = _os.getenv("DATABASE_URL")
        if _url:
            if _url.startswith("postgres://"):
                _url = _url.replace("postgres://", "postgresql://", 1)
            if "sslmode" not in _url:
                _url += ("&" if "?" in _url else "?") + "sslmode=require"
            return _pg.connect(_url)
        return _pg.connect(
            host=_os.getenv("PG_HOST", "localhost"),
            port=int(_os.getenv("PG_PORT", "5432")),
            dbname=_os.getenv("PG_DB", "delaypilot_db"),
            user=_os.getenv("PG_USER", "postgres"),
            password=_os.getenv("PG_PASSWORD", "delaypilot2026"),
        )

    _conn = _make_conn()
    try:
        with _conn.cursor() as _cur:
            _cur.execute(
                """SELECT id, triggered_by, triggered_at, started_at,
                          finished_at, status, current_step, step_detail,
                          outcome, error_message, backfill_start, backfill_end,
                          api_calls_made, new_auc15, new_prauc15,
                          new_auc30, new_prauc30, new_mae_reg
                   FROM retrain_jobs WHERE id = %s""",
                (job_id,),
            )
            _row = _cur.fetchone()
    finally:
        _conn.close()

    if _row is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404, detail=f"Job {job_id} not found."
        )

    return {
        "job_id":         _row[0],
        "triggered_by":   _row[1],
        "triggered_at":   _row[2].isoformat() if _row[2] else None,
        "started_at":     _row[3].isoformat() if _row[3] else None,
        "finished_at":    _row[4].isoformat() if _row[4] else None,
        "status":         _row[5],
        "current_step":   _row[6],
        "step_detail":    _row[7],
        "outcome":        _row[8],
        "error_message":  _row[9],
        "backfill_start": str(_row[10]) if _row[10] else None,
        "backfill_end":   str(_row[11]) if _row[11] else None,
        "api_calls_made": _row[12],
        "candidate_metrics": {
            "auc15":   _row[13],
            "prauc15": _row[14],
            "auc30":   _row[15],
            "prauc30": _row[16],
            "mae_reg": _row[17],
        } if _row[13] is not None else None,
    }


# For running with: python api_main.py (useful in dev)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)

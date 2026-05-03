import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


logger = logging.getLogger(__name__)


class V3FinalModelService:
    """
    Loader + inference helper for the v3_final CatBoost models.

    It:
      - loads metadata_v3_final.json and cause_groups_v3_final.json
      - loads the 3 CatBoost models (bin15, bin30, reg2_delay_ge5)
      - exposes a predict_one() method that:
          * takes a dict/Series/DataFrame with all required features
          * applies CatBoost-ready preprocessing (categoricals + numeric coercion)
          * returns p_delay_15, p_delay_30 and UI-friendly minutes prediction
    """

    def __init__(self, models_dir: Optional[str] = None,
                 metadata_filename: Optional[str] = None) -> None:
        base_dir = Path(models_dir) if models_dir is not None else Path(__file__).parent / "models"
        self.models_dir = base_dir

        _meta_filename   = metadata_filename or "metadata_v3_final.json"
        _groups_filename = _meta_filename.replace("metadata_", "cause_groups_")
        meta_path   = self.models_dir / _meta_filename
        groups_path = self.models_dir / _groups_filename

        if not meta_path.exists():
            logger.error("metadata_v3_final.json not found at %s", meta_path)
            raise FileNotFoundError(f"metadata_v3_final.json not found at {meta_path}")

        logger.info("Loading model metadata from %s", meta_path)
        self.metadata: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.cause_groups: Dict[str, str] = {}
        if groups_path.exists():
            logger.info("Loading cause groups from %s", groups_path)
            self.cause_groups = json.loads(groups_path.read_text(encoding="utf-8"))

        self.feature_names = list(self.metadata.get("features", []))
        self.cat_features = list(self.metadata.get("categorical_features", []))
        self.thresholds = dict(self.metadata.get("thresholds", {}))

        # Resolve model paths: prefer metadata["paths"], but fall back to local files
        paths = self.metadata.get("paths", {})
        self.bin15 = CatBoostClassifier()
        self.bin30 = CatBoostClassifier()
        self.reg2 = CatBoostRegressor()

        bin15_path = self._resolve_model_path(paths.get("bin15"), "cb_bin15_v3_final.cbm")
        bin30_path = self._resolve_model_path(paths.get("bin30"), "cb_bin30_v3_final.cbm")
        reg2_path = self._resolve_model_path(paths.get("reg2_delay_ge5"), "cb_reg_delay_ge5_v3_final.cbm")

        logger.info("Loading CatBoost model bin15 from %s", bin15_path)
        self.bin15.load_model(str(bin15_path))
        logger.info("Loading CatBoost model bin30 from %s", bin30_path)
        self.bin30.load_model(str(bin30_path))
        logger.info("Loading CatBoost model reg2_delay_ge5 from %s", reg2_path)
        self.reg2.load_model(str(reg2_path))

    def _resolve_model_path(self, configured: Optional[str], fallback_name: str) -> Path:
        """
        Resolve a model path:
          - if `configured` exists on disk, use it
          - otherwise, look for <models_dir>/<fallback_name>
        """
        if configured:
            p = Path(configured)
            if p.exists():
                return p

        local = self.models_dir / fallback_name
        if local.exists():
            return local

        logger.error(
            "Could not find model file for %s. Tried metadata path=%r and local=%s",
            fallback_name,
            configured,
            local,
        )
        raise FileNotFoundError(
            f"Could not find model file for {fallback_name}. "
            f"Tried metadata path: {configured!r} and local: {local}"
        )

    # ---------- Preprocessing ----------

    def _to_feature_dataframe(self, raw: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        """
        Convert an input row (dict/Series/1-row DataFrame) into a DataFrame
        with exactly the columns expected by the model (in the correct order).
        Missing features are filled with NaN and handled in preprocessing.
        """
        if isinstance(raw, pd.DataFrame):
            # Assume already 1 row with at least the feature columns
            df = raw.copy()
            if len(df) != 1:
                raise ValueError("Expected a single-row DataFrame for prediction.")
            # Ensure all expected features exist
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan
            return df[self.feature_names]

        if isinstance(raw, pd.Series):
            data = {k: raw.get(k, np.nan) for k in self.feature_names}
        else:
            # Mapping / dict-like
            data = {k: raw.get(k, np.nan) for k in self.feature_names}

        return pd.DataFrame([data], columns=self.feature_names)

    def _prepare_for_catboost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply CatBoost-compatible preprocessing:
          - fill categoricals with '__MISSING__'
          - coerce numerics, replace inf, fill NaNs with 0

        This mirrors the logic from the training notebooks in a simplified way.
        """
        X = df.copy()

        # Ensure categories exist even if metadata list is out-of-sync
        cat_cols = [c for c in self.cat_features if c in X.columns]

        for c in cat_cols:
            X[c] = X[c].astype("object").fillna("__MISSING__")

        num_cols = [c for c in X.columns if c not in cat_cols]
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        # Replace inf values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill NaN in numeric columns
        X[num_cols] = X[num_cols].fillna(0)

        # Handle pandas nullable integer NA (pd.NA) which is distinct from
        # np.nan and is NOT caught by fillna(0) when dtype is still object.
        # Convert all numeric columns to plain float64 - the only dtype
        # CatBoost's C++ layer accepts without internal NaN conversion issues.
        X[num_cols] = X[num_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce").fillna(0.0).astype("float64")
        )

        return X

    # ---------- Prediction ----------

    def predict_one(self, feature_row: Mapping[str, Any] | pd.Series | pd.DataFrame) -> Dict[str, Any]:
        """
        Run all 3 v3_final models on a single feature row.

        Returns:
          - p_delay_15, p_delay_30               (probabilities)
          - pred_delay_15, pred_delay_30         (binary labels using best thresholds)
          - minutes_pred                         (raw regressor output)
          - minutes_ui                           (UI-rounded minutes with guardrails)
          - thresholds                           (thresholds used)
        """
        try:
            X_df = self._to_feature_dataframe(feature_row)
            X_cb = self._prepare_for_catboost(X_df)
            import math
            import numpy as np

            # Classifier: bin15
            try:
                _p15_raw = self.bin15.predict_proba(X_cb)[:, 1][0]
                p15 = float(_p15_raw)
                if math.isnan(p15) or math.isinf(p15):
                    p15 = 0.0
            except Exception:
                p15 = 0.0

            # Classifier: bin30
            try:
                _p30_raw = self.bin30.predict_proba(X_cb)[:, 1][0]
                p30 = float(_p30_raw)
                if math.isnan(p30) or math.isinf(p30):
                    p30 = 0.0
            except Exception:
                p30 = 0.0

            # Regressor: reg2
            try:
                _min_raw = self.reg2.predict(X_cb)[0]
                minutes_pred = float(np.nan_to_num(float(_min_raw), nan=0.0, posinf=0.0, neginf=0.0))
            except Exception:
                minutes_pred = 0.0

            def _safe_float(val, default):
                try:
                    v = float(val)
                    return default if (math.isnan(v) or math.isinf(v)) else v
                except (TypeError, ValueError):
                    return default

            thr15 = _safe_float(self.thresholds.get("bin15_best_valid"), 0.3)
            thr30 = _safe_float(self.thresholds.get("bin30_best_valid"), 0.4)
            reg_min_raw = self.thresholds.get("reg_train_delay_min", 5)
            reg_train_min = int(_safe_float(reg_min_raw, 5.0))

            pred15 = 1 if p15 >= thr15 else 0
            pred30 = 1 if p30 >= thr30 else 0

            if p30 >= thr30:
                minutes_ui = max(minutes_pred, 30.0)
            elif p15 >= thr15:
                minutes_ui = max(minutes_pred, float(reg_train_min))
            else:
                minutes_ui = 0.0

            if math.isnan(minutes_ui) or math.isinf(minutes_ui):
                minutes_ui = 0.0

            return {
                "p_delay_15": p15,
                "p_delay_30": p30,
                "pred_delay_15": pred15,
                "pred_delay_30": pred30,
                "minutes_pred": minutes_pred,
                "minutes_ui": minutes_ui,
                "thresholds": {
                    "bin15_best_valid": thr15,
                    "bin30_best_valid": thr30,
                    "reg_train_delay_min": reg_train_min,
                },
            }
        except Exception:
            logger.exception("Model prediction failed")
            raise


__all__ = ["V3FinalModelService"]

"""
Train RandomForestRegressor on stop_arrivals (+ passenger_flow features) for minutes_to_next_bus.
Saves model.pkl (joblib) for Flask app inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
ARRIVALS_PATH = BASE_DIR / "stop_arrivals.csv"
FLOW_PATH = BASE_DIR / "passenger_flow.csv"
MODEL_OUT = BASE_DIR / "model.pkl"

NUMERIC_FEATURES = ["hour_of_day", "delay_min", "avg_passengers_waiting"]
CAT_FEATURES = ["line_id", "stop_id", "traffic_level", "weather_condition"]
TARGET = "minutes_to_next_bus"


def _load_and_merge() -> pd.DataFrame:
    if not ARRIVALS_PATH.is_file():
        raise FileNotFoundError(f"Missing {ARRIVALS_PATH}")
    arrivals = pd.read_csv(ARRIVALS_PATH)
    required = {
        "line_id",
        "stop_id",
        "hour_of_day",
        "delay_min",
        "traffic_level",
        "weather_condition",
        TARGET,
    }
    if not required.issubset(arrivals.columns):
        raise ValueError(f"stop_arrivals.csv must contain columns: {sorted(required)}")

    arrivals["hour_of_day"] = pd.to_numeric(arrivals["hour_of_day"], errors="coerce")
    arrivals["delay_min"] = pd.to_numeric(arrivals["delay_min"], errors="coerce")
    arrivals[TARGET] = pd.to_numeric(arrivals[TARGET], errors="coerce")

    for c in CAT_FEATURES:
        arrivals[c] = arrivals[c].astype(str).str.lower().str.strip()

    if FLOW_PATH.is_file():
        flow = pd.read_csv(FLOW_PATH)
        need_f = {"line_id", "stop_id", "hour_of_day", "avg_passengers_waiting"}
        if need_f.issubset(flow.columns):
            flow["hour_of_day"] = pd.to_numeric(flow["hour_of_day"], errors="coerce")
            flow["avg_passengers_waiting"] = pd.to_numeric(
                flow["avg_passengers_waiting"], errors="coerce"
            )
            agg = flow.groupby(["line_id", "stop_id", "hour_of_day"], as_index=False)[
                "avg_passengers_waiting"
            ].mean()
            for c in ("line_id", "stop_id"):
                agg[c] = agg[c].astype(str).str.strip()
            merged = arrivals.merge(
                agg, on=["line_id", "stop_id", "hour_of_day"], how="left"
            )
        else:
            merged = arrivals.copy()
            merged["avg_passengers_waiting"] = np.nan
    else:
        merged = arrivals.copy()
        merged["avg_passengers_waiting"] = np.nan

    med_wait = float(
        pd.to_numeric(merged["avg_passengers_waiting"], errors="coerce").median()
    )
    if np.isnan(med_wait):
        med_wait = 0.0
    merged["avg_passengers_waiting"] = pd.to_numeric(
        merged["avg_passengers_waiting"], errors="coerce"
    ).fillna(med_wait)

    merged = merged.dropna(subset=[TARGET, "hour_of_day", "delay_min"])
    merged = merged[
        (merged[TARGET] >= 0.5) & (merged[TARGET] <= 120.0)
    ]  # trim extreme outliers for stability

    return merged


def build_pipeline() -> Pipeline:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CAT_FEATURES),
        ],
        remainder="drop",
    )

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=24,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("prep", preprocessor), ("rf", rf)])


def main() -> int:
    print("[train_model] Loading and merging CSVs…")
    df = _load_and_merge()
    print(f"[train_model] Rows after cleaning: {len(df):,}")

    X = df[NUMERIC_FEATURES + CAT_FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    print("[train_model] Fitting RandomForestRegressor…")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print("[train_model] Test metrics:")
    print(f"  MAE:  {mae:.4f} min")
    print(f"  RMSE: {rmse:.4f} min")
    print(f"  R²:   {r2:.4f}")

    artifact = {
        "pipeline": pipe,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "target": TARGET,
            "features": NUMERIC_FEATURES + CAT_FEATURES,
        },
    }

    joblib.dump(artifact, MODEL_OUT)
    print(f"[train_model] Saved {MODEL_OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[train_model] ERROR: {e}", file=sys.stderr)
        raise

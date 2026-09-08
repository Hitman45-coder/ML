from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .data import FEATURES


def build_model(max_iter: int = 350, learning_rate: float = 0.06) -> TransformedTargetRegressor:
    estimator = Pipeline([
        ("regressor", HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=31,
            l2_regularization=0.5, random_state=42,
        )),
    ])
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)


def hour_workday_baseline(train_frame: Any, frame: Any) -> np.ndarray:
    """Return a leakage-safe mean demand baseline keyed by hour and working-day."""
    grouped = train_frame.groupby(["hr", "workingday"])["cnt"].mean()
    fallback = float(train_frame["cnt"].mean())
    values = [grouped.get((int(hour), int(workingday)), fallback)
              for hour, workingday in zip(frame["hr"], frame["workingday"])]
    return np.asarray(values, dtype=float)


def evaluate_predictions(y_true: Any, y_pred: Any) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

    true = np.maximum(np.asarray(y_true, dtype=float), 0)
    pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "rmsle": float(np.sqrt(mean_squared_log_error(true, pred))),
    }


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run `make train` first.")
    bundle = joblib.load(path)
    if bundle.get("features") != FEATURES or "model" not in bundle:
        raise ValueError("Model artifact has an incompatible schema")
    return bundle


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

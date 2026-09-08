from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .data import FEATURES, build_features, chronological_split, load_hourly_data
from .model import (
    build_model,
    evaluate_predictions,
    hour_workday_baseline,
    save_bundle,
    write_json,
)


def train(
    data_path: Path,
    artifacts_dir: Path,
    enable_mlflow: bool = False,
) -> dict[str, dict[str, float]]:
    frame = load_hourly_data(data_path)
    train_frame, valid_frame, test_frame = chronological_split(frame)
    x_train, y_train = build_features(train_frame)
    x_valid, y_valid = build_features(valid_frame)
    x_test, y_test = build_features(test_frame)

    candidate_configs = [250, 350, 500]
    candidates = [build_model(max_iter=max_iter) for max_iter in candidate_configs]
    candidate_results = []
    candidate_validation_mae = {}
    for max_iter, candidate in zip(candidate_configs, candidates):
        candidate.fit(x_train, y_train)
        validation_mae = evaluate_predictions(y_valid, candidate.predict(x_valid))["mae"]
        candidate_results.append((validation_mae, candidate))
        candidate_validation_mae[str(max_iter)] = validation_mae
    _, model = min(candidate_results, key=lambda result: result[0])
    valid_pred = model.predict(x_valid)
    valid_metrics = evaluate_predictions(y_valid, valid_pred)
    baseline_pred = hour_workday_baseline(train_frame, valid_frame)
    baseline_metrics = evaluate_predictions(y_valid, baseline_pred)

    final_x = pd.concat([x_train, x_valid])
    final_y = pd.concat([y_train, y_valid])
    model.fit(final_x, final_y)
    test_pred = model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_pred)
    residuals = (y_valid.to_numpy() - valid_pred).tolist()
    residual_q = float(pd.Series(residuals).abs().quantile(0.90))
    selected_max_iter = model.regressor_.named_steps["regressor"].max_iter
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    split_boundaries = {
        "train_end": train_frame.timestamp.max().isoformat(),
        "validation_end": valid_frame.timestamp.max().isoformat(),
        "test_end": test_frame.timestamp.max().isoformat(),
    }
    bundle = {
        "model": model, "features": FEATURES, "model_version": "0.1.0",
        "package_version": __version__,
        "residual_quantile_90": residual_q,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_max_iter": selected_max_iter,
    }
    save_bundle(bundle, artifacts_dir / "model.joblib")
    report = {
        "validation_model": valid_metrics, "validation_baseline": baseline_metrics,
        "test_model": test_metrics,
        "split_boundaries": split_boundaries,
        "data_sha256": digest, "features": FEATURES,
        "selected_max_iter": selected_max_iter,
        "candidate_validation_mae": candidate_validation_mae,
    }
    write_json(report, artifacts_dir / "metrics.json")
    write_json(
        {
            "model_version": "0.1.0",
            "package_version": __version__,
            "trained_at": bundle["trained_at"],
            "data_sha256": digest,
            "features": FEATURES,
            "split_boundaries": split_boundaries,
            "selected_max_iter": bundle["selected_max_iter"],
            "candidate_validation_mae": candidate_validation_mae,
        },
        artifacts_dir / "metadata.json",
    )
    if enable_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("Install the optional tracking extra before using --mlflow") from exc
        with mlflow.start_run():
            mlflow.log_params(
                {"max_iter": bundle["selected_max_iter"], "target_transform": "log1p"}
            )
            mlflow.log_metrics(
                {f"validation_model_{key}": value for key, value in valid_metrics.items()}
            )
            mlflow.log_metrics(
                {f"test_model_{key}": value for key, value in test_metrics.items()}
            )
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
    return {
        "validation_model": valid_metrics,
        "validation_baseline": baseline_metrics,
        "test_model": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/hour.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--mlflow", action="store_true", help="Log this run to an MLflow server")
    args = parser.parse_args()
    print(train(args.data, args.artifacts, enable_mlflow=args.mlflow))


if __name__ == "__main__":
    main()

import importlib

import pytest
from fastapi.testclient import TestClient

from demand_forecasting.data import FEATURES, build_features
from demand_forecasting.model import build_model, save_bundle

from .test_data import sample_frame


def test_health_degraded_without_model(monkeypatch, tmp_path):
    module = importlib.import_module("demand_forecasting.api")
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "missing.joblib")
    module._bundle = None
    assert TestClient(module.app).get("/health").json()["model_loaded"] is False


def test_forecast_fails_safely_without_model(monkeypatch, tmp_path):
    module = importlib.import_module("demand_forecasting.api")
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "missing.joblib")
    module._bundle = None
    response = TestClient(module.app).post(
        "/v1/forecast",
        json={
            "timestamp": "2012-06-01T10:00:00", "season": 2, "holiday": 0,
            "workingday": 1, "weathersit": 1, "temperature_c": 20.5,
            "feels_like_c": 25.0, "humidity_percent": 50.0, "windspeed_kmh": 13.4,
        },
    )
    assert response.status_code == 503


def test_root_explains_where_to_use_the_api():
    module = importlib.import_module("demand_forecasting.api")
    response = TestClient(module.app).get("/")
    assert response.status_code == 200
    assert "Demand Forecasting" in response.text
    assert "text/html" in response.headers["content-type"]


def test_favicon_request_is_handled():
    module = importlib.import_module("demand_forecasting.api")
    assert TestClient(module.app).get("/favicon.ico").status_code == 204


def test_metrics_endpoint_returns_training_report(monkeypatch, tmp_path):
    module = importlib.import_module("demand_forecasting.api")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"test_model": {"mae": 1.0}}', encoding="utf-8")
    monkeypatch.setattr(module, "METRICS_PATH", metrics_path)
    response = TestClient(module.app).get("/v1/metrics")
    assert response.status_code == 200
    assert response.json()["test_model"]["mae"] == 1.0


def test_forecast_contract(monkeypatch, tmp_path):
    x, y = build_features(sample_frame(30))
    model = build_model().fit(x, y)
    path = tmp_path / "model.joblib"
    save_bundle(
        {
            "model": model,
            "features": FEATURES,
            "model_version": "test",
            "residual_quantile_90": 5.0,
        },
        path,
    )
    module = importlib.import_module("demand_forecasting.api")
    monkeypatch.setattr(module, "MODEL_PATH", path)
    module._bundle = None
    response = TestClient(module.app).post("/v1/forecast", json={
        "timestamp": "2012-06-01T10:00:00", "season": 2, "holiday": 0,
        "workingday": 1, "weathersit": 1, "temperature_c": 20.5,
        "feels_like_c": 25.0, "humidity_percent": 50.0, "windspeed_kmh": 13.4,
    })
    assert response.status_code == 200
    assert response.json()["model_version"] == "test"
    second = TestClient(module.app).post(
        "/v1/forecast",
        json={
            "timestamp": "2012-06-01T10:00:00", "season": 2, "holiday": 0,
            "workingday": 1, "weathersit": 1, "temperature_c": 20.5,
            "feels_like_c": 25.0, "humidity_percent": 50.0, "windspeed_kmh": 13.4,
        },
    )
    assert second.json()["forecast_demand"] == response.json()["forecast_demand"]


def test_forecast_rejects_invalid_humidity():
    module = importlib.import_module("demand_forecasting.api")
    response = TestClient(module.app).post(
        "/v1/forecast",
        json={
            "timestamp": "2012-06-01T10:00:00", "season": 2, "holiday": 0,
            "workingday": 1, "weathersit": 1, "temperature_c": 20.5,
            "feels_like_c": 25.0, "humidity_percent": 101.0, "windspeed_kmh": 13.4,
        },
    )
    assert response.status_code == 422


def test_forecast_accepts_human_weather_units(monkeypatch, tmp_path):
    x, y = build_features(sample_frame(30))
    model = build_model().fit(x, y)
    path = tmp_path / "model.joblib"
    save_bundle(
        {
            "model": model,
            "features": FEATURES,
            "model_version": "units",
            "residual_quantile_90": 5.0,
        },
        path,
    )
    module = importlib.import_module("demand_forecasting.api")
    monkeypatch.setattr(module, "MODEL_PATH", path)
    module._bundle = None
    response = TestClient(module.app).post(
        "/v1/forecast",
        json={
            "timestamp": "2012-06-01T10:00:00", "season": 2, "holiday": 0,
            "workingday": 1, "weathersit": 1, "temperature_c": 50.0,
            "feels_like_c": 55.0, "humidity_percent": 50.0, "windspeed_kmh": 90.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["model_version"] == "units"


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature_c", 50.1), ("feels_like_c", 55.1), ("windspeed_kmh", 90.1)],
)
def test_forecast_rejects_expanded_weather_bounds(field, value):
    module = importlib.import_module("demand_forecasting.api")
    payload = {
        "timestamp": "2012-06-01T10:00:00", "season": 2, "holiday": 0,
        "workingday": 1, "weathersit": 1, "temperature_c": 20.5,
        "feels_like_c": 25.0, "humidity_percent": 50.0, "windspeed_kmh": 13.4,
    }
    payload[field] = value
    assert TestClient(module.app).post("/v1/forecast", json=payload).status_code == 422

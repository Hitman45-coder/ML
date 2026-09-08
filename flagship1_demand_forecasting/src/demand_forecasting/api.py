from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import FEATURES, normalize_weather_inputs
from .model import load_bundle

app = FastAPI(title="Demand Forecasting Decision API", version="0.1.0")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "artifacts/metrics.json"))
STATIC_DIR = Path(__file__).parent / "static"
PREPARE_THRESHOLD = float(os.getenv("PREPARE_THRESHOLD", "250"))
SURGE_THRESHOLD = float(os.getenv("SURGE_THRESHOLD", "500"))
_bundle = None
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


class ForecastRequest(BaseModel):
    timestamp: datetime
    season: int = Field(ge=1, le=4)
    holiday: int = Field(ge=0, le=1)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temperature_c: float = Field(ge=0, le=41, description="Outdoor temperature in °C")
    feels_like_c: float = Field(ge=0, le=50, description="Feels-like temperature in °C")
    humidity_percent: float = Field(ge=0, le=100, description="Relative humidity in percent")
    windspeed_kmh: float = Field(ge=0, le=67, description="Wind speed in km/h")


class ForecastResponse(BaseModel):
    forecast_demand: float
    planning_lower: float
    planning_upper: float
    recommendation: str
    model_version: str
    latency_ms: float


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the browser UI from the same origin as the prediction API."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Avoid a browser-generated 404 when opening the service directly."""
    return Response(status_code=204)


@app.get("/v1/metrics")
def metrics() -> JSONResponse:
    """Expose the immutable training report for the portfolio dashboard."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics report not found. Run `make train` first.",
        )
    return JSONResponse(content=json.loads(METRICS_PATH.read_text(encoding="utf-8")))


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle(MODEL_PATH)
    return _bundle


@app.get("/health")
def health() -> dict[str, str | bool]:
    try:
        bundle = _get_bundle()
        return {"status": "ok", "model_loaded": True, "model_version": bundle["model_version"]}
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    started = time.perf_counter()
    try:
        bundle = _get_bundle()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    ts = request.timestamp
    weather = normalize_weather_inputs(
        temperature_c=request.temperature_c,
        feels_like_c=request.feels_like_c,
        humidity_percent=request.humidity_percent,
        windspeed_kmh=request.windspeed_kmh,
    )
    row = {
        "season": request.season, "yr": int(ts.year >= 2012), "mnth": ts.month,
        "holiday": request.holiday, "weekday": ts.weekday(), "workingday": request.workingday,
        "weathersit": request.weathersit, **weather, "hour": ts.hour,
        "day_of_year": ts.timetuple().tm_yday,
    }
    prediction = max(0.0, float(bundle["model"].predict(pd.DataFrame([row], columns=FEATURES))[0]))
    band = float(bundle.get("residual_quantile_90", 0.0))
    recommendation = (
        "SURGE"
        if prediction + band >= SURGE_THRESHOLD
        else "PREPARE"
        if prediction + band >= PREPARE_THRESHOLD
        else "NORMAL"
    )
    return ForecastResponse(
        forecast_demand=round(prediction, 2), planning_lower=round(max(0, prediction - band), 2),
        planning_upper=round(prediction + band, 2), recommendation=recommendation,
        model_version=bundle["model_version"],
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )

# Demand Forecasting Decision Service

This project forecasts hourly bike demand and turns the forecast into a simple operational planning signal. It is a production-style portfolio system: the model uses chronological evaluation, excludes post-outcome leakage, persists a versioned artifact, and is served by a validated FastAPI contract.

## Data and responsible use

The model uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), downloaded by `scripts/download_data.py`. It is a learning/demo system, not a live dispatch recommendation. The dataset represents one historical service and should not be assumed to generalize to a new city or season.

## Quickstart

```bash
# uv is recommended; it creates/uses a locked project environment.
uv sync --extra dev
make install
make train
make test
make lint
make run-api
```

If `uv` is unavailable, create and activate a standard virtual environment first (`python3 -m venv .venv && source .venv/bin/activate`); `make install` will use pip as a fallback.

Then open `/` for the browser dashboard, `/docs` for interactive API documentation, check `/health`, or call:

```bash
curl -X POST http://localhost:8000/v1/forecast \
  -H 'content-type: application/json' \
  -d '{"timestamp":"2012-06-01T10:00:00","season":2,"holiday":0,"workingday":1,"weathersit":1,"temp":0.5,"atemp":0.5,"hum":0.5,"windspeed":0.2}'
```

The training command creates ignored local files in `artifacts/`: `model.joblib`, `metrics.json`, and `metadata.json`. The split is 70% train, 15% validation, and 15% final test in time order. `casual`, `registered`, and `cnt` never enter the feature matrix.

### Docker

Train the model on the host first, then build and start the API. The runtime image contains the serving code only; the trained artifacts are mounted read-only.

```bash
make train
docker compose up --build
```

The dashboard and API are then available at `http://localhost:8000`. Stop the foreground service with `Ctrl-C`, or add `-d` to run it in the background and use `docker compose logs -f` to inspect logs.

The image defaults to the non-root `appuser`. On restricted rootless environments that cannot map subordinate container UIDs, run `CONTAINER_USER=0 docker compose up --build` as a local compatibility fallback.

## API contract

`GET /` serves the browser dashboard. It calls the same-origin API and shows the forecast, planning interval, recommendation, model version, latency, and held-out metrics. `/docs` remains available as the interactive OpenAPI console.

`POST /v1/forecast` accepts timestamp, weather, and calendar inputs. It returns a non-negative demand forecast, a documented planning band from the 90th percentile validation residual, a coarse `NORMAL`/`PREPARE`/`SURGE` recommendation, model version, and request latency. Set `PREPARE_THRESHOLD` and `SURGE_THRESHOLD` environment variables to tune the operational cutoffs. `/health` reports a degraded state until a model artifact exists.

## Engineering decisions

- A histogram gradient boosting regressor handles nonlinear weather/calendar interactions while remaining inexpensive to train and serve.
- The target is log-transformed during fitting to reduce the effect of demand skew; metrics are reported after inverse transformation in demand units.
- Validation is used for model/band decisions; the chronological test set is held out for the final report.
- CI runs Ruff, pytest, and a Docker build. Runtime containers do not download data or train models.
- The UI is deliberately served by FastAPI from the same container as the backend, so a deployed demo has no cross-origin configuration or separate frontend build step.
- Run `python3 -m demand_forecasting.train --mlflow` only after installing the optional `tracking` extra and configuring an MLflow tracking URI; ordinary training has no tracking-server dependency.

## Next improvements

Use a stronger seasonal/time-series baseline, prediction intervals with conformal calibration, a business cost simulation for stock-outs versus excess capacity, MLflow tracking, drift checks on incoming weather distributions, and a real deployment with protected secrets and telemetry. These are intentionally not claimed as completed features.

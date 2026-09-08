from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

RAW_COLUMNS = {
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
}
TARGET = "cnt"
EXCLUDED = {"instant", "dteday", "hr", "casual", "registered", "cnt"}
FEATURES = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed", "hour", "day_of_year",
]
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_dataset(path: Path, url: str = DATA_URL) -> Path:
    """Download and extract the hourly dataset, returning its local path."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        with ZipFile(BytesIO(response.read())) as archive:
            member = next(name for name in archive.namelist() if name.endswith("hour.csv"))
            path.write_bytes(archive.read(member))
    return path


def load_hourly_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = RAW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "timestamp" not in frame or TARGET not in frame:
        raise ValueError("Input must contain timestamp and cnt columns")
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_year"] = ts.dt.dayofyear
    missing = [c for c in FEATURES if c not in out]
    if missing:
        raise ValueError(f"Input is missing feature columns: {missing}")
    return out[FEATURES].astype(float), out[TARGET].astype(float)


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    if min(train_end, valid_end - train_end, n - valid_end) < 1:
        raise ValueError("At least three chronological rows are required")
    return ordered.iloc[:train_end], ordered.iloc[train_end:valid_end], ordered.iloc[valid_end:]

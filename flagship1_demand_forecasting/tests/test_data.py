import pandas as pd
import pytest

from demand_forecasting.data import FEATURES, build_features, chronological_split, download_dataset


def sample_frame(n=20):
    timestamps = pd.date_range("2012-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps, "season": 1, "yr": 1, "mnth": timestamps.month,
        "holiday": 0, "weekday": timestamps.dayofweek, "workingday": 1,
        "weathersit": 1, "temp": 0.5, "atemp": 0.5, "hum": 0.5, "windspeed": 0.2,
        "cnt": range(n), "casual": range(n), "registered": range(n),
    })


def test_features_exclude_post_outcome_columns_and_add_calendar_features():
    x, y = build_features(sample_frame())
    assert list(x.columns) == FEATURES
    assert not {"cnt", "casual", "registered"}.intersection(x.columns)
    assert len(x) == len(y) == 20


def test_split_is_chronological_and_non_overlapping():
    train, valid, test = chronological_split(sample_frame())
    assert train.timestamp.max() < valid.timestamp.min() < test.timestamp.min()


def test_small_frames_fail_with_clear_error():
    with pytest.raises(ValueError, match="At least three"):
        chronological_split(sample_frame(2))


def test_download_dataset_skips_existing_file(tmp_path):
    target = tmp_path / "hour.csv"
    target.write_text("already downloaded", encoding="utf-8")
    assert download_dataset(target) == target
    assert target.read_text(encoding="utf-8") == "already downloaded"

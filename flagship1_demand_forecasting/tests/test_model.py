from demand_forecasting.data import build_features
from demand_forecasting.model import build_model, evaluate_predictions

from .test_data import sample_frame


def test_model_trains_and_returns_nonnegative_predictions():
    x, y = build_features(sample_frame(30))
    model = build_model()
    model.fit(x, y)
    assert (model.predict(x) >= 0).all()


def test_metrics_are_finite_and_named():
    metrics = evaluate_predictions([1, 2, 3], [1, 3, 2])
    assert set(metrics) == {"mae", "rmse", "rmsle"}
    assert all(value >= 0 for value in metrics.values())

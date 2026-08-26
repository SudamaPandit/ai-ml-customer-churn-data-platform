import pandas as pd

from src.monitor import classification_monitoring, population_stability_index


def test_psi_is_zero_for_same_population():
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert population_stability_index(values, values) == 0.0


def test_monitoring_returns_metrics():
    metrics = classification_monitoring(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0.1, 0.9, 0.2, 0.8]),
    )
    assert metrics["roc_auc"] == 1.0
    assert 0.0 <= metrics["positive_prediction_rate"] <= 1.0

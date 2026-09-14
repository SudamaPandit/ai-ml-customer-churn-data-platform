import pandas as pd
import pytest

from src.monitor import classification_monitoring, population_stability_index


def test_psi_is_zero_for_same_population():
    values = pd.Series(range(1, 11))
    assert population_stability_index(values, values) == 0.0


def test_psi_includes_values_outside_reference_range():
    score = population_stability_index(pd.Series(range(1, 101)), pd.Series(range(101, 201)))
    assert score > 0


def test_monitoring_returns_metrics():
    metrics = classification_monitoring(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0.1, 0.9, 0.2, 0.8]),
    )
    assert metrics["roc_auc"] == 1.0


def test_monitoring_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        classification_monitoring(pd.Series([0, 1]), pd.Series([0.2, 1.2]))

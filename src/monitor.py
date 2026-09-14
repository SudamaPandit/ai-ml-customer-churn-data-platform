from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        raise ValueError("reference and current populations must not be empty")

    edges = reference.quantile([i / bins for i in range(bins + 1)]).drop_duplicates().to_numpy()
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_dist = pd.cut(reference, bins=edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    current_dist = pd.cut(current, bins=edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    ref = reference_dist.clip(lower=1e-6)
    cur = current_dist.reindex(reference_dist.index, fill_value=1e-6).clip(lower=1e-6)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def classification_monitoring(
    y_true: pd.Series, probabilities: pd.Series
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("labels and probabilities must be non-empty and equal length")
    if not probabilities.between(0, 1).all():
        raise ValueError("probabilities must be between 0 and 1")
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "positive_prediction_rate": float((probabilities >= 0.5).mean()),
    }

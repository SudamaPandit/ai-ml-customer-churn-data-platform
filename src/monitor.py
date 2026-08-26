from __future__ import annotations

import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = reference.quantile([i / bins for i in range(bins + 1)]).drop_duplicates().sort_values().to_numpy()
    if len(edges) < 3:
        return 0.0
    reference_dist = pd.cut(reference, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    current_dist = pd.cut(current, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    ref = reference_dist.clip(lower=1e-6)
    cur = current_dist.reindex(reference_dist.index, fill_value=1e-6).clip(lower=1e-6)
    return float(((cur - ref) * (cur / ref).apply(lambda x: __import__('math').log(x))).sum())


def classification_monitoring(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "positive_prediction_rate": float((probabilities >= 0.5).mean()),
    }

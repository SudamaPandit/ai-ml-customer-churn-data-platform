from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import build_features


def score_customers(model: Pipeline, customers: pd.DataFrame) -> pd.DataFrame:
    features = build_features(customers)
    probabilities = model.predict_proba(features)[:, 1]
    result = customers[["customer_id"]].copy()
    result["churn_probability"] = probabilities
    result["risk_band"] = pd.cut(
        probabilities,
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return result

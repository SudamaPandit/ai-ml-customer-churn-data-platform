from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = ["tenure_months", "monthly_charges", "support_tickets"]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "charges_per_tenure_month",
    "tickets_per_tenure_month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(BASE_FEATURE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    result = df.copy()
    numeric = result[BASE_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Feature values must be finite numbers")
    if numeric.lt(0).any().any():
        raise ValueError("Feature values cannot be negative")
    result[BASE_FEATURE_COLUMNS] = numeric
    denominator = result["tenure_months"].clip(lower=1)
    result["charges_per_tenure_month"] = result["monthly_charges"] / denominator
    result["tickets_per_tenure_month"] = result["support_tickets"] / denominator
    return result

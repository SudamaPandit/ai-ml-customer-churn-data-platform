from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "charges_per_tenure_month",
    "tickets_per_tenure_month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["charges_per_tenure_month"] = result["monthly_charges"] / result["tenure_months"].clip(lower=1)
    result["tickets_per_tenure_month"] = result["support_tickets"] / result["tenure_months"].clip(lower=1)
    return result

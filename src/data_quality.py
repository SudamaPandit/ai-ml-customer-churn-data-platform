from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "churned",
}
NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "support_tickets"]


def validate_customer_dataset(df: pd.DataFrame) -> dict[str, float | int]:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Customer dataset must not be empty")
    if df["customer_id"].isna().any() or not df["customer_id"].is_unique:
        raise ValueError("customer_id must be non-null and unique")

    numeric = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("numeric features must be finite numbers")
    if numeric.lt(0).any().any():
        raise ValueError("tenure, charges, and ticket counts cannot be negative")

    target = pd.to_numeric(df["churned"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("churned must be binary")

    return {
        "row_count": int(len(df)),
        "churn_rate": float(target.mean()),
        "null_rate": float(df.isna().mean().mean()),
    }

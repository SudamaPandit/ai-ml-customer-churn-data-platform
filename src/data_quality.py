from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "churned",
}


def validate_customer_dataset(df: pd.DataFrame) -> dict[str, float | int]:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Customer dataset must not be empty")
    if df["customer_id"].isna().any():
        raise ValueError("customer_id contains nulls")
    if not df["customer_id"].is_unique:
        raise ValueError("customer_id must be unique")
    if df["tenure_months"].lt(0).any():
        raise ValueError("tenure_months cannot be negative")
    if df["monthly_charges"].lt(0).any():
        raise ValueError("monthly_charges cannot be negative")
    if df["support_tickets"].lt(0).any():
        raise ValueError("support_tickets cannot be negative")
    if not df["churned"].isin([0, 1, True, False]).all():
        raise ValueError("churned must be binary")

    return {
        "row_count": int(len(df)),
        "churn_rate": float(df["churned"].astype(int).mean()),
        "null_rate": float(df.isna().mean().mean()),
    }

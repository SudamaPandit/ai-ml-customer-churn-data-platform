import pandas as pd
import pytest

from src.data_quality import validate_customer_dataset


def sample_df():
    return pd.DataFrame([
        {"customer_id": 1, "tenure_months": 12, "monthly_charges": 80, "support_tickets": 2, "churned": 0},
        {"customer_id": 2, "tenure_months": 3, "monthly_charges": 100, "support_tickets": 5, "churned": 1},
    ])


def test_valid_dataset_returns_metrics():
    metrics = validate_customer_dataset(sample_df())
    assert metrics["row_count"] == 2
    assert metrics["churn_rate"] == 0.5


def test_missing_columns_fail():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_customer_dataset(sample_df().drop(columns=["churned"]))


def test_duplicate_customer_ids_fail():
    df = sample_df()
    df.loc[1, "customer_id"] = 1
    with pytest.raises(ValueError, match="unique"):
        validate_customer_dataset(df)


def test_negative_values_fail():
    df = sample_df()
    df.loc[0, "monthly_charges"] = -1
    with pytest.raises(ValueError, match="monthly_charges"):
        validate_customer_dataset(df)

import numpy as np
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
    assert metrics["row_count"] == 2 and metrics["churn_rate"] == 0.5


@pytest.mark.parametrize("column,value,message", [
    ("customer_id", 1, "unique"),
    ("monthly_charges", -1, "negative"),
    ("support_tickets", np.nan, "finite"),
    ("tenure_months", "bad", "finite"),
    ("churned", 2, "binary"),
])
def test_invalid_dataset_fails(column, value, message):
    df = sample_df()
    df.loc[1, column] = value
    with pytest.raises(ValueError, match=message):
        validate_customer_dataset(df)


def test_missing_columns_fail():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_customer_dataset(sample_df().drop(columns=["churned"]))

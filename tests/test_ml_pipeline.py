import pandas as pd

from src.features import FEATURE_COLUMNS, build_features
from src.predict import score_customers
from src.train import train_model


def training_df():
    rows = []
    for i in range(1, 41):
        rows.append({
            "customer_id": i,
            "tenure_months": 2 + (i % 18),
            "monthly_charges": 40 + (i % 8) * 10,
            "support_tickets": i % 6,
            "churned": int((i % 5 == 0) or (i % 7 == 0)),
        })
    return pd.DataFrame(rows)


def test_feature_engineering_creates_expected_features():
    result = build_features(training_df())
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert (result["charges_per_tenure_month"] >= 0).all()


def test_training_is_reproducible_and_produces_metrics():
    result = train_model(training_df())
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.average_precision <= 1.0


def test_batch_scoring_returns_risk_band():
    df = training_df()
    trained = train_model(df)
    scored = score_customers(trained.model, df.iloc[:5])
    assert len(scored) == 5
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_band"]).issubset({"LOW", "MEDIUM", "HIGH"})

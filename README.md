# Customer Churn Prediction Pipeline

A batch machine-learning example that validates customer data, builds
reproducible features, trains a scikit-learn classifier, scores customers, and
calculates drift/performance metrics.

## Implemented flow

1. `src/data_quality.py` rejects missing, non-finite, negative, duplicate, or
   invalid target data.
2. `src/features.py` creates tenure-normalized behavioral features and uses
   the same code for training and scoring.
3. `src/train.py` trains a scaled, class-weighted logistic regression model
   and returns ROC-AUC and average precision.
4. `src/predict.py` emits churn probability and LOW/MEDIUM/HIGH risk bands.
5. `src/monitor.py` calculates PSI and classification monitoring metrics.
6. `src/storage.py` can publish prediction Parquet to S3.
7. `glue/prepare_features.py` shows the equivalent distributed feature
   expressions for an externally managed Glue job.

The PostgreSQL schema keeps prediction history with a composite
`(customer_id, prediction_date)` key.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Tests are offline and do not require AWS credentials. Optional Bedrock
summarization is disabled unless `AI_ENABLED=true` and a model ID is set.

## Current boundaries

- Training and model persistence are local; MLflow or another registry is not
  implemented.
- There is no deployable Airflow DAG or Terraform in this repository.
- The Glue file contains feature expressions, not a complete Glue entry point.
- S3 publication is implemented but not exercised by CI.
- Labels in this compact sample are assumed to be point-in-time correct;
  temporal label construction is not implemented.
- There is no automated retraining, alert delivery, rollback, or online scoring.

This project intentionally demonstrates ML-oriented data validation, feature
consistency, batch scoring, SQL history, and monitoring without claiming a
production MLOps platform.

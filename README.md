# AI/ML Customer Churn Data Platform

A machine learning pipeline for predicting customer churn, covering data preparation, feature engineering, model training, batch scoring, and model monitoring.

## Architecture

```text
PostgreSQL / Source Data
        |
        v
   AWS S3 Bronze
        |
        v
 AWS Glue / PySpark
        |
        +--> Data Quality
        |
        v
   S3 Silver / Features
        |
        v
 Feature Engineering (Python)
        |
        +------------------+
        |                  |
        v                  v
   ML Training        Batch Scoring
 XGBoost / sklearn         |
        |                  v
        v             Predictions
      MLflow                 |
        |                   v
        v             Redshift / S3 Gold
   Model Registry           |
        |                   v
        +------------> Monitoring
                          |
                    Drift / Quality
                          |
                          v
                 Airflow / Alerts

Optional AI operations:
Aggregate monitoring metrics -> Amazon Bedrock -> incident summary
```

## What the project covers

- Source-to-feature data pipeline for repeatable model training.
- PySpark transformations for scalable feature preparation.
- Point-in-time aware feature generation to reduce training leakage.
- Reproducible training with explicit feature and schema contracts.
- MLflow tracking and model registry integration.
- Batch scoring and prediction persistence for downstream analytics.
- Data-quality checks before training and scoring.
- Model-performance and feature-drift monitoring.
- Airflow orchestration for training, scoring, and monitoring workflows.
- Terraform definitions for AWS infrastructure.
- CI tests that run without AWS credentials.
- Optional Bedrock integration for operational summaries; model/data quality rules remain deterministic.

## Technology Stack

Python, SQL, PostgreSQL, Pandas, NumPy, scikit-learn, XGBoost, PySpark, MLflow, Apache Airflow, Amazon S3, AWS Glue, Amazon Redshift, Amazon SageMaker, Terraform, Docker, GitHub Actions, Amazon Bedrock.

## Repository Structure

```text
src/
  data_quality.py       # deterministic dataset validation
  features.py           # reusable feature engineering
  train.py              # model training and evaluation
  predict.py            # batch scoring
  monitor.py            # drift/performance metrics
  ai_insights.py        # optional Bedrock operational summaries
  storage.py            # S3 persistence adapter

dags/
  churn_ml_pipeline.py  # Airflow orchestration

glue/
  prepare_features.py   # distributed feature preparation

sql/
  schema.sql
  analytics.sql

infra/
  main.tf
  variables.tf

tests/
  test_data_quality.py
  test_features.py
  test_train.py
  test_predict.py
  test_monitor.py
  test_ai_insights.py

.github/workflows/
  ci.yml
```

## Local Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The CI workflow avoids cloud credentials and external service calls. AWS, MLflow, and Bedrock integrations are isolated behind adapters so the test suite stays deterministic and fast.

## ML Lifecycle

1. Ingest source/customer history.
2. Validate schema, nulls, ranges, duplicates, and target quality.
3. Build reusable features using historical information only.
4. Train and evaluate the churn classifier.
5. Track metrics and parameters in MLflow.
6. Register the candidate model.
7. Batch-score current customers.
8. Persist predictions for analytics and retention workflows.
9. Monitor input drift and model performance.
10. Use optional AI assistance to summarize operational anomalies for engineers.

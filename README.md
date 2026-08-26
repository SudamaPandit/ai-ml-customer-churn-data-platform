# AI/ML Customer Churn Data Platform

Production-oriented machine learning platform demonstrating how a Senior Data Engineer can build the data foundation, ML pipeline, and MLOps lifecycle for customer churn prediction.

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

## Senior Data Engineering Focus

- Source-to-feature data pipeline designed for repeatable ML training.
- PySpark transformations for scalable feature preparation.
- Point-in-time aware feature generation to reduce training leakage risk.
- Reproducible model training with explicit feature/schema contracts.
- MLflow tracking and model registry integration point.
- Batch scoring and prediction persistence for downstream analytics.
- Data-quality checks before training and scoring.
- Model-performance and feature-drift monitoring.
- Airflow orchestration for training, scoring, and monitoring workflows.
- Terraform for AWS infrastructure definitions.
- CI executes unit and ML tests without requiring AWS credentials.
- Optional Bedrock integration for operational summaries only; AI never bypasses deterministic quality gates.

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
  storage.py             # S3 persistence adapter

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

The CI pipeline intentionally excludes cloud credentials and external service calls. AWS/MLflow/Bedrock integrations are isolated behind adapters so the test suite remains deterministic and fast.

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

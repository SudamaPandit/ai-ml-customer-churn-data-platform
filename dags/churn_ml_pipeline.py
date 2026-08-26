from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def validate():
    print("Validate source data quality")


def train():
    print("Train candidate churn model and log to MLflow")


def score():
    print("Batch score current customers and publish predictions")


def monitor():
    print("Calculate drift and performance metrics")

with DAG(
    dag_id="customer_churn_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["ml", "churn", "data-engineering"],
) as dag:
    quality = PythonOperator(task_id="validate", python_callable=validate)
    training = PythonOperator(task_id="train", python_callable=train)
    scoring = PythonOperator(task_id="score", python_callable=score)
    monitoring = PythonOperator(task_id="monitor", python_callable=monitor)
    quality >> training >> scoring >> monitoring

from __future__ import annotations

import os
from io import BytesIO

import pandas as pd


def write_predictions_to_s3(df: pd.DataFrame, key: str) -> str:
    """Persist predictions to S3; credentials are obtained from the AWS runtime."""
    import boto3

    bucket = os.environ["PREDICTIONS_BUCKET"]
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    return f"s3://{bucket}/{key}"

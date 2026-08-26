from __future__ import annotations

import json
import os
from typing import Any


def build_insight_prompt(metrics: dict[str, Any]) -> str:
    return (
        "Review these aggregate ML monitoring metrics. Identify anomalies, "
        "likely engineering causes, and three investigation actions. "
        "Do not infer or request personal data.\n"
        f"Metrics: {json.dumps(metrics, sort_keys=True)}"
    )


def summarize_with_bedrock(metrics: dict[str, Any]) -> str:
    if os.getenv("AI_ENABLED", "false").lower() != "true":
        return "AI insights disabled"
    model_id = os.environ["BEDROCK_MODEL_ID"]
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": build_insight_prompt(metrics)}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"]

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_quality import validate_customer_dataset
from src.features import FEATURE_COLUMNS, build_features


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    roc_auc: float
    average_precision: float


def train_model(df: pd.DataFrame, seed: int = 42) -> TrainingResult:
    validate_customer_dataset(df)
    y = df["churned"].astype(int)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Training requires both churn classes with at least two rows each")

    prepared = build_features(df)
    x = prepared[FEATURE_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )

    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([("scale", StandardScaler(), FEATURE_COLUMNS)])),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return TrainingResult(
        model=pipeline,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        average_precision=float(average_precision_score(y_test, probabilities)),
    )

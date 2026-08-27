# Customer Churn Prediction Pipeline

Batch ML pipeline that scores customers weekly on churn risk — from raw
transactional data to a monitored model in production. Built to mirror how
I'd actually stand this up at work: reproducible features, a model registry,
drift monitoring, and no manual steps between "new data lands" and
"predictions are refreshed."

## Why this shape

Churn models rot quietly. The usual failure mode isn't a crash — it's
someone six months later scoring on a feature set that's subtly different
from what the model was trained on, or a training set that leaked
future information into the labels. So the two things I optimized for
here weren't accuracy, they were:

- **Point-in-time correctness** — features are built only from data that
  would have actually been available at prediction time. No accidental
  leakage from "future" columns during training.
- **A feature/schema contract that can't drift silently** — `data_quality.py`
  runs before training *and* before every scoring run, and it fails loudly
  rather than scoring against a dataset that's changed shape underneath it.

## Architecture

```
Source data (Postgres)
       │
       ▼
  S3 raw landing
       │
       ▼
  Glue/PySpark ── feature prep at scale
       │
       ▼
  S3 feature store (silver)
       │
   ┌───┴────┐
   ▼        ▼
Training  Batch scoring
(XGBoost/  │
sklearn)   ▼
   │     Predictions → S3/Redshift
   ▼
MLflow (tracking + registry)
   │
   ▼
Drift & performance monitoring ── Airflow-orchestrated, alerts on threshold breach
```
<img width="1536" height="1024" alt="AIML Customer Churn Data Platform" src="https://github.com/user-attachments/assets/94f90c20-deea-4860-bf2f-f2c1d39b7b46" />


An optional step summarizes monitoring anomalies via Bedrock — see
"On the Bedrock piece" below for how that's scoped.

## Repository structure

```
src/
  data_quality.py   deterministic validation — nulls, ranges, dupes, target sanity
  features.py       point-in-time feature construction
  train.py          training + evaluation, logs to MLflow
  predict.py        batch scoring
  monitor.py        drift + performance metrics vs. training baseline
  ai_insights.py     optional Bedrock summarization of monitoring output
  storage.py        S3 adapter, isolated so it's mockable in tests
dags/churn_ml_pipeline.py   Airflow DAG wiring the above together
glue/prepare_features.py   the distributed version of feature prep
sql/                        schema + analytics queries
infra/                      Terraform for the AWS pieces
tests/                       unit tests, run without any AWS/MLflow credentials
```

## Running it locally

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Everything that touches AWS, MLflow, or Bedrock sits behind an adapter
(`storage.py`, the MLflow client wrapper, `ai_insights.py`), so the test
suite runs fully offline and fast. That's also just good practice —
I didn't want a CI run to depend on a live AWS account existing.

## How I actually ran the AWS parts

I didn't keep a Glue cluster or Redshift instance running for a personal
project — that adds up fast with no offsetting value. What I did:

- Developed and unit-tested the Spark logic locally against sample data
  (same PySpark APIs Glue uses under the hood).
- Ran the Glue job itself a handful of times against S3 free-tier storage
  to validate it end-to-end and capture logs/output — a few hundred rupees
  total, not an ongoing cost.
- Swapped Redshift for Postgres for the "warehouse" layer, since the point
  I'm demonstrating is the data-modeling and access pattern, not Redshift's
  specific columnar internals.
- SageMaker is referenced as the natural home for training at real scale,
  but training here runs locally — the pipeline code doesn't care where
  the compute happens.

## On the Bedrock piece

The monitoring step can optionally send *aggregate* metrics (row counts,
drift scores, null-rate deltas — never raw customer data) to Bedrock to
draft a plain-English incident summary. It's explicitly advisory: the
actual pass/fail gate is the deterministic checks in `data_quality.py` and
`monitor.py`. I added this because "engineer bolts an LLM onto everything"
is an easy trap, and I wanted the boundary between "deterministic gate"
and "AI-assisted convenience" to be obvious in the code, not just the docs.

## What I'd change at real production scale

- Feature store would move to something purpose-built (Feast, or a
  warehouse-native feature layer) instead of flat S3/Parquet — this works
  for a batch-weekly use case but wouldn't scale to many models sharing
  features.
- Model rollback is currently manual via the MLflow registry UI; I'd wire
  automatic rollback on monitoring-threshold breach.
- No online/real-time scoring path — this is batch-only by design, but a
  churn-intervention system usually wants at least a near-real-time trigger
  eventually.

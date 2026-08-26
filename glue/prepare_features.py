"""AWS Glue/PySpark entry point for distributed feature preparation."""

from pyspark.sql import functions as F


def prepare_features(df):
    return (
        df.withColumn("charges_per_tenure_month", F.col("monthly_charges") / F.greatest(F.col("tenure_months"), F.lit(1)))
          .withColumn("tickets_per_tenure_month", F.col("support_tickets") / F.greatest(F.col("tenure_months"), F.lit(1)))
    )

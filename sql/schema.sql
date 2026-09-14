CREATE TABLE IF NOT EXISTS customer_churn_predictions (
    customer_id BIGINT NOT NULL,
    prediction_date DATE NOT NULL,
    churn_probability NUMERIC(8,6) NOT NULL
        CHECK (churn_probability BETWEEN 0 AND 1),
    risk_band VARCHAR(10) NOT NULL
        CHECK (risk_band IN ('LOW', 'MEDIUM', 'HIGH')),
    model_version VARCHAR(100) NOT NULL,
    PRIMARY KEY (customer_id, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_date
    ON customer_churn_predictions(prediction_date);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_risk
    ON customer_churn_predictions(risk_band);

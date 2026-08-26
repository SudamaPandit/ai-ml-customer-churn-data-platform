SELECT
    prediction_date,
    risk_band,
    COUNT(*) AS customer_count,
    AVG(churn_probability) AS avg_churn_probability
FROM customer_churn_predictions
GROUP BY prediction_date, risk_band
ORDER BY prediction_date DESC, risk_band;

SELECT
    prediction_date,
    COUNT(*) FILTER (WHERE risk_band = 'HIGH') AS high_risk_customers,
    AVG(churn_probability) AS avg_churn_probability
FROM customer_churn_predictions
GROUP BY prediction_date
ORDER BY prediction_date DESC;

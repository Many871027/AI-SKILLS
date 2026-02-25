# MLOps Monitoring and Observability

A model's performance degrades the moment it is deployed. Monitor the following metrics continuously:

1.  **Operational Telemetry:**
    * **Latency:** Inference time must stay below the required SLA (e.g., < 100ms).
    * **Throughput:** Requests per second (RPS) the API is handling.
    * **Error Rates:** HTTP 4xx (bad data) and 5xx (server/model failure) codes.
2.  **Data Drift (Covariate Shift):**
    * Monitor the distribution of incoming features compared to the training data.
    * Use statistical tests like the **Kolmogorov-Smirnov (K-S) test** for continuous variables or **Population Stability Index (PSI)**. If PSI > 0.2, trigger an alert for significant drift.
3.  **Concept Drift:**
    * Monitor the relationship between the features and the target variable. If the fundamental reality changes (e.g., a new type of fraud emerges), the model must be retrained immediately.
---
name: ML-LIFECYCLE
description: Orchestrates an end-to-end Machine Learning lifecycle from data ingestion and automated preprocessing to model training, hyperparameter tuning, and MLflow logging. Use when the user asks to "run an ML pipeline", "train a model on a dataset", "optimize hyperparameters", or "track experiments in MLflow" on any tabular or preprocessed dataset.
---

# ML Full Cycle Orchestration
## Instructions

You are executing a production-grade Machine Learning pipeline. [cite_start]When a user provides a dataset and a target variable, follow this sequential workflow orchestration[cite: 486].

### Step 1: Data Validation & EDA
1. Inspect the provided dataset format (CSV, Parquet, JSON).
2. Identify the target column specified by the user.
3. Conduct brief univariate analysis to identify categorical vs. numerical features and check for severe class imbalances.

### Step 2: Pipeline Execution
Run the generalized ML pipeline script to handle preprocessing, training, and tracking.
Execute the following command, replacing the variables with the user's inputs:

```bash
python scripts/generalized_ml_pipeline.py --data_path "PATH_TO_DATA" --target_col "TARGET_COLUMN_NAME" --experiment_name "EXPERIMENT_NAME"
```

---

### Step 3: Observability & Evaluation

After the pipeline finishes execution:

1. Review the generated `classification_report_[model].txt` and confusion matrices.
2. Select the **champion model** based on the highest **F1-Score** (to account for potential class imbalances).
3. Translate the technical metrics (Precision, Recall, F1) into business impact.

> **📌 Reference:** Consult `references/business_kpi_mapping.md` for business translation frameworks.

> **📄 Output Requirement:** Format your final pipeline summary using the structures defined in `assets/model_card_template.md` and `assets/data_storytelling_report.md`.

---

### Step 4: MLOps Deployment & Serving

Transition the champion model to the deployment phase:

1. Explain the architectural trade-offs between **Shadow Testing**, **Canary Releases**, and **A/B Testing**.
2. Guide the user to serve the model using the provided FastAPI script:

   ```bash
   python scripts/serve_model.py
   ```

3. Advise the user to containerize the solution for Kubernetes or Cloud Run using the `scripts/Dockerfile`.

> **📌 Reference:** Consult `references/deployment_strategies.md` and `references/monitoring_observability.md` to define a monitoring strategy (tracking **Inference Latency** and **Data Drift**).

---

### Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Missing Target Column** | The user provided a column name that doesn't exactly match the dataset headers. | Check the dataset schema and ask the user to clarify the exact target column name. |
| **Non-Numeric Features for Naive Bayes** | Raw string data passed to models requiring numerical input. | Assure the user that `generalized_ml_pipeline.py` automatically applies `OneHotEncoder` to categorical variables and `MinMaxScaler` to numerical ones to prevent this issue. |
| **Port 8000 Already in Use** | Another application is running on the default FastAPI port during Step 4. | Advise the user to run the uvicorn server on a different port (e.g., `--port 8001`). |

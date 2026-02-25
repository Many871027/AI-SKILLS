# ML-LIFECYCLE 🚀

Welcome to the **ML-LIFECYCLE** repository! This project encapsulates an autonomous, agentic skill designed to orchestrate an end-to-end Machine Learning pipeline. From raw data ingestion to production deployment, it seamlessly guides you through a complete lifecycle with fully integrated MLOps practices.

---

## 📖 Overview

The `ML-LIFECYCLE` skill acts as an intelligent orchestrator. It expects a dataset (structured tabular data like CSV) and automates standard Data Science workflows across three primary ML paradigms:

1. **Supervised Classification:** Hyperparameter grid-search across various algorithms to predict target labels.
2. **Unsupervised Clustering:** Segmenting your dataset without prior labels to discover intrinsic structures.
3. **Anomaly Detection & Dimensionality Reduction:** Outlier detection and principal component analysis (PCA) for visual insights.

Throughout the process, the skill heavily utilizes **MLflow** for rigorous experiment tracking, recording models, metrics (F1-score, Silhouette Score, etc.), and generating visual artifacts like Confusion Matrices and Scatter Plots.

---

## 🛠 Features pipeline

### 1. Data Validation & Preprocessing
* Automated Data Type Inference (Distinguishes between categorical and numerical columns).
* Numerical features: Mean imputation + `MinMaxScaler`.
* Categorical features: Mode imputation + `OneHotEncoder`.
* Resolves class imbalances using **SMOTE** dynamically during supervised learning.

### 2. Generalization Pipeline (`scripts/generalized_ml_pipeline.py`)
This script contains the backbone of the entire training stage! Once initiated, it branches out into three distinct sub-pipelines.

* **Classification Models Supported:**
  * `LogisticRegression`
  * `RandomForest`
  * `K-Nearest Neighbors (KNN)`
  * `Gaussian Naive Bayes`
  * `DecisionTree`
* **Clustering Models:**
  * `KMeans`
  * `DBSCAN`
  * `GaussianMixture`
* **Anomaly/Dimensionality Strategies:**
  * `IsolationForest`
  * `PCA (Principal Component Analysis)`

*All models map out hyperparameter grids, evaluating configuration permutations. The best parameters are flagged as the Champion models.*

### 3. MLOps Model Deployment (`scripts/serve_model.py`)
Following evaluation, models can instantly be served using **FastAPI**. 
* Loads the champion model straight from the MLflow artifact tracking registry during the application's ASGI `lifespan`.
* Integrates an asynchronous framework for rapid REST inferences.
* Exposes `/health` to verify model accessibility and `/predict` for dynamic JSON payload inferences.

---

## 🚀 Quickstart

### Prerequisites:
Ensure your environment is configured for python data science and ML:
```bash
pip install -r requirements.txt
```
*(Requirements typically include `pandas`, `scikit-learn`, `imbalanced-learn`, `mlflow`, `fastapi`, `uvicorn`, `pydantic`, `matplotlib`)*

### Running the Pipeline:
Select a dataset (e.g., `ai4i2020.csv`), and launch the generalization pipeline:
```bash
python .agent/skills/ML-LIFECYCLE/scripts/generalized_ml_pipeline.py \
  --data_path "your_dataset.csv" \
  --target_col "target_class_column" \
  --experiment_name "predictive_maintenance_agent"
```
Once run, observe your local directory for the `mlruns` tracking folder which stores metrics, tags, and pickled models safely!

### Serving the Best Model:
Once you have trained the models and established your Champions, kick off the FastAPI Inference Engine!
```bash
python .agent/skills/ML-LIFECYCLE/scripts/serve_model.py
```
> Navigate to `http://localhost:8000/docs` to test out the endpoints via the generated Swagger UI interface.

---

## 🚧 Common Issues & Troubleshooting

* **Missing Target Column:** Always verify the dataset headers exactly match the spelling/casing provided via `--target_col`.
* **OS Error (No space left on device):** MLFlow logs many models to your local disk by default. For massive parameter grids or massive serialized tree models (like Random Forest), verify disk space availability before tracking hundreds of permutations.
* **Port conflicts:** If you're running multiple inference services, supply a different port manually within `uvicorn.run()` or via command-line arguments.

---

*This repository embodies a modular, scalable blueprint suitable for rapid experimentation environments!*

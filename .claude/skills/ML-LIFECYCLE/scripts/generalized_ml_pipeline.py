"""
Generalized ML Pipeline - Multi-Paradigm
==========================================
Supports three ML paradigms in a single orchestrated pipeline:
  1. Supervised Classification (with SMOTE + hyperparameter grid search)
  2. Unsupervised Clustering (KMeans, DBSCAN, GaussianMixture)
  3. Anomaly Detection & Dimensionality Reduction (IsolationForest, PCA)

All experiments are tracked in MLflow with full artifact logging.
"""

import argparse
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt

# --- Supervised ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# --- Unsupervised & Clustering ---
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# --- Sklearn Utilities ---
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score
)

import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
import scipy.sparse as sp


# =============================================================================
# PREPROCESSING
# =============================================================================

def build_preprocessor(X):
    """
    Build a dynamic preprocessing pipeline based on data types.
    - Numeric: Impute nulls with mean -> MinMaxScaler [0, 1]
    - Categorical: Impute nulls with mode -> OneHotEncoder
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def to_dense(X):
    """Convert sparse matrix to dense array if needed."""
    if sp.issparse(X):
        return X.toarray()
    return X


# =============================================================================
# 1. SUPERVISED CLASSIFICATION PIPELINE
# =============================================================================

def run_supervised_pipeline(X_train, X_test, y_train, y_test, preprocessor, experiment_name):
    """
    Train and evaluate supervised classification models with hyperparameter grids.
    Applies SMOTE for class balancing. Logs all results to MLflow.
    """
    print("\n" + "=" * 70)
    print("  SUPERVISED CLASSIFICATION PIPELINE")
    print("=" * 70)

    # Preprocess
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("Preprocessing complete: Numerics scaled [0,1], Categoricals encoded.")

    # SMOTE for class imbalance
    print(f"Original training samples: {X_train_processed.shape[0]}. Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
        print(f"Post-SMOTE training samples: {X_train_res.shape[0]}.")
    except Exception as e:
        print(f"SMOTE failed ({e}). Using original data.")
        X_train_res, y_train_res = X_train_processed, y_train

    # Dense conversion (required for GaussianNB, MLP, etc.)
    X_train_res = to_dense(X_train_res)
    X_test_processed = to_dense(X_test_processed)

    # Model registry with hyperparameter grids
    models_and_params = [
        {'name': 'LogisticRegression', 'class': LogisticRegression, 
         'grid': [{'C': [0.1, 1, 10], 'max_iter': [1000]}]},
        {'name': 'RandomForest', 'class': RandomForestClassifier, 
         'grid': [{'n_estimators': [50, 100], 'max_depth': [None, 10]}]},
        # SVM and MLP Neural Net are computationally expensive (e.g., O(n^2) or O(n^3))
        # and limit scalability on large datasets, so they are avoided here.
        # GradientBoosting is also omitted to prioritize fast convergence models.
        {'name': 'KNN', 'class': KNeighborsClassifier, 
         'grid': [{'n_neighbors': [3, 5, 7]}]},
        {'name': 'GaussianNB', 'class': GaussianNB, 
         'grid': [{'var_smoothing': [1e-9, 1e-8, 1e-7]}]},
        {'name': 'DecisionTree', 'class': DecisionTreeClassifier, 
         'grid': [{'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}]},
    ]

    mlflow.set_experiment(f"{experiment_name}_Supervised")
    best_f1 = 0
    best_model_name = ""

    for m in models_and_params:
        model_name = m['name']
        ModelClass = m['class']
        grid = ParameterGrid(m['grid'])

        print(f"\n--- Evaluating {model_name} ({len(grid)} configs) ---")

        for i, params in enumerate(grid):
            run_name = f"{model_name}_v{i}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.set_tag("task_type", "classification")
                mlflow.set_tag("smote_applied", "True")
                mlflow.log_param("model_type", model_name)

                # Train
                model = ModelClass(**params)
                model.fit(X_train_res, y_train_res)

                # Predict
                y_pred = model.predict(X_test_processed)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score_weighted", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                # Classification report
                report = classification_report(y_test, y_pred, zero_division=0)
                mlflow.log_text(report, f"classification_report_{model_name}.txt")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title(f'Confusion Matrix - {model_name} v{i}')
                fig.colorbar(im, ax=ax)
                tick_marks = np.arange(len(np.unique(y_test)))
                ax.set_xticks(tick_marks)
                ax.set_xticklabels(np.unique(y_test), rotation=45)
                ax.set_yticks(tick_marks)
                ax.set_yticklabels(np.unique(y_test))
                ax.set_ylabel('True label')
                ax.set_xlabel('Predicted label')
                fig.tight_layout()

                cm_filename = f"cm_{model_name}_{i}.png"
                fig.savefig(cm_filename)
                mlflow.log_artifact(cm_filename)
                plt.close(fig)
                os.remove(cm_filename)

                # Log the full pipeline (preprocessor + classifier)
                full_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                mlflow.sklearn.log_model(full_pipeline, f"model_{model_name}")

                print(f"  v{i}: Params={params} -> Acc={acc:.4f}, F1={f1:.4f}")

                # Track champion
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = f"{model_name}_v{i}"

    print(f"\n{'=' * 70}")
    print(f"  CHAMPION (Supervised): {best_model_name} | F1={best_f1:.4f}")
    print(f"{'=' * 70}")

    return best_model_name, best_f1


# =============================================================================
# 2. UNSUPERVISED CLUSTERING PIPELINE
# =============================================================================

def run_clustering_pipeline(X_train, preprocessor, experiment_name):
    """
    Train clustering models and evaluate with Silhouette Score.
    Logs all results and cluster visualizations to MLflow.
    """
    print("\n" + "=" * 70)
    print("  UNSUPERVISED CLUSTERING PIPELINE")
    print("=" * 70)

    X_processed = preprocessor.fit_transform(X_train)
    X_processed = to_dense(X_processed)
    print(f"Clustering on {X_processed.shape[0]} samples, {X_processed.shape[1]} features.")

    models_and_params = [
        {'name': 'KMeans', 'class': KMeans, 
         'grid': [{'n_clusters': [3, 5, 7], 'random_state': [42]}]},
        {'name': 'DBSCAN', 'class': DBSCAN, 
         'grid': [{'eps': [0.3, 0.5, 1.0], 'min_samples': [5, 10]}]},
        {'name': 'GaussianMixture', 'class': GaussianMixture, 
         'grid': [{'n_components': [3, 5, 7], 'random_state': [42]}]},
    ]

    mlflow.set_experiment(f"{experiment_name}_Clustering")
    best_silhouette = -1
    best_cluster_model = ""

    for m in models_and_params:
        model_name = m['name']
        ModelClass = m['class']
        grid = ParameterGrid(m['grid'])

        print(f"\n--- Evaluating {model_name} ({len(grid)} configs) ---")

        for i, params in enumerate(grid):
            run_name = f"{model_name}_v{i}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.set_tag("task_type", "clustering")
                mlflow.log_param("model_type", model_name)

                model = ModelClass(**params)

                # GaussianMixture uses fit/predict, others use fit_predict
                if model_name == 'GaussianMixture':
                    labels = model.fit(X_processed).predict(X_processed)
                else:
                    labels = model.fit_predict(X_processed)

                n_unique = len(set(labels) - {-1})  # Exclude noise label for DBSCAN
                mlflow.log_metric("n_clusters_found", n_unique)

                if n_unique > 1:
                    sil_score = silhouette_score(X_processed, labels)
                    mlflow.log_metric("silhouette_score", sil_score)
                    print(f"  v{i}: Params={params} -> Clusters={n_unique}, Silhouette={sil_score:.4f}")

                    if sil_score > best_silhouette:
                        best_silhouette = sil_score
                        best_cluster_model = f"{model_name}_v{i}"
                else:
                    mlflow.log_metric("silhouette_score", -1)
                    print(f"  v{i}: Params={params} -> Only {n_unique} cluster(s) found. Skipping silhouette.")

                # 2D visualization using PCA
                try:
                    pca_2d = PCA(n_components=2)
                    X_2d = pca_2d.fit_transform(X_processed)

                    fig, ax = plt.subplots(figsize=(10, 7))
                    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.5, s=10)
                    fig.colorbar(scatter, ax=ax, label='Cluster')
                    ax.set_title(f'Cluster Visualization - {model_name} v{i}')
                    ax.set_xlabel('PCA Component 1')
                    ax.set_ylabel('PCA Component 2')
                    fig.tight_layout()

                    viz_filename = f"cluster_viz_{model_name}_{i}.png"
                    fig.savefig(viz_filename, dpi=100)
                    mlflow.log_artifact(viz_filename)
                    plt.close(fig)
                    os.remove(viz_filename)
                except Exception as e:
                    print(f"  Warning: Could not generate cluster visualization: {e}")

                mlflow.sklearn.log_model(model, f"model_{model_name}")

    if best_silhouette > -1:
        print(f"\n{'=' * 70}")
        print(f"  BEST CLUSTERING: {best_cluster_model} | Silhouette={best_silhouette:.4f}")
        print(f"{'=' * 70}")

    return best_cluster_model, best_silhouette


# =============================================================================
# 3. ANOMALY DETECTION & DIMENSIONALITY REDUCTION PIPELINE
# =============================================================================

def run_anomaly_dimred_pipeline(X_train, y_train, preprocessor, experiment_name):
    """
    Run anomaly detection (IsolationForest) and dimensionality reduction (PCA).
    Logs all results to MLflow.
    """
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION & DIMENSIONALITY REDUCTION PIPELINE")
    print("=" * 70)

    X_processed = preprocessor.fit_transform(X_train)
    X_processed = to_dense(X_processed)
    print(f"Processing {X_processed.shape[0]} samples, {X_processed.shape[1]} features.")

    mlflow.set_experiment(f"{experiment_name}_Anomaly_DimRed")

    # --- Anomaly Detection: Isolation Forest ---
    anomaly_params = [
        {'name': 'IsolationForest', 'class': IsolationForest,
         'grid': [{'contamination': [0.01, 0.05, 0.1], 'random_state': [42]}]},
    ]

    for m in anomaly_params:
        model_name = m['name']
        ModelClass = m['class']
        grid = ParameterGrid(m['grid'])

        print(f"\n--- Evaluating {model_name} ({len(grid)} configs) ---")

        for i, params in enumerate(grid):
            run_name = f"{model_name}_v{i}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.set_tag("task_type", "anomaly_detection")
                mlflow.log_param("model_type", model_name)

                model = ModelClass(**params)
                predictions = model.fit_predict(X_processed)

                n_anomalies = int((predictions == -1).sum())
                n_normal = int((predictions == 1).sum())
                anomaly_ratio = n_anomalies / len(predictions)

                mlflow.log_metric("n_anomalies", n_anomalies)
                mlflow.log_metric("n_normal", n_normal)
                mlflow.log_metric("anomaly_ratio", anomaly_ratio)

                # If we have actual labels, compare anomaly predictions vs real failures
                if y_train is not None:
                    # Map IsolationForest: -1 (anomaly) -> 1 (failure), 1 (normal) -> 0
                    pred_labels = np.where(predictions == -1, 1, 0)
                    y_arr = np.array(y_train)
                    
                    if len(np.unique(y_arr)) == 2:
                        acc = accuracy_score(y_arr, pred_labels)
                        f1 = f1_score(y_arr, pred_labels, average='weighted', zero_division=0)
                        mlflow.log_metric("accuracy_vs_labels", acc)
                        mlflow.log_metric("f1_vs_labels", f1)
                        print(f"  v{i}: contamination={params.get('contamination', 'N/A')} -> "
                              f"Anomalies={n_anomalies}, Normal={n_normal}, "
                              f"Acc_vs_labels={acc:.4f}, F1_vs_labels={f1:.4f}")
                    else:
                        print(f"  v{i}: contamination={params.get('contamination', 'N/A')} -> "
                              f"Anomalies={n_anomalies}, Normal={n_normal}")
                else:
                    print(f"  v{i}: contamination={params.get('contamination', 'N/A')} -> "
                          f"Anomalies={n_anomalies}, Normal={n_normal}")

                # Anomaly visualization
                try:
                    pca_2d = PCA(n_components=2)
                    X_2d = pca_2d.fit_transform(X_processed)

                    fig, ax = plt.subplots(figsize=(10, 7))
                    colors = np.where(predictions == -1, 'red', 'blue')
                    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.4, s=10)
                    ax.set_title(f'Anomaly Detection - {model_name} v{i} (Red=Anomaly)')
                    ax.set_xlabel('PCA Component 1')
                    ax.set_ylabel('PCA Component 2')
                    fig.tight_layout()

                    viz_filename = f"anomaly_viz_{model_name}_{i}.png"
                    fig.savefig(viz_filename, dpi=100)
                    mlflow.log_artifact(viz_filename)
                    plt.close(fig)
                    os.remove(viz_filename)
                except Exception as e:
                    print(f"  Warning: Could not generate anomaly visualization: {e}")

                mlflow.sklearn.log_model(model, f"model_{model_name}")

    # --- Dimensionality Reduction: PCA ---
    dimred_params = [
        {'name': 'PCA', 'class': PCA,
         'grid': [{'n_components': [2, 5, 10]}]},
    ]

    for m in dimred_params:
        model_name = m['name']
        ModelClass = m['class']
        grid = ParameterGrid(m['grid'])

        print(f"\n--- Evaluating {model_name} ({len(grid)} configs) ---")

        for i, params in enumerate(grid):
            # Ensure n_components does not exceed available features
            n_comp = params.get('n_components', 2)
            if n_comp > X_processed.shape[1]:
                print(f"  v{i}: Skipping n_components={n_comp} (exceeds {X_processed.shape[1]} features)")
                continue

            run_name = f"{model_name}_v{i}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.set_tag("task_type", "dim_reduction")
                mlflow.log_param("model_type", model_name)

                model = ModelClass(**params)
                X_reduced = model.fit_transform(X_processed)

                explained_var_total = float(np.sum(model.explained_variance_ratio_))
                mlflow.log_metric("explained_variance_total", explained_var_total)

                # Log individual component variances
                for j, var_ratio in enumerate(model.explained_variance_ratio_):
                    mlflow.log_metric(f"explained_var_pc{j+1}", float(var_ratio))

                print(f"  v{i}: n_components={n_comp} -> Total Explained Variance={explained_var_total:.4f}")

                # 2D scatterplot if n_components >= 2
                if n_comp >= 2:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        if y_train is not None:
                            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                               c=np.array(y_train), cmap='coolwarm', alpha=0.5, s=10)
                            fig.colorbar(scatter, ax=ax, label='Target')
                        else:
                            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5, s=10)
                        ax.set_title(f'PCA Projection - {n_comp} components (Var={explained_var_total:.2%})')
                        ax.set_xlabel(f'PC1 ({model.explained_variance_ratio_[0]:.2%})')
                        ax.set_ylabel(f'PC2 ({model.explained_variance_ratio_[1]:.2%})')
                        fig.tight_layout()

                        viz_filename = f"pca_viz_{n_comp}comp.png"
                        fig.savefig(viz_filename, dpi=100)
                        mlflow.log_artifact(viz_filename)
                        plt.close(fig)
                        os.remove(viz_filename)
                    except Exception as e:
                        print(f"  Warning: Could not generate PCA visualization: {e}")

                mlflow.sklearn.log_model(model, f"model_{model_name}_{n_comp}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_ml_pipeline(data_path, target_col, experiment_name):
    """
    Main orchestrator: loads data, preprocesses, and runs all three sub-pipelines.
    """
    print(f"Starting ML Pipeline with data: {data_path}")

    # 1. Data Ingestion
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    if target_col not in data.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        print(f"Available columns: {list(data.columns)}")
        return

    # 2. Separate Features and Target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. MLflow Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mlruns_uri = "file:///" + os.path.join(base_dir, "mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(mlruns_uri)
    print(f"MLflow Tracking URI: {mlruns_uri}")

    # 4. Build preprocessor (shared across pipelines)
    preprocessor = build_preprocessor(X_train)

    # === PIPELINE 1: Supervised Classification ===
    best_supervised, best_f1 = run_supervised_pipeline(
        X_train, X_test, y_train, y_test, preprocessor, experiment_name
    )

    # === PIPELINE 2: Unsupervised Clustering ===
    # Rebuild preprocessor for full training data (no labels needed)
    preprocessor_clustering = build_preprocessor(X_train)
    best_cluster, best_sil = run_clustering_pipeline(
        X_train, preprocessor_clustering, experiment_name
    )

    # === PIPELINE 3: Anomaly Detection & Dimensionality Reduction ===
    preprocessor_anomaly = build_preprocessor(X_train)
    run_anomaly_dimred_pipeline(
        X_train, y_train, preprocessor_anomaly, experiment_name
    )

    # Final Summary
    print("\n" + "=" * 70)
    print("  FULL PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Supervised Champion:  {best_supervised} (F1={best_f1:.4f})")
    if best_sil > -1:
        print(f"  Best Clustering:      {best_cluster} (Silhouette={best_sil:.4f})")
    else:
        print(f"  Best Clustering:      No valid clustering found")
    print(f"  Anomaly + PCA:        Logged to MLflow experiment '{experiment_name}_Anomaly_DimRed'")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Multi-Paradigm ML Pipeline")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the CSV data file")
    parser.add_argument("--target_col", type=str, required=True,
                        help="Name of the target column")
    parser.add_argument("--experiment_name", type=str, default="Universal_ML_Pipeline",
                        help="Base experiment name in MLflow")

    args = parser.parse_args()
    run_ml_pipeline(args.data_path, args.target_col, args.experiment_name)
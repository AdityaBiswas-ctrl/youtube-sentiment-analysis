"""
Evaluate the trained sentiment classification model.
DVC Stage 4: Model Evaluation

Computes metrics on test data, logs to MLflow, and saves metrics JSON.
"""

import os
import json
import logging
import warnings

import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# pyrefly: ignore [missing-import]
import mlflow

from src.features.build_features import load_vectorizer, transform_features

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for the evaluate DVC stage."""
    params = load_params()
    config = load_config()
    mlflow_params = params["mlflow"]

    # Configure MLflow
    tracking_uri = mlflow_params.get("tracking_uri", "mlruns")
    
    # Check if we should use local storage (fallback if remote is localhost and likely unavailable in CI)
    use_local = False
    if tracking_uri.startswith("http"):
        import socket
        try:
            # Simple connection check with 2s timeout
            host = tracking_uri.split("//")[-1].split(":")[0]
            port = int(tracking_uri.split(":")[-1]) if ":" in tracking_uri.split("//")[-1] else 80
            with socket.create_connection((host, port), timeout=2):
                pass
        except (Exception, socket.timeout):
            logger.warning(f"Could not connect to MLflow server at {tracking_uri}. Falling back to local 'mlruns'.")
            use_local = True
    
    if use_local:
        mlflow.set_tracking_uri("mlruns")
    else:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(mlflow_params["experiment_name"])

    # Load test data
    test_path = config["paths"]["test_data"]
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    logger.info(f"Test samples: {len(test_df)}")

    # Load model artifacts
    model_path = config["paths"]["model"]
    vectorizer_path = config["paths"]["vectorizer"]
    le_path = config["paths"]["label_encoder"]

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    vectorizer = load_vectorizer(vectorizer_path)

    label_encoder = joblib.load(le_path)
    class_names = label_encoder.classes_.tolist()

    # Transform test data
    X_test = transform_features(test_df["clean_text"], vectorizer)
    y_test = label_encoder.transform(test_df["label"])

    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Per-class metrics
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"Test Accuracy:  {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall:    {recall:.4f}")
    logger.info(f"Test F1-Score:  {f1:.4f}")
    logger.info(
        f"\nClassification Report:\n"
        f"{classification_report(y_test, y_pred, target_names=class_names)}"
    )

    # Build metrics dict
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "model_type": params["train"]["model_type"],
        "num_test_samples": len(test_df),
        "num_classes": len(class_names),
        "classes": class_names,
        "confusion_matrix": cm,
        "per_class": {},
    }

    for cls in class_names:
        if cls in report:
            metrics["per_class"][cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1_score": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }

    # Log to MLflow
    with mlflow.start_run(run_name=f"{params['train']['model_type']}_evaluation"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        for cls in class_names:
            if cls in report:
                mlflow.log_metric(
                    f"{cls}_precision", report[cls]["precision"]
                )
                mlflow.log_metric(
                    f"{cls}_recall", report[cls]["recall"]
                )
                mlflow.log_metric(
                    f"{cls}_f1", report[cls]["f1-score"]
                )

        mlflow.log_param("model_type", params["train"]["model_type"])
        mlflow.log_param("num_test_samples", len(test_df))

        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    # Save metrics JSON (DVC metrics file)
    metrics_path = config["paths"]["metrics"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

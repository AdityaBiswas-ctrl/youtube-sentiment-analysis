"""
Train sentiment classification models (XGBoost / LightGBM).
DVC Stage 3: Model Training

Trains the selected model, logs everything to MLflow,
and saves model artifacts.
"""

import os
import logging
import warnings

import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
# pyrefly: ignore [missing-import]
import mlflow.xgboost

from src.features.build_features import fit_transform_features, save_vectorizer

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


def get_model(model_type: str, params: dict):
    """
    Create a model instance based on the selected type.
    Supports 'xgboost' and 'lightgbm'.
    """
    if model_type == "xgboost":
        # pyrefly: ignore [missing-import]
        from xgboost import XGBClassifier
        model_params = params["train"]["xgboost"]
        model = XGBClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            eval_metric=model_params["eval_metric"],
            use_label_encoder=False,
            random_state=params["preprocess"]["random_state"],
            n_jobs=-1,
        )
        return model, model_params

    elif model_type == "lightgbm":
        # pyrefly: ignore [missing-import]
        from lightgbm import LGBMClassifier
        model_params = params["train"]["lightgbm"]
        model = LGBMClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            num_leaves=model_params["num_leaves"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            random_state=params["preprocess"]["random_state"],
            n_jobs=-1,
            verbose=-1,
        )
        return model, model_params

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose 'xgboost' or 'lightgbm'."
        )


def main():
    """Main entry point for the train DVC stage."""
    params = load_params()
    config = load_config()

    model_type = params["train"]["model_type"]
    mlflow_params = params["mlflow"]

    logger.info(f"Training model: {model_type}")

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

    # Load training data
    train_path = config["paths"]["train_data"]
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path, encoding="utf-8")
    logger.info(f"Training samples: {len(train_df)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    class_names = label_encoder.classes_.tolist()
    logger.info(f"Classes: {class_names}")

    # Build features
    X_train, vectorizer = fit_transform_features(train_df["clean_text"])

    # Create model
    model, model_params = get_model(model_type, params)

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}_training"):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("max_features", params["train"]["max_features"])
        mlflow.log_param("ngram_range", str(params["train"]["ngram_range"]))
        mlflow.log_param("num_train_samples", len(train_df))
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("classes", str(class_names))

        for key, value in model_params.items():
            mlflow.log_param(f"model_{key}", value)

        # Train
        logger.info("Training model...")
        model.fit(X_train, y_train)

        # Training accuracy
        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        mlflow.log_metric("train_accuracy", round(train_accuracy, 4))
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        # Log model to MLflow
        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"youtube-sentiment-{model_type}",
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"youtube-sentiment-{model_type}",
            )

        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    # Save artifacts locally (for DVC tracking)
    model_path = config["paths"]["model"]
    vectorizer_path = config["paths"]["vectorizer"]
    le_path = config["paths"]["label_encoder"]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    save_vectorizer(vectorizer, vectorizer_path)

    joblib.dump(label_encoder, le_path)
    logger.info(f"Saved label encoder to {le_path}")


if __name__ == "__main__":
    main()

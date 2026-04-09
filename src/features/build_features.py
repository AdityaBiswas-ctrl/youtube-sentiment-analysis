"""
Feature extraction utilities.
Builds TF-IDF feature matrices from cleaned text data.
"""

import os
import logging

import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def build_vectorizer(max_features: int = 5000, ngram_range: tuple = (1, 2)):
    """Create a TF-IDF vectorizer with the specified parameters."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        sublinear_tf=True,
        dtype="float32",
    )


def fit_transform_features(texts, vectorizer=None):
    """
    Fit and transform text data to TF-IDF features.
    Returns (feature_matrix, vectorizer).
    """
    params = load_params()
    train_params = params["train"]

    if vectorizer is None:
        ngram_range = tuple(train_params["ngram_range"])
        vectorizer = build_vectorizer(
            max_features=train_params["max_features"],
            ngram_range=ngram_range,
        )

    features = vectorizer.fit_transform(texts)
    logger.info(
        f"Built TF-IDF matrix: {features.shape[0]} samples x "
        f"{features.shape[1]} features"
    )
    return features, vectorizer


def transform_features(texts, vectorizer):
    """Transform text data using a pre-fitted vectorizer."""
    features = vectorizer.transform(texts)
    return features


def save_vectorizer(vectorizer, path: str):
    """Save vectorizer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info(f"Saved vectorizer to {path}")


def load_vectorizer(path: str):
    """Load vectorizer from disk."""
    vectorizer = joblib.load(path)
    logger.info(f"Loaded vectorizer from {path}")
    return vectorizer

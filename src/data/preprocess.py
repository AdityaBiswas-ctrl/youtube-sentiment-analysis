"""
Preprocess YouTube comments for model training.
DVC Stage 2: Data Preprocessing

Cleans text, filters short comments, and creates train/test splits.
"""

import os
import re
import logging

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_config():
    """Load application config from configs/config.yaml."""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """
    Clean a comment string:
    - Remove URLs
    - Remove HTML tags
    - Remove special characters (keep basic punctuation)
    - Normalize whitespace
    - Lowercase
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol but keep text
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    text = text.lower()

    return text


def main():
    """Main entry point for the preprocess DVC stage."""
    params = load_params()
    config = load_config()

    preprocess_params = params["preprocess"]
    test_size = preprocess_params["test_size"]
    min_length = preprocess_params["min_length"]
    random_state = preprocess_params["random_state"]

    # Load raw data
    raw_path = config["paths"]["raw_data"]
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} comments")

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Filter short comments
    df = df[df["clean_text"].str.len() >= min_length].copy()
    logger.info(f"After filtering (min_length={min_length}): {len(df)} comments")

    # Remove duplicates based on clean_text
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    logger.info(f"After deduplication: {len(df)} comments")

    # Use vader_label as the target
    df["label"] = df["vader_label"]

    # Select relevant columns
    df = df[["clean_text", "label", "vader_compound", "author", "video_id"]].copy()

    # Check label distribution
    label_dist = df["label"].value_counts()
    logger.info(f"Label distribution:\n{label_dist}")

    # Train/test split with stratification
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    logger.info(f"Train set: {len(train_df)} | Test set: {len(test_df)}")

    # Save processed data
    train_path = config["paths"]["train_data"]
    test_path = config["paths"]["test_data"]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info(f"Saved train data to {train_path}")
    logger.info(f"Saved test data to {test_path}")


if __name__ == "__main__":
    main()

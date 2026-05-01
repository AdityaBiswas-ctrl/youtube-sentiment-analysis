"""
Flask API server for YouTube Sentiment Analysis.
Serves the dashboard and provides real-time analysis endpoints.
"""

import os
import re
import json
import logging

import yaml
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(
    __name__,
    static_folder="../../static",
    static_url_path="/static",
)
CORS(app)

# Global state
vader_analyzer = SentimentIntensityAnalyzer()
ml_model = None
vectorizer = None
label_encoder = None
model_metrics = None


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_ml_artifacts():
    """Load trained model, vectorizer, and label encoder."""
    global ml_model, vectorizer, label_encoder, model_metrics

    config = load_config()

    model_path = config["paths"]["model"]
    vectorizer_path = config["paths"]["vectorizer"]
    le_path = config["paths"]["label_encoder"]
    metrics_path = config["paths"]["metrics"]

    try:
        ml_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(le_path)
        logger.info("Loaded ML model artifacts successfully")
    except FileNotFoundError:
        logger.warning(
            "ML model artifacts not found. "
            "Run 'dvc repro' to train the model first. "
            "VADER-only mode active."
        )

    try:
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)
        logger.info("Loaded model metrics")
    except FileNotFoundError:
        logger.warning("Model metrics not found")


def extract_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def clean_text(text: str) -> str:
    """Clean comment text for ML prediction."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def analyze_comment(text: str) -> dict:
    """Analyze a single comment with VADER and optionally ML model."""
    # VADER analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    compound = vader_scores["compound"]

    if compound >= 0.05:
        vader_label = "positive"
    elif compound <= -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"

    result = {
        "vader": {
            "label": vader_label,
            "compound": round(compound, 4),
            "pos": round(vader_scores["pos"], 4),
            "neu": round(vader_scores["neu"], 4),
            "neg": round(vader_scores["neg"], 4),
        }
    }

    # ML model prediction
    if ml_model is not None and vectorizer is not None:
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        prediction = ml_model.predict(features)[0]
        ml_label = label_encoder.inverse_transform([prediction])[0]

        # Get prediction probabilities if available
        try:
            probas = ml_model.predict_proba(features)[0]
            proba_dict = {
                label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(probas)
            }
        except AttributeError:
            proba_dict = {}

        result["ml_model"] = {
            "label": ml_label,
            "probabilities": proba_dict,
        }

    return result


@app.route("/")
def index():
    """Serve the dashboard."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    Analyze YouTube video comments.
    Expects JSON: { video_url, api_key, max_comments }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    video_url = data.get("video_url", "")
    api_key = data.get("api_key", "") or os.environ.get("YOUTUBE_API_KEY", "")
    max_comments = min(int(data.get("max_comments", 200)), 500)

    if not video_url:
        return jsonify({"error": "video_url is required"}), 400
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        # Build YouTube client
        youtube = build("youtube", "v3", developerKey=api_key)

        # Get video info
        video_response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id,
        ).execute()

        if not video_response.get("items"):
            return jsonify({"error": "Video not found"}), 404

        video_info = video_response["items"][0]
        video_data = {
            "title": video_info["snippet"]["title"],
            "channel": video_info["snippet"]["channelTitle"],
            "thumbnail": video_info["snippet"]["thumbnails"]
            .get("high", {})
            .get("url", ""),
            "view_count": int(
                video_info["statistics"].get("viewCount", 0)
            ),
            "like_count": int(
                video_info["statistics"].get("likeCount", 0)
            ),
            "comment_count": int(
                video_info["statistics"].get("commentCount", 0)
            ),
        }

        # Fetch comments
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            comment_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
            ).execute()

            for item in comment_response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = snippet.get("textDisplay", "")

                # Analyze sentiment
                sentiment = analyze_comment(comment_text)

                comments.append({
                    "text": comment_text,
                    "author": snippet.get("authorDisplayName", "Unknown"),
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                    "sentiment": sentiment,
                })

            next_page_token = comment_response.get("nextPageToken")
            if not next_page_token:
                break

        # Compute aggregate statistics
        vader_labels = [c["sentiment"]["vader"]["label"] for c in comments]
        vader_compounds = [
            c["sentiment"]["vader"]["compound"] for c in comments
        ]

        stats = {
            "total_comments": len(comments),
            "vader": {
                "positive": vader_labels.count("positive"),
                "neutral": vader_labels.count("neutral"),
                "negative": vader_labels.count("negative"),
                "avg_compound": round(
                    sum(vader_compounds) / max(len(vader_compounds), 1), 4
                ),
            },
        }

        # ML model stats
        if ml_model is not None:
            ml_labels = [
                c["sentiment"].get("ml_model", {}).get("label", "")
                for c in comments
            ]
            stats["ml_model"] = {
                "positive": ml_labels.count("positive"),
                "neutral": ml_labels.count("neutral"),
                "negative": ml_labels.count("negative"),
            }

        # Sort: top positive and negative
        sorted_by_compound = sorted(
            comments, key=lambda c: c["sentiment"]["vader"]["compound"]
        )
        top_negative = sorted_by_compound[:3]
        top_positive = sorted_by_compound[-3:][::-1]

        # Score distribution histogram data
        histogram = {"bins": [], "counts": []}
        bin_edges = [
            -1.0, -0.8, -0.6, -0.4, -0.2,
            0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ]
        for i in range(len(bin_edges) - 1):
            low, high = bin_edges[i], bin_edges[i + 1]
            count = sum(
                1 for c in vader_compounds if low <= c < high
            )
            histogram["bins"].append(f"{low:.1f}")
            histogram["counts"].append(count)

        return jsonify({
            "video": video_data,
            "comments": comments,
            "stats": stats,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "histogram": histogram,
            "model_metrics": model_metrics,
            "ml_model_available": ml_model is not None,
        })

    except HttpError as e:
        error_detail = str(e)
        if "commentsDisabled" in error_detail:
            return jsonify(
                {"error": "Comments are disabled for this video"}
            ), 400
        elif "403" in error_detail:
            return jsonify(
                {"error": "API quota exceeded or invalid API key"}
            ), 403
        return jsonify({"error": f"YouTube API error: {error_detail}"}), 500
    except Exception as e:
        logger.exception("Error analyzing video")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Return current model info and metrics."""
    return jsonify({
        "ml_model_available": ml_model is not None,
        "metrics": model_metrics,
        "model_type": (
            model_metrics.get("model_type", "unknown")
            if model_metrics else None
        ),
    })


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "ml_model_loaded": ml_model is not None,
    })


# Initialize on startup
load_ml_artifacts()


if __name__ == "__main__":
    config = load_config()
    port = int(os.environ.get("PORT", config["api"]["port"]))
    app.run(
        host=config["api"]["host"],
        port=port,
        debug=config["api"]["debug"],
    )

"""
Fetch YouTube comments using the YouTube Data API v3.
DVC Stage 1: Data Ingestion

Reads video IDs and limits from params.yaml, fetches comments,
and generates pseudo-labels using VADER sentiment analysis.
"""

import os
import sys
import csv
import re
import logging
from datetime import datetime

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def load_config():
    """Load application config from configs/config.yaml."""
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from a YouTube URL or return as-is if already an ID."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def get_vader_label(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple:
    """
    Classify text sentiment using VADER.
    Returns (label, compound_score).
    """
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound


def fetch_comments_for_video(
    youtube, video_id: str, max_comments: int = 500
) -> list:
    """
    Fetch comments for a single video using pagination.
    Returns list of comment dicts.
    """
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
            )
            response = request.execute()
        except HttpError as e:
            logger.warning(f"HTTP error fetching comments for {video_id}: {e}")
            break
        except Exception as e:
            logger.warning(f"Error fetching comments for {video_id}: {e}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "author": snippet.get("authorDisplayName", "Unknown"),
                "text": snippet.get("textDisplay", ""),
                "published_at": snippet.get("publishedAt", ""),
                "like_count": snippet.get("likeCount", 0),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    logger.info(f"Fetched {len(comments)} comments for video {video_id}")
    return comments


def main():
    """Main entry point for the fetch_data DVC stage."""
    params = load_params()
    config = load_config()

    video_ids = params["data"]["video_ids"]
    max_comments = params["data"]["max_comments"]

    # Get API key from environment
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        logger.error(
            "YOUTUBE_API_KEY environment variable not set. "
            "Set it with: export YOUTUBE_API_KEY=your_key"
        )
        sys.exit(1)

    # Build YouTube API client
    youtube = build(
        config["youtube"]["api_service_name"],
        config["youtube"]["api_version"],
        developerKey=api_key,
    )

    # Initialize VADER for pseudo-labeling
    analyzer = SentimentIntensityAnalyzer()

    # Fetch comments from all videos
    all_comments = []
    for vid in video_ids:
        video_id = extract_video_id(vid)
        comments = fetch_comments_for_video(
            youtube, video_id, max_comments=max_comments
        )
        all_comments.extend(comments)

    if not all_comments:
        logger.error("No comments fetched. Check video IDs and API key.")
        sys.exit(1)

    # Add VADER pseudo-labels
    for comment in all_comments:
        label, compound = get_vader_label(comment["text"], analyzer)
        comment["vader_label"] = label
        comment["vader_compound"] = round(compound, 4)

    # Save to CSV
    output_path = config["paths"]["raw_data"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "video_id", "comment_id", "author", "text",
        "published_at", "like_count", "vader_label", "vader_compound"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comments)

    logger.info(
        f"Saved {len(all_comments)} comments to {output_path}"
    )

    # Print label distribution
    from collections import Counter
    dist = Counter(c["vader_label"] for c in all_comments)
    logger.info(f"Label distribution: {dict(dist)}")


if __name__ == "__main__":
    main()

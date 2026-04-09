"""
Unit tests for the preprocessing module.
"""

import pytest
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import clean_text


class TestCleanText:
    """Tests for the clean_text function."""

    def test_removes_urls(self):
        text = "Check this out http://example.com and https://youtu.be/abc123"
        result = clean_text(text)
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_html_tags(self):
        text = "<b>Bold</b> and <a href='#'>link</a> text"
        result = clean_text(text)
        assert "<b>" not in result
        assert "<a" not in result
        assert "bold" in result
        assert "text" in result

    def test_removes_email_addresses(self):
        text = "Email me at user@example.com please"
        result = clean_text(text)
        assert "@" not in result
        assert "example.com" not in result

    def test_removes_mentions(self):
        text = "@user123 This is great!"
        result = clean_text(text)
        assert "@user123" not in result
        assert "great" in result

    def test_lowercases(self):
        text = "THIS IS UPPERCASE TEXT"
        result = clean_text(text)
        assert result == "this is uppercase text"

    def test_normalizes_whitespace(self):
        text = "  too   many    spaces   "
        result = clean_text(text)
        assert result == "too many spaces"

    def test_keeps_basic_punctuation(self):
        text = "Hello! How are you? I'm fine, thanks."
        result = clean_text(text)
        assert "!" in result
        assert "?" in result
        assert "," in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_handles_non_string(self):
        assert clean_text(12345) == ""

    def test_removes_hashtag_symbol_keeps_text(self):
        text = "I love #Python and #coding"
        result = clean_text(text)
        assert "python" in result
        assert "coding" in result
        assert "#" not in result

    def test_complex_comment(self):
        text = (
            '<a href="#">@user</a> Check http://link.com #amazing '
            'video!! user@email.com   Great  content 🔥🔥'
        )
        result = clean_text(text)
        # Should have cleaned text without URLs, emails, HTML, mentions
        assert "http" not in result
        assert "@" not in result
        assert "<a" not in result
        assert "amazing" in result
        assert "great" in result
        assert "content" in result


class TestLabelAssignment:
    """Tests for VADER label thresholds."""

    def test_positive_threshold(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("This is absolutely fantastic and wonderful!")
        assert scores["compound"] >= 0.05

    def test_negative_threshold(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("This is terrible, awful, and disgusting.")
        assert scores["compound"] <= -0.05

    def test_neutral_range(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("The meeting is at 3pm.")
        assert -0.05 < scores["compound"] < 0.05

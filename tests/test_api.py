"""
Tests for the Flask API endpoints.
"""

# pyrefly: ignore [missing-import]
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Tests for the /api/health endpoint."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "ml_model_loaded" in data


class TestModelInfoEndpoint:
    """Tests for the /api/model-info endpoint."""

    def test_model_info(self, client):
        response = client.get("/api/model-info")
        assert response.status_code == 200
        data = response.get_json()
        assert "ml_model_available" in data
        assert "metrics" in data


class TestAnalyzeEndpoint:
    """Tests for the /api/analyze endpoint."""

    def test_missing_json(self, client):
        response = client.post("/api/analyze")
        assert response.status_code == 400

    def test_missing_video_url(self, client):
        response = client.post(
            "/api/analyze",
            data=json.dumps({"api_key": "test"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_missing_api_key(self, client):
        response = client.post(
            "/api/analyze",
            data=json.dumps({"video_url": "https://youtube.com/watch?v=test123"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_invalid_url(self, client):
        response = client.post(
            "/api/analyze",
            data=json.dumps({
                "video_url": "not-a-url",
                "api_key": "test-key",
            }),
            content_type="application/json",
        )
        # Should return 400 for invalid URL or 500 for API error
        assert response.status_code in [400, 500]


class TestDashboard:
    """Tests for the dashboard page."""

    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200

"""
Tests for Flask application routes.
"""

import pytest
import io


class TestIndexRoute:
    """Tests for index (upload) page."""

    def test_index_returns_200(self, client):
        """Index page should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_contains_upload_form(self, client):
        """Index page should contain upload form."""
        response = client.get("/")
        assert b"form" in response.data.lower() or b"upload" in response.data.lower()


class TestHealthRoute:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Health endpoint should return JSON."""
        response = client.get("/health")
        assert response.content_type == "application/json"

    def test_health_contains_status(self, client):
        """Health response should contain status field."""
        response = client.get("/health")
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_camera_status(self, client):
        """Health response should indicate camera status."""
        response = client.get("/health")
        data = response.get_json()
        assert "camera_active" in data


class TestStatusRoute:
    """Tests for status endpoint."""

    def test_status_returns_200(self, client):
        """Status endpoint should return 200 OK."""
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_returns_json(self, client):
        """Status endpoint should return JSON."""
        response = client.get("/status")
        assert response.content_type == "application/json"

    def test_status_structure_no_video(self, client):
        """Status should have correct structure when no video loaded."""
        response = client.get("/status")
        data = response.get_json()

        assert "current_frame" in data
        assert "total_frames" in data
        assert "fps" in data
        assert "is_playing" in data
        assert "players" in data


class TestUploadRoute:
    """Tests for video upload endpoint."""

    def test_upload_no_file_returns_400(self, client):
        """Upload without file should return 400."""
        response = client.post("/upload")
        assert response.status_code == 400

    def test_upload_empty_filename_returns_400(self, client):
        """Upload with empty filename should return 400."""
        data = {"video": (io.BytesIO(b""), "")}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_upload_invalid_extension_returns_400(self, client):
        """Upload with invalid extension should return 400."""
        data = {"video": (io.BytesIO(b"fake content"), "test.txt")}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 400


class TestDashboardRoute:
    """Tests for dashboard page."""

    def test_dashboard_redirects_without_video(self, client):
        """Dashboard should redirect to index if no video loaded."""
        response = client.get("/dashboard")
        assert response.status_code == 302  # Redirect


class TestVideoFeedRoute:
    """Tests for video feed endpoint."""

    def test_video_feed_returns_404_without_video(self, client):
        """Video feed should return 404 if no video loaded."""
        response = client.get("/video_feed")
        assert response.status_code == 404


class TestResetRoute:
    """Tests for reset endpoint."""

    def test_reset_returns_200(self, client):
        """Reset endpoint should return 200 OK."""
        response = client.post("/reset")
        assert response.status_code == 200

    def test_reset_returns_success(self, client):
        """Reset should return success status."""
        response = client.post("/reset")
        data = response.get_json()
        assert data.get("success") is True


class TestTogglePlayRoute:
    """Tests for toggle play endpoint."""

    def test_toggle_play_without_video_returns_404(self, client):
        """Toggle play without video should return 404."""
        response = client.post("/toggle_play")
        assert response.status_code == 404

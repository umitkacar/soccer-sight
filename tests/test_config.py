"""
Tests for configuration module.
"""

import os
import pytest
from pathlib import Path


class TestFlaskConfig:
    """Tests for Flask configuration."""

    def test_secret_key_exists(self):
        """SECRET_KEY should exist and not be empty."""
        from config import FlaskConfig

        assert FlaskConfig.SECRET_KEY is not None
        assert len(FlaskConfig.SECRET_KEY) > 0

    def test_secret_key_not_hardcoded(self):
        """SECRET_KEY should not be the old hardcoded value."""
        from config import FlaskConfig

        hardcoded_key = "futbl-tracking-secret-key"
        assert FlaskConfig.SECRET_KEY != hardcoded_key

    def test_secret_key_length(self):
        """SECRET_KEY should be sufficiently long (>32 chars)."""
        from config import FlaskConfig

        assert len(FlaskConfig.SECRET_KEY) >= 32

    def test_upload_folder_path(self):
        """Upload folder should be a valid path."""
        from config import FlaskConfig

        assert FlaskConfig.UPLOAD_FOLDER is not None
        assert "uploads" in FlaskConfig.UPLOAD_FOLDER

    def test_max_content_length(self):
        """MAX_CONTENT_LENGTH should be reasonable."""
        from config import FlaskConfig

        # Should be at least 10MB
        assert FlaskConfig.MAX_CONTENT_LENGTH >= 10 * 1024 * 1024
        # Should not exceed 1GB
        assert FlaskConfig.MAX_CONTENT_LENGTH <= 1024 * 1024 * 1024


class TestAllowedExtensions:
    """Tests for allowed file extensions."""

    def test_allowed_extensions_exist(self):
        """ALLOWED_EXTENSIONS should be defined."""
        from config import ALLOWED_EXTENSIONS

        assert ALLOWED_EXTENSIONS is not None
        assert len(ALLOWED_EXTENSIONS) > 0

    def test_common_video_formats_allowed(self):
        """Common video formats should be allowed."""
        from config import ALLOWED_EXTENSIONS

        expected = {"mp4", "avi", "mov"}
        assert expected.issubset(ALLOWED_EXTENSIONS)

    def test_dangerous_extensions_not_allowed(self):
        """Dangerous file extensions should not be allowed."""
        from config import ALLOWED_EXTENSIONS

        dangerous = {"exe", "sh", "py", "js", "html", "php"}
        assert dangerous.isdisjoint(ALLOWED_EXTENSIONS)


class TestDetectionConfig:
    """Tests for YOLO detection configuration."""

    def test_inference_size_valid(self):
        """Inference size should be a valid YOLO input size."""
        from config import DetectionConfig

        valid_sizes = {320, 416, 512, 640, 1280}
        assert DetectionConfig.INFERENCE_SIZE in valid_sizes

    def test_frame_skip_reasonable(self):
        """Frame skip should be reasonable (1-10)."""
        from config import DetectionConfig

        assert 1 <= DetectionConfig.FRAME_SKIP <= 10


class TestTrackingConfig:
    """Tests for tracking configuration."""

    def test_max_players_valid(self):
        """MAX_PLAYERS should be reasonable for football."""
        from config import TrackingConfig

        # Football has 11 players per team, max could track subset
        assert 1 <= TrackingConfig.MAX_PLAYERS <= 22

    def test_lost_track_timeout_valid(self):
        """LOST_TRACK_TIMEOUT should be reasonable."""
        from config import TrackingConfig

        # Should be between 10 frames and 5 seconds worth
        assert 10 <= TrackingConfig.LOST_TRACK_TIMEOUT <= 150


class TestOCRConfig:
    """Tests for OCR configuration."""

    def test_confidence_threshold_valid(self):
        """OCR confidence threshold should be between 0 and 1."""
        from config import OCRConfig

        assert 0.0 <= OCRConfig.CONFIDENCE_THRESHOLD <= 1.0

    def test_lock_threshold_valid(self):
        """Lock threshold should be reasonable."""
        from config import OCRConfig

        assert 1 <= OCRConfig.LOCK_THRESHOLD <= 10

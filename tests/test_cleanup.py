"""
Tests for cleanup utilities.
"""

import pytest
import time
from pathlib import Path


class TestCleanupFunctions:
    """Tests for cleanup utility functions."""

    def test_get_file_age_hours(self, temp_upload_dir):
        """get_file_age_hours should return correct age."""
        from cleanup import get_file_age_hours

        # Create a test file
        test_file = temp_upload_dir / "test.txt"
        test_file.write_text("test content")

        # File just created should be < 1 hour old
        age = get_file_age_hours(test_file)
        assert age < 1.0

    def test_cleanup_old_uploads_empty_dir(self, temp_upload_dir):
        """cleanup_old_uploads should handle empty directory."""
        from cleanup import cleanup_old_uploads
        import config

        # Temporarily override upload folder
        original_folder = config.UPLOAD_FOLDER
        config.UPLOAD_FOLDER = temp_upload_dir

        try:
            removed = cleanup_old_uploads(max_age_hours=0)
            assert removed == 0
        finally:
            config.UPLOAD_FOLDER = original_folder

    def test_get_upload_folder_size_mb(self, temp_upload_dir):
        """get_upload_folder_size_mb should return folder size."""
        from cleanup import get_upload_folder_size_mb
        import config

        # Create test files
        test_file = temp_upload_dir / "test.bin"
        test_file.write_bytes(b"x" * 1024 * 1024)  # 1MB file

        # Temporarily override upload folder
        original_folder = config.UPLOAD_FOLDER
        config.UPLOAD_FOLDER = temp_upload_dir

        try:
            size = get_upload_folder_size_mb()
            assert size >= 0.9  # Should be ~1MB
            assert size <= 1.1
        finally:
            config.UPLOAD_FOLDER = original_folder


class TestAutoCleanup:
    """Tests for auto cleanup functionality."""

    def test_run_auto_cleanup_no_error(self, temp_upload_dir):
        """run_auto_cleanup should not raise errors."""
        from cleanup import run_auto_cleanup
        import config

        # Temporarily override upload folder
        original_folder = config.UPLOAD_FOLDER
        config.UPLOAD_FOLDER = temp_upload_dir

        try:
            # Should not raise any exception
            run_auto_cleanup()
        finally:
            config.UPLOAD_FOLDER = original_folder

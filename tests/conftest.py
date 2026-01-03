"""
Pytest configuration and fixtures for Football Player Tracking tests.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def app():
    """Create application instance for testing."""
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    flask_app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()

    yield flask_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def temp_upload_dir():
    """Create a temporary upload directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_video_path():
    """Return path to sample video if exists."""
    video_path = PROJECT_ROOT / "videos" / "test_video.mp4"
    if video_path.exists():
        return str(video_path)
    return None

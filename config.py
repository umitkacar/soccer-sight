"""
Configuration module for Football Player Tracking Application.
Centralizes all constants and environment-based settings.
"""

import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
VIDEO_FOLDER = BASE_DIR / "videos"
OCR_DEBUG_FOLDER = BASE_DIR / "ocr_debug"

# Flask configuration
class FlaskConfig:
    """Flask application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE_MB", 500)) * 1024 * 1024
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", 5000))


# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


# YOLO/Detection configuration
class DetectionConfig:
    """YOLO detection settings."""
    MODEL_PATH = os.getenv("YOLO_MODEL", "yolo11l.pt")
    INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", 640))
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", 2))
    CONFIDENCE_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", 0.5))


# Tracking configuration
class TrackingConfig:
    """ByteTrack tracking settings."""
    MAX_PLAYERS = int(os.getenv("MAX_PLAYERS", 8))
    LOST_TRACK_TIMEOUT = int(os.getenv("LOST_TRACK_TIMEOUT", 35))


# OCR configuration
class OCRConfig:
    """Jersey number OCR settings."""
    HUNTING_OCR_INTERVAL = int(os.getenv("HUNTING_OCR_INTERVAL", 2))
    VERIFICATION_INTERVAL = int(os.getenv("VERIFICATION_INTERVAL", 50))
    LOCK_THRESHOLD = int(os.getenv("LOCK_THRESHOLD", 3))
    CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE", 0.5))
    DEBUG_SAVE = os.getenv("OCR_DEBUG_SAVE", "true").lower() == "true"


# Team detection (HSV color ranges)
class TeamConfig:
    """Team classification via HSV color analysis."""
    GREEN_HSV_LOWER = (35, 40, 40)
    GREEN_HSV_UPPER = (85, 255, 255)
    # Red team HSV range
    RED_HSV_LOWER = (0, 100, 100)
    RED_HSV_UPPER = (10, 255, 255)
    # Turquoise team HSV range
    TURQUOISE_HSV_LOWER = (80, 100, 100)
    TURQUOISE_HSV_UPPER = (100, 255, 255)


# Logging configuration
class LogConfig:
    """Logging settings."""
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    FILE = os.getenv("LOG_FILE", None)  # None = console only


# Cleanup configuration
class CleanupConfig:
    """File cleanup settings."""
    UPLOAD_MAX_AGE_HOURS = int(os.getenv("UPLOAD_MAX_AGE_HOURS", 24))
    AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "true").lower() == "true"

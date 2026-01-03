"""
File cleanup utilities for Football Player Tracking Application.
Handles cleanup of old uploaded files and temporary data.
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from config import CleanupConfig, UPLOAD_FOLDER, OCR_DEBUG_FOLDER
from logger import get_logger

logger = get_logger("cleanup")


def get_file_age_hours(filepath: Path) -> float:
    """
    Get the age of a file in hours.

    Args:
        filepath: Path to the file

    Returns:
        Age in hours
    """
    mtime = filepath.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / 3600


def cleanup_old_uploads(max_age_hours: int = None) -> int:
    """
    Remove uploaded files older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours (default from config)

    Returns:
        Number of files removed
    """
    if max_age_hours is None:
        max_age_hours = CleanupConfig.UPLOAD_MAX_AGE_HOURS

    removed_count = 0
    upload_path = Path(UPLOAD_FOLDER)

    if not upload_path.exists():
        logger.debug("Upload folder does not exist, nothing to clean")
        return 0

    for filepath in upload_path.iterdir():
        if filepath.is_file():
            age = get_file_age_hours(filepath)
            if age > max_age_hours:
                try:
                    filepath.unlink()
                    removed_count += 1
                    logger.info(f"Removed old upload: {filepath.name} (age: {age:.1f}h)")
                except Exception as e:
                    logger.error(f"Failed to remove {filepath}: {e}")

    if removed_count > 0:
        logger.info(f"Cleanup complete: removed {removed_count} old files")

    return removed_count


def cleanup_ocr_debug(max_files: int = 1000) -> int:
    """
    Remove old OCR debug images, keeping only the most recent.

    Args:
        max_files: Maximum number of debug files to keep

    Returns:
        Number of files removed
    """
    removed_count = 0
    debug_path = Path(OCR_DEBUG_FOLDER)

    if not debug_path.exists():
        return 0

    # Get all debug files sorted by modification time (oldest first)
    files = sorted(debug_path.glob("*.jpg"), key=lambda p: p.stat().st_mtime)

    # Remove oldest files if over limit
    files_to_remove = len(files) - max_files
    if files_to_remove > 0:
        for filepath in files[:files_to_remove]:
            try:
                filepath.unlink()
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove debug file {filepath}: {e}")

        logger.info(f"OCR debug cleanup: removed {removed_count} old files")

    return removed_count


def get_upload_folder_size_mb() -> float:
    """
    Get the total size of the upload folder in MB.

    Returns:
        Size in megabytes
    """
    upload_path = Path(UPLOAD_FOLDER)
    if not upload_path.exists():
        return 0.0

    total_size = sum(f.stat().st_size for f in upload_path.iterdir() if f.is_file())
    return total_size / (1024 * 1024)


def run_auto_cleanup():
    """
    Run automatic cleanup if enabled in config.
    Called on application startup and periodically.
    """
    if not CleanupConfig.AUTO_CLEANUP:
        logger.debug("Auto cleanup is disabled")
        return

    logger.info("Running automatic cleanup...")

    # Clean old uploads
    uploads_removed = cleanup_old_uploads()

    # Clean OCR debug folder
    debug_removed = cleanup_ocr_debug()

    # Report upload folder size
    upload_size = get_upload_folder_size_mb()
    logger.info(f"Upload folder size: {upload_size:.1f} MB")


if __name__ == "__main__":
    # Run cleanup directly
    run_auto_cleanup()

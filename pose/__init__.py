"""
Pose Estimation Module for Football Player Tracking.

Provides pose-guided ROI extraction for better jersey number detection.
Uses YOLO11-pose for keypoint detection.

Usage:
    from pose import PoseEstimator, ROIExtractor

    # Initialize pose estimator
    estimator = PoseEstimator()

    # Extract keypoints from player crop
    keypoints = estimator.estimate(player_crop)

    # Get jersey ROI from keypoints
    extractor = ROIExtractor()
    jersey_roi = extractor.extract_jersey_roi(image, keypoints)
"""

from .pose_estimator import PoseEstimator, Keypoint, PoseResult
from .roi_extractor import ROIExtractor

__all__ = [
    'PoseEstimator',
    'ROIExtractor',
    'Keypoint',
    'PoseResult',
]

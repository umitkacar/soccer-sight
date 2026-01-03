"""
ROI Extractor using Pose Keypoints.

Extracts accurate jersey/torso regions using body keypoints
instead of fixed percentage crops.

Benefits over fixed-crop:
- Handles rotated/tilted players
- Accurate torso localization
- Better OCR input quality
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .pose_estimator import PoseResult, Keypoint, KeypointIndex


@dataclass
class ROIResult:
    """Result of ROI extraction."""
    roi: np.ndarray  # Extracted region
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original image
    method: str  # 'pose', 'fallback', 'none'
    confidence: float  # Confidence in extraction quality


class ROIExtractor:
    """
    Extract jersey/torso ROI using pose keypoints.

    Uses shoulder and hip keypoints to define accurate torso region.
    Falls back to percentage-based crop if keypoints unavailable.
    """

    def __init__(
        self,
        padding_ratio: float = 0.1,
        jersey_top_ratio: float = 0.0,  # Start at shoulders
        jersey_bottom_ratio: float = 0.6,  # 60% down to hips (jersey area)
        min_roi_size: int = 20
    ):
        """
        Initialize ROI extractor.

        Args:
            padding_ratio: Extra padding around detected torso
            jersey_top_ratio: How far up from shoulders (0 = at shoulders)
            jersey_bottom_ratio: How far down from shoulders to hips
            min_roi_size: Minimum ROI dimension
        """
        self.padding_ratio = padding_ratio
        self.jersey_top_ratio = jersey_top_ratio
        self.jersey_bottom_ratio = jersey_bottom_ratio
        self.min_roi_size = min_roi_size

    def extract_jersey_roi(
        self,
        image: np.ndarray,
        pose: Optional[PoseResult] = None
    ) -> ROIResult:
        """
        Extract jersey ROI from image using pose keypoints.

        Args:
            image: BGR player crop image
            pose: PoseResult with keypoints (optional)

        Returns:
            ROIResult with extracted jersey region
        """
        if image is None or image.size == 0:
            return ROIResult(
                roi=np.array([]),
                bbox=(0, 0, 0, 0),
                method='none',
                confidence=0.0
            )

        h, w = image.shape[:2]

        # Try pose-based extraction first
        if pose is not None and pose.has_torso_keypoints():
            roi_result = self._extract_from_pose(image, pose)
            if roi_result.roi.size > 0:
                return roi_result

        # Fallback to fixed percentage crop
        return self._extract_fallback(image)

    def _extract_from_pose(
        self,
        image: np.ndarray,
        pose: PoseResult
    ) -> ROIResult:
        """Extract ROI using pose keypoints."""
        h, w = image.shape[:2]

        # Get shoulder keypoints
        left_shoulder = pose.left_shoulder
        right_shoulder = pose.right_shoulder

        # Get hip keypoints
        left_hip = pose.left_hip
        right_hip = pose.right_hip

        # Calculate torso boundaries
        # X boundaries from shoulders (or hips if shoulders missing)
        x_points = []
        if left_shoulder:
            x_points.append(left_shoulder.x)
        if right_shoulder:
            x_points.append(right_shoulder.x)
        if not x_points:
            if left_hip:
                x_points.append(left_hip.x)
            if right_hip:
                x_points.append(right_hip.x)

        if not x_points:
            return ROIResult(
                roi=np.array([]),
                bbox=(0, 0, 0, 0),
                method='pose_failed',
                confidence=0.0
            )

        x_min = min(x_points)
        x_max = max(x_points)

        # Y boundaries
        y_points_top = []
        y_points_bottom = []

        if left_shoulder:
            y_points_top.append(left_shoulder.y)
        if right_shoulder:
            y_points_top.append(right_shoulder.y)

        if left_hip:
            y_points_bottom.append(left_hip.y)
        if right_hip:
            y_points_bottom.append(right_hip.y)

        # Estimate if missing
        if not y_points_top and y_points_bottom:
            # Estimate shoulders from hips (approx 40% up from hips)
            y_top = min(y_points_bottom) - 0.4 * h
        elif y_points_top:
            y_top = min(y_points_top)
        else:
            y_top = 0.15 * h

        if not y_points_bottom and y_points_top:
            # Estimate hips from shoulders (approx 35% down)
            y_bottom = max(y_points_top) + 0.35 * h
        elif y_points_bottom:
            y_bottom = max(y_points_bottom)
        else:
            y_bottom = 0.5 * h

        # Apply jersey ratios (top part of torso)
        torso_height = y_bottom - y_top
        jersey_top = y_top - (torso_height * self.jersey_top_ratio)
        jersey_bottom = y_top + (torso_height * self.jersey_bottom_ratio)

        # Calculate width (add some padding for arms)
        torso_width = x_max - x_min
        padding_x = torso_width * self.padding_ratio
        padding_y = torso_height * self.padding_ratio

        # Final bbox with padding
        x1 = int(max(0, x_min - padding_x))
        y1 = int(max(0, jersey_top - padding_y))
        x2 = int(min(w, x_max + padding_x))
        y2 = int(min(h, jersey_bottom + padding_y))

        # Validate size
        if (x2 - x1) < self.min_roi_size or (y2 - y1) < self.min_roi_size:
            return ROIResult(
                roi=np.array([]),
                bbox=(x1, y1, x2, y2),
                method='pose_too_small',
                confidence=0.0
            )

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        # Calculate confidence based on keypoint availability
        kp_count = sum([
            1 if pose.left_shoulder else 0,
            1 if pose.right_shoulder else 0,
            1 if pose.left_hip else 0,
            1 if pose.right_hip else 0
        ])
        confidence = kp_count / 4.0

        return ROIResult(
            roi=roi,
            bbox=(x1, y1, x2, y2),
            method='pose',
            confidence=confidence
        )

    def _extract_fallback(self, image: np.ndarray) -> ROIResult:
        """Fallback fixed-percentage extraction."""
        h, w = image.shape[:2]

        # Same as existing extract_jersey_region logic
        top = int(h * 0.15)
        bottom = int(h * 0.50)
        left = int(w * 0.10)
        right = int(w * 0.90)

        roi = image[top:bottom, left:right]

        if roi.size == 0:
            return ROIResult(
                roi=image,
                bbox=(0, 0, w, h),
                method='fallback_full',
                confidence=0.3
            )

        return ROIResult(
            roi=roi,
            bbox=(left, top, right, bottom),
            method='fallback',
            confidence=0.5
        )

    def extract_torso_roi(
        self,
        image: np.ndarray,
        pose: Optional[PoseResult] = None
    ) -> ROIResult:
        """
        Extract full torso ROI (shoulders to hips).

        Useful for team classification where full jersey is needed.

        Args:
            image: BGR player crop image
            pose: PoseResult with keypoints

        Returns:
            ROIResult with full torso region
        """
        if image is None or image.size == 0:
            return ROIResult(
                roi=np.array([]),
                bbox=(0, 0, 0, 0),
                method='none',
                confidence=0.0
            )

        # For full torso, use jersey_bottom_ratio = 1.0
        original_ratio = self.jersey_bottom_ratio
        self.jersey_bottom_ratio = 1.0

        result = self.extract_jersey_roi(image, pose)

        self.jersey_bottom_ratio = original_ratio
        return result

    def extract_number_region(
        self,
        image: np.ndarray,
        pose: Optional[PoseResult] = None
    ) -> ROIResult:
        """
        Extract jersey number region (upper back/chest area).

        More focused region where jersey numbers typically appear.

        Args:
            image: BGR player crop image
            pose: PoseResult with keypoints

        Returns:
            ROIResult optimized for number recognition
        """
        if image is None or image.size == 0:
            return ROIResult(
                roi=np.array([]),
                bbox=(0, 0, 0, 0),
                method='none',
                confidence=0.0
            )

        # For number region, use tighter bounds
        original_top = self.jersey_top_ratio
        original_bottom = self.jersey_bottom_ratio

        self.jersey_top_ratio = 0.05  # Slightly above shoulders
        self.jersey_bottom_ratio = 0.5  # Middle of torso

        result = self.extract_jersey_roi(image, pose)

        self.jersey_top_ratio = original_top
        self.jersey_bottom_ratio = original_bottom

        return result


def create_roi_extractor(
    mode: str = 'jersey',
    **kwargs
) -> ROIExtractor:
    """
    Factory function to create ROI extractor with preset configurations.

    Args:
        mode: 'jersey' (default), 'torso', or 'number'
        **kwargs: Additional configuration

    Returns:
        Configured ROIExtractor
    """
    if mode == 'jersey':
        return ROIExtractor(
            padding_ratio=0.1,
            jersey_top_ratio=0.0,
            jersey_bottom_ratio=0.6,
            **kwargs
        )
    elif mode == 'torso':
        return ROIExtractor(
            padding_ratio=0.15,
            jersey_top_ratio=0.0,
            jersey_bottom_ratio=1.0,
            **kwargs
        )
    elif mode == 'number':
        return ROIExtractor(
            padding_ratio=0.05,
            jersey_top_ratio=0.05,
            jersey_bottom_ratio=0.5,
            **kwargs
        )
    else:
        return ROIExtractor(**kwargs)

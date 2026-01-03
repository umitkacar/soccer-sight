"""
Pose Estimator using YOLO11-pose.

Detects human body keypoints for accurate jersey region extraction.

COCO Keypoint Format (17 keypoints):
    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder  <- Used for jersey ROI
    6: right_shoulder <- Used for jersey ROI
    7: left_elbow
    8: right_elbow
    9: left_wrist
    10: right_wrist
    11: left_hip      <- Used for jersey ROI
    12: right_hip     <- Used for jersey ROI
    13: left_knee
    14: right_knee
    15: left_ankle
    16: right_ankle
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import IntEnum


class KeypointIndex(IntEnum):
    """COCO keypoint indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class Keypoint:
    """A single keypoint with coordinates and confidence."""
    x: float
    y: float
    confidence: float

    @property
    def is_visible(self) -> bool:
        """Check if keypoint is visible (confidence > threshold)."""
        # Lowered from 0.3 to 0.2 for better detection on small player crops
        return self.confidence > 0.2

    def as_tuple(self) -> Tuple[float, float]:
        """Return (x, y) tuple."""
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        """Return (x, y) as integers."""
        return (int(self.x), int(self.y))


@dataclass
class PoseResult:
    """Pose estimation result for a single person."""
    keypoints: List[Keypoint]
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    confidence: float = 0.0

    def get_keypoint(self, index: KeypointIndex) -> Optional[Keypoint]:
        """Get keypoint by index if visible."""
        if 0 <= index < len(self.keypoints):
            kp = self.keypoints[index]
            return kp if kp.is_visible else None
        return None

    @property
    def left_shoulder(self) -> Optional[Keypoint]:
        return self.get_keypoint(KeypointIndex.LEFT_SHOULDER)

    @property
    def right_shoulder(self) -> Optional[Keypoint]:
        return self.get_keypoint(KeypointIndex.RIGHT_SHOULDER)

    @property
    def left_hip(self) -> Optional[Keypoint]:
        return self.get_keypoint(KeypointIndex.LEFT_HIP)

    @property
    def right_hip(self) -> Optional[Keypoint]:
        return self.get_keypoint(KeypointIndex.RIGHT_HIP)

    @property
    def nose(self) -> Optional[Keypoint]:
        return self.get_keypoint(KeypointIndex.NOSE)

    def has_torso_keypoints(self) -> bool:
        """Check if we have enough keypoints to define torso."""
        # Need at least 2 of 4 torso keypoints
        torso_kps = [
            self.left_shoulder,
            self.right_shoulder,
            self.left_hip,
            self.right_hip
        ]
        visible = sum(1 for kp in torso_kps if kp is not None)
        return visible >= 2


class PoseEstimator:
    """
    YOLO11-pose based human pose estimator.

    Uses YOLO11-pose model to detect body keypoints.
    Optimized for football player crops.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        device: str = None,
        conf_threshold: float = 0.15  # Lowered from 0.25 for small player crops
    ):
        """
        Initialize pose estimator.

        Args:
            model_name: YOLO pose model name
            device: 'cpu', 'cuda', or None (auto)
            conf_threshold: Detection confidence threshold
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self._initialized = False

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device

    def initialize(self) -> bool:
        """Load YOLO pose model."""
        try:
            from ultralytics import YOLO

            print(f"Loading pose model: {self.model_name}")
            self.model = YOLO(self.model_name)

            self._initialized = True
            print(f"Pose estimator initialized on {self.device}")
            return True

        except ImportError as e:
            print(f"YOLO not available: {e}")
            return False
        except Exception as e:
            print(f"Pose model load failed: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def estimate(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose from player crop.

        Args:
            image: BGR player crop image

        Returns:
            PoseResult or None if detection failed
        """
        if not self._initialized or image is None or image.size == 0:
            return None

        try:
            # Run inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device
            )

            if not results or len(results) == 0:
                return None

            result = results[0]

            # Check if any persons detected
            if result.keypoints is None or len(result.keypoints) == 0:
                return None

            # Get first person's keypoints (player crop should have one person)
            kpts_data = result.keypoints.data[0]  # Shape: (17, 3) - x, y, conf

            # Convert to Keypoint objects
            keypoints = []
            for i in range(17):
                x, y, conf = kpts_data[i].tolist()
                keypoints.append(Keypoint(x=x, y=y, confidence=conf))

            # Get bounding box if available
            bbox = None
            if result.boxes is not None and len(result.boxes) > 0:
                box = result.boxes[0].xyxy[0].tolist()
                bbox = tuple(box)
                box_conf = result.boxes[0].conf[0].item()
            else:
                box_conf = 0.0

            return PoseResult(
                keypoints=keypoints,
                bbox=bbox,
                confidence=box_conf
            )

        except Exception as e:
            # Silently fail - pose estimation is optional enhancement
            return None

    def estimate_batch(self, images: List[np.ndarray]) -> List[Optional[PoseResult]]:
        """
        Batch pose estimation.

        Args:
            images: List of BGR player crop images

        Returns:
            List of PoseResult (or None for failed detections)
        """
        if not self._initialized:
            return [None] * len(images)

        results = []
        for img in images:
            results.append(self.estimate(img))
        return results

    def draw_keypoints(
        self,
        image: np.ndarray,
        pose: PoseResult,
        draw_skeleton: bool = True
    ) -> np.ndarray:
        """
        Draw keypoints on image for visualization.

        Args:
            image: BGR image
            pose: PoseResult with keypoints
            draw_skeleton: Whether to draw connecting lines

        Returns:
            Image with keypoints drawn
        """
        if pose is None:
            return image

        output = image.copy()

        # Skeleton connections (pairs of keypoint indices)
        skeleton = [
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes to ears
            (5, 6),          # Shoulder to shoulder
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10), # Right arm
            (5, 11), (6, 12),  # Shoulders to hips
            (11, 12),        # Hip to hip
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]

        # Draw skeleton
        if draw_skeleton:
            for i, j in skeleton:
                kp1 = pose.keypoints[i]
                kp2 = pose.keypoints[j]
                if kp1.is_visible and kp2.is_visible:
                    cv2.line(
                        output,
                        kp1.as_int_tuple(),
                        kp2.as_int_tuple(),
                        (0, 255, 255),
                        2
                    )

        # Draw keypoints
        for i, kp in enumerate(pose.keypoints):
            if kp.is_visible:
                # Use different colors for different body parts
                if i in [5, 6, 11, 12]:  # Torso keypoints
                    color = (0, 255, 0)  # Green
                elif i in [0, 1, 2, 3, 4]:  # Head
                    color = (255, 0, 0)  # Blue
                else:  # Limbs
                    color = (0, 0, 255)  # Red

                cv2.circle(output, kp.as_int_tuple(), 4, color, -1)

        return output

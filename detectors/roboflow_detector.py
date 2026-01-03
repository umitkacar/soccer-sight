"""
Roboflow Football Player Detection Module.

Uses Roboflow's pre-trained model for detecting:
- player
- goalkeeper
- referee
- ball

Requires internet connection for API inference.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    class_id: int


class RoboflowDetector:
    """
    Football player detector using Roboflow API.

    Classes:
        0: ball
        1: goalkeeper
        2: player
        3: referee
    """

    CLASS_NAMES = ['ball', 'goalkeeper', 'player', 'referee']
    CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        api_key: str = None,
        model_id: str = "football-players-detection-3zvbc/2",
        confidence_threshold: float = 0.25
    ):
        """
        Initialize Roboflow detector.

        Args:
            api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var)
            model_id: Model ID on Roboflow
            confidence_threshold: Minimum confidence for detections
        """
        self.api_key = api_key or os.environ.get('ROBOFLOW_API_KEY')
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.client = None
        self.is_initialized = False

        self._init_client()

    def _init_client(self):
        """Initialize Roboflow inference client."""
        try:
            from inference_sdk import InferenceHTTPClient

            if not self.api_key:
                print("Warning: No Roboflow API key provided")
                return

            self.client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=self.api_key
            )
            self.is_initialized = True
            print(f"Roboflow detector initialized: {self.model_id}")
            print(f"Classes: {self.CLASS_NAMES}")

        except ImportError:
            print("inference-sdk not installed. Run: pip install inference-sdk")
        except Exception as e:
            print(f"Roboflow initialization failed: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            return []

        try:
            # Save frame temporarily for API
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                temp_path = f.name
                cv2.imwrite(temp_path, frame)

            # Run inference
            result = self.client.infer(
                temp_path,
                model_id=self.model_id
            )

            # Clean up temp file
            os.unlink(temp_path)

            # Parse results
            detections = []
            for pred in result.get('predictions', []):
                conf = pred['confidence']
                if conf < self.confidence_threshold:
                    continue

                class_name = pred['class']
                class_id = self.CLASS_MAP.get(class_name, 2)  # default to player

                # Convert center format to corner format
                x, y = pred['x'], pred['y']
                w, h = pred['width'], pred['height']
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(int(x), int(y)),
                    class_id=class_id
                ))

            return detections

        except Exception as e:
            print(f"Roboflow detection error: {e}")
            return []

    def detect_from_path(self, image_path: str) -> List[Detection]:
        """
        Detect objects from image file.

        Args:
            image_path: Path to image file

        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            return []

        try:
            result = self.client.infer(image_path, model_id=self.model_id)

            detections = []
            for pred in result.get('predictions', []):
                conf = pred['confidence']
                if conf < self.confidence_threshold:
                    continue

                class_name = pred['class']
                class_id = self.CLASS_MAP.get(class_name, 2)

                x, y = pred['x'], pred['y']
                w, h = pred['width'], pred['height']
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(int(x), int(y)),
                    class_id=class_id
                ))

            return detections

        except Exception as e:
            print(f"Roboflow detection error: {e}")
            return []

    def to_supervision_format(self, detections: List[Detection]) -> tuple:
        """
        Convert detections to supervision format for tracking.

        Returns:
            (xyxy, confidence, class_ids) arrays
        """
        if not detections:
            return np.empty((0, 4)), np.array([]), np.array([])

        xyxy = np.array([d.bbox for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        return xyxy, confidence, class_ids

    def filter_players(self, detections: List[Detection]) -> List[Detection]:
        """Filter to only player and goalkeeper detections."""
        return [d for d in detections if d.class_name in ['player', 'goalkeeper']]

    def filter_ball(self, detections: List[Detection]) -> List[Detection]:
        """Filter to only ball detections."""
        return [d for d in detections if d.class_name == 'ball']

    def filter_referee(self, detections: List[Detection]) -> List[Detection]:
        """Filter to only referee detections."""
        return [d for d in detections if d.class_name == 'referee']


def create_roboflow_detector(api_key: str = None) -> Optional[RoboflowDetector]:
    """
    Factory function to create Roboflow detector.

    Args:
        api_key: Roboflow API key

    Returns:
        RoboflowDetector instance or None if initialization fails
    """
    detector = RoboflowDetector(api_key=api_key)
    if detector.is_initialized:
        return detector
    return None

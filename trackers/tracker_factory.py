"""
Tracker Factory for Multi-Object Tracking.

Supports:
- ByteTrack: Fast, good baseline via supervision
- BoT-SORT: Better occlusion handling with ReID via boxmot
"""

from enum import Enum
from typing import Optional, Any
import numpy as np


class TrackerType(Enum):
    """Available tracker types."""
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"
    OCSORT = "ocsort"
    DEEPOCSORT = "deepocsort"


AVAILABLE_TRACKERS = [t.value for t in TrackerType]


class BoTSORTWrapper:
    """
    Wrapper for BoxMOT's BoT-SORT tracker.

    BoT-SORT combines:
    - Motion prediction (Kalman filter)
    - Appearance features (ReID)
    - Camera motion compensation

    Better for:
    - Occlusion handling
    - Similar jersey differentiation
    - Re-identification after lost
    """

    def __init__(
        self,
        reid_weights: str = "osnet_x0_25_msmt17.pt",
        track_high_thresh: float = 0.3,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 90,
        match_thresh: float = 0.7,
        with_reid: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize BoT-SORT tracker.

        Args:
            reid_weights: ReID model weights (osnet_x0_25_msmt17.pt recommended)
            track_high_thresh: High confidence threshold for tracking
            track_low_thresh: Low confidence threshold
            new_track_thresh: Threshold for creating new tracks
            track_buffer: Frames to keep lost tracks (90 = 3 sec at 30fps)
            match_thresh: IoU threshold for matching
            with_reid: Enable ReID features for better jersey differentiation
            device: Device to run on (cpu/cuda)
        """
        self.tracker = None
        self.with_reid = with_reid
        self.device = device

        try:
            from boxmot import BotSort

            self.tracker = BotSort(
                reid_weights=reid_weights if with_reid else None,
                device=device,
                half=False,  # Use FP32 for CPU
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh
            )
            print(f"BoT-SORT initialized (ReID: {with_reid})")

        except ImportError:
            print("BoxMOT not installed. Run: pip install boxmot")
        except Exception as e:
            print(f"BoT-SORT initialization failed: {e}")

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: Array of [x1, y1, x2, y2, conf, class_id]
            frame: Current frame (BGR) for ReID features

        Returns:
            Array of [x1, y1, x2, y2, track_id, conf, class_id, ...]
        """
        if self.tracker is None:
            return np.empty((0, 7))

        if len(detections) == 0:
            # Still need to update tracker for track management
            return self.tracker.update(np.empty((0, 6)), frame)

        return self.tracker.update(detections, frame)

    def reset(self):
        """Reset tracker state."""
        if self.tracker is not None:
            self.tracker.reset()


class ByteTrackWrapper:
    """
    Wrapper for supervision's ByteTrack tracker.

    ByteTrack is:
    - Very fast (1200+ FPS)
    - Good baseline performance
    - No ReID (pure motion-based)
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.20,
        lost_track_buffer: int = 90,
        minimum_matching_threshold: float = 0.7,
        frame_rate: int = 30
    ):
        """
        Initialize ByteTrack tracker.

        Args:
            track_activation_threshold: Confidence to activate track
            lost_track_buffer: Frames to keep lost tracks
            minimum_matching_threshold: IoU threshold for matching
            frame_rate: Video frame rate
        """
        self.tracker = None

        try:
            import supervision as sv

            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate
            )
            print(f"ByteTrack initialized (buffer: {lost_track_buffer})")

        except ImportError:
            print("supervision not installed. Run: pip install supervision")
        except Exception as e:
            print(f"ByteTrack initialization failed: {e}")

    def update_with_detections(self, detections):
        """
        Update tracker with supervision Detections object.

        Args:
            detections: supervision.Detections object

        Returns:
            supervision.Detections with tracker_id
        """
        if self.tracker is None:
            return detections

        return self.tracker.update_with_detections(detections)

    def reset(self):
        """Reset tracker state."""
        if self.tracker is not None:
            self.tracker.reset()


def create_tracker(
    tracker_type: str = "botsort",
    track_buffer: int = 90,
    with_reid: bool = True,
    device: str = "cpu",
    frame_rate: int = 30,
    **kwargs
) -> Optional[Any]:
    """
    Factory function to create tracker.

    Args:
        tracker_type: "bytetrack" or "botsort"
        track_buffer: Frames to keep lost tracks
        with_reid: Enable ReID (BoT-SORT only)
        device: Device for inference
        frame_rate: Video frame rate
        **kwargs: Additional tracker-specific args

    Returns:
        Tracker instance
    """
    tracker_type = tracker_type.lower()

    if tracker_type == "botsort":
        return BoTSORTWrapper(
            track_buffer=track_buffer,
            with_reid=with_reid,
            device=device,
            track_high_thresh=kwargs.get('track_high_thresh', 0.3),
            track_low_thresh=kwargs.get('track_low_thresh', 0.1),
            new_track_thresh=kwargs.get('new_track_thresh', 0.6),
            match_thresh=kwargs.get('match_thresh', 0.7)
        )

    elif tracker_type == "bytetrack":
        return ByteTrackWrapper(
            track_activation_threshold=kwargs.get('track_activation_threshold', 0.20),
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=kwargs.get('minimum_matching_threshold', 0.7),
            frame_rate=frame_rate
        )

    elif tracker_type == "ocsort":
        try:
            from boxmot import OCSORT
            return OCSORT(
                det_thresh=kwargs.get('det_thresh', 0.3),
                max_age=track_buffer,
                min_hits=kwargs.get('min_hits', 3),
                iou_threshold=kwargs.get('iou_threshold', 0.3),
                delta_t=kwargs.get('delta_t', 3),
                asso_func=kwargs.get('asso_func', 'iou'),
                inertia=kwargs.get('inertia', 0.2)
            )
        except ImportError:
            print("BoxMOT not installed for OC-SORT")
            return None

    elif tracker_type == "deepocsort":
        try:
            from boxmot import DeepOCSORT
            return DeepOCSORT(
                reid_weights="osnet_x0_25_msmt17.pt" if with_reid else None,
                device=device,
                max_age=track_buffer
            )
        except ImportError:
            print("BoxMOT not installed for Deep OC-SORT")
            return None

    else:
        print(f"Unknown tracker type: {tracker_type}")
        print(f"Available: {AVAILABLE_TRACKERS}")
        return None

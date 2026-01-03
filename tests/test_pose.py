"""
Tests for the pose estimation module.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKeypoint:
    """Test Keypoint dataclass."""

    def test_keypoint_creation(self):
        """Test creating a keypoint."""
        from pose import Keypoint

        kp = Keypoint(x=100.5, y=200.3, confidence=0.8)
        assert kp.x == 100.5
        assert kp.y == 200.3
        assert kp.confidence == 0.8

    def test_keypoint_visibility_high_conf(self):
        """Test keypoint visibility with high confidence."""
        from pose import Keypoint

        kp = Keypoint(x=100, y=200, confidence=0.8)
        assert kp.is_visible is True

    def test_keypoint_visibility_low_conf(self):
        """Test keypoint visibility with low confidence."""
        from pose import Keypoint

        kp = Keypoint(x=100, y=200, confidence=0.2)
        assert kp.is_visible is False

    def test_keypoint_as_tuple(self):
        """Test keypoint as tuple."""
        from pose import Keypoint

        kp = Keypoint(x=100.5, y=200.3, confidence=0.8)
        assert kp.as_tuple() == (100.5, 200.3)

    def test_keypoint_as_int_tuple(self):
        """Test keypoint as integer tuple."""
        from pose import Keypoint

        kp = Keypoint(x=100.7, y=200.3, confidence=0.8)
        assert kp.as_int_tuple() == (100, 200)


class TestPoseResult:
    """Test PoseResult dataclass."""

    def test_pose_result_creation(self):
        """Test creating a pose result."""
        from pose import PoseResult, Keypoint

        keypoints = [Keypoint(x=i*10, y=i*10, confidence=0.8) for i in range(17)]
        pose = PoseResult(keypoints=keypoints, confidence=0.9)

        assert len(pose.keypoints) == 17
        assert pose.confidence == 0.9

    def test_pose_result_get_shoulder(self):
        """Test getting shoulder keypoints."""
        from pose import PoseResult, Keypoint

        keypoints = [Keypoint(x=0, y=0, confidence=0.1) for _ in range(17)]
        # Set shoulders with high confidence
        keypoints[5] = Keypoint(x=50, y=100, confidence=0.9)  # left shoulder
        keypoints[6] = Keypoint(x=150, y=100, confidence=0.9)  # right shoulder

        pose = PoseResult(keypoints=keypoints)

        assert pose.left_shoulder is not None
        assert pose.left_shoulder.x == 50
        assert pose.right_shoulder is not None
        assert pose.right_shoulder.x == 150

    def test_pose_result_invisible_keypoint(self):
        """Test invisible keypoints return None."""
        from pose import PoseResult, Keypoint

        keypoints = [Keypoint(x=0, y=0, confidence=0.1) for _ in range(17)]
        pose = PoseResult(keypoints=keypoints)

        # All keypoints have low confidence
        assert pose.left_shoulder is None
        assert pose.right_shoulder is None

    def test_has_torso_keypoints_true(self):
        """Test has_torso_keypoints with visible keypoints."""
        from pose import PoseResult, Keypoint

        keypoints = [Keypoint(x=0, y=0, confidence=0.1) for _ in range(17)]
        keypoints[5] = Keypoint(x=50, y=100, confidence=0.9)  # left shoulder
        keypoints[6] = Keypoint(x=150, y=100, confidence=0.9)  # right shoulder

        pose = PoseResult(keypoints=keypoints)
        assert pose.has_torso_keypoints() is True

    def test_has_torso_keypoints_false(self):
        """Test has_torso_keypoints with no visible keypoints."""
        from pose import PoseResult, Keypoint

        keypoints = [Keypoint(x=0, y=0, confidence=0.1) for _ in range(17)]
        pose = PoseResult(keypoints=keypoints)
        assert pose.has_torso_keypoints() is False


class TestPoseEstimator:
    """Test PoseEstimator class."""

    def test_pose_estimator_init(self):
        """Test pose estimator initialization."""
        from pose import PoseEstimator

        estimator = PoseEstimator()
        assert estimator.model_name == "yolo11n-pose.pt"
        assert not estimator.is_initialized

    def test_pose_estimator_device_autodetect(self):
        """Test auto device detection."""
        from pose import PoseEstimator

        estimator = PoseEstimator(device=None)
        assert estimator.device in ['cpu', 'cuda']

    def test_pose_estimator_explicit_device(self):
        """Test explicit device setting."""
        from pose import PoseEstimator

        estimator = PoseEstimator(device='cpu')
        assert estimator.device == 'cpu'

    def test_estimate_without_init(self):
        """Test estimate returns None without initialization."""
        from pose import PoseEstimator

        estimator = PoseEstimator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = estimator.estimate(image)
        assert result is None

    def test_estimate_empty_image(self):
        """Test estimate with empty image."""
        from pose import PoseEstimator

        estimator = PoseEstimator()
        result = estimator.estimate(np.array([]))
        assert result is None


class TestROIExtractor:
    """Test ROIExtractor class."""

    def test_roi_extractor_init(self):
        """Test ROI extractor initialization."""
        from pose import ROIExtractor

        extractor = ROIExtractor()
        assert extractor.padding_ratio == 0.1
        assert extractor.min_roi_size == 20

    def test_extract_jersey_roi_empty_image(self):
        """Test extraction with empty image."""
        from pose import ROIExtractor

        extractor = ROIExtractor()
        result = extractor.extract_jersey_roi(np.array([]))

        assert result.method == 'none'
        assert result.roi.size == 0

    def test_extract_jersey_roi_fallback(self):
        """Test extraction without pose (fallback)."""
        from pose import ROIExtractor

        extractor = ROIExtractor()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        result = extractor.extract_jersey_roi(image, pose=None)

        assert result.method == 'fallback'
        assert result.roi.size > 0
        assert result.confidence == 0.5

    def test_extract_jersey_roi_with_pose(self):
        """Test extraction with valid pose."""
        from pose import ROIExtractor, PoseResult, Keypoint

        # Create mock pose with torso keypoints
        keypoints = [Keypoint(x=0, y=0, confidence=0.1) for _ in range(17)]
        # Shoulders
        keypoints[5] = Keypoint(x=30, y=40, confidence=0.9)   # left shoulder
        keypoints[6] = Keypoint(x=70, y=40, confidence=0.9)   # right shoulder
        # Hips
        keypoints[11] = Keypoint(x=35, y=120, confidence=0.9)  # left hip
        keypoints[12] = Keypoint(x=65, y=120, confidence=0.9)  # right hip

        pose = PoseResult(keypoints=keypoints)

        extractor = ROIExtractor()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        result = extractor.extract_jersey_roi(image, pose)

        assert result.method == 'pose'
        assert result.roi.size > 0
        assert result.confidence == 1.0  # All 4 torso keypoints visible

    def test_extract_torso_roi(self):
        """Test full torso extraction."""
        from pose import ROIExtractor

        extractor = ROIExtractor()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        result = extractor.extract_torso_roi(image)

        assert result.method == 'fallback'
        assert result.roi.size > 0

    def test_extract_number_region(self):
        """Test number region extraction."""
        from pose import ROIExtractor

        extractor = ROIExtractor()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        result = extractor.extract_number_region(image)

        assert result.method == 'fallback'
        assert result.roi.size > 0


class TestROIResult:
    """Test ROIResult dataclass."""

    def test_roi_result_creation(self):
        """Test creating ROI result."""
        from pose.roi_extractor import ROIResult

        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        result = ROIResult(
            roi=roi,
            bbox=(10, 20, 60, 70),
            method='pose',
            confidence=0.9
        )

        assert result.roi.shape == (50, 50, 3)
        assert result.bbox == (10, 20, 60, 70)
        assert result.method == 'pose'
        assert result.confidence == 0.9


class TestFactory:
    """Test factory functions."""

    def test_create_roi_extractor_jersey(self):
        """Test creating jersey ROI extractor."""
        from pose.roi_extractor import create_roi_extractor

        extractor = create_roi_extractor('jersey')
        assert extractor.jersey_bottom_ratio == 0.6

    def test_create_roi_extractor_torso(self):
        """Test creating torso ROI extractor."""
        from pose.roi_extractor import create_roi_extractor

        extractor = create_roi_extractor('torso')
        assert extractor.jersey_bottom_ratio == 1.0

    def test_create_roi_extractor_number(self):
        """Test creating number ROI extractor."""
        from pose.roi_extractor import create_roi_extractor

        extractor = create_roi_extractor('number')
        assert extractor.jersey_bottom_ratio == 0.5

    def test_create_roi_extractor_unknown(self):
        """Test creating with unknown mode uses defaults."""
        from pose.roi_extractor import create_roi_extractor

        extractor = create_roi_extractor('unknown_mode')
        # Should use default ROIExtractor
        assert extractor is not None


class TestKeypointIndex:
    """Test KeypointIndex enum."""

    def test_keypoint_indices(self):
        """Test keypoint index values."""
        from pose.pose_estimator import KeypointIndex

        assert KeypointIndex.NOSE == 0
        assert KeypointIndex.LEFT_SHOULDER == 5
        assert KeypointIndex.RIGHT_SHOULDER == 6
        assert KeypointIndex.LEFT_HIP == 11
        assert KeypointIndex.RIGHT_HIP == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

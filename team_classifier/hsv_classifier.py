"""
HSV Color-based Team Classifier.

Uses HSV color space analysis to classify teams.
Fast but sensitive to lighting conditions.

Accuracy: ~75%
Speed: Very Fast (1ms per crop)
"""

import cv2
import numpy as np
from typing import Tuple
from .base import TeamClassifier, TeamType


class HSVTeamClassifier(TeamClassifier):
    """
    HSV color-based team classification.

    Analyzes the dominant colors in HSV space to determine team.
    Works well when teams have distinctly different colored jerseys.

    Pros:
    - Very fast (no ML required)
    - No GPU needed
    - Simple to configure

    Cons:
    - Sensitive to lighting changes
    - Struggles with similar jersey colors
    - Requires manual HSV range configuration
    """

    def __init__(
        self,
        team_a_hsv_lower: Tuple[int, int, int] = (0, 100, 100),
        team_a_hsv_upper: Tuple[int, int, int] = (10, 255, 255),
        team_b_hsv_lower: Tuple[int, int, int] = (80, 100, 100),
        team_b_hsv_upper: Tuple[int, int, int] = (100, 255, 255),
        min_ratio: float = 0.10
    ):
        """
        Initialize HSV classifier.

        Args:
            team_a_hsv_lower: Lower HSV bound for Team A
            team_a_hsv_upper: Upper HSV bound for Team A
            team_b_hsv_lower: Lower HSV bound for Team B
            team_b_hsv_upper: Upper HSV bound for Team B
            min_ratio: Minimum ratio of pixels to classify as team
        """
        super().__init__(name="hsv")
        self.team_a_lower = np.array(team_a_hsv_lower)
        self.team_a_upper = np.array(team_a_hsv_upper)
        self.team_b_lower = np.array(team_b_hsv_lower)
        self.team_b_upper = np.array(team_b_hsv_upper)
        self.min_ratio = min_ratio

    def initialize(self) -> bool:
        """HSV classifier needs no initialization."""
        self._initialized = True
        return True

    def classify(self, image: np.ndarray) -> TeamType:
        """
        Classify player using HSV color analysis.

        Args:
            image: BGR player crop image

        Returns:
            TeamType classification
        """
        if image is None or image.size == 0:
            return TeamType.UNKNOWN

        # Extract jersey region
        jersey = self.extract_jersey_region(image)

        if jersey is None or jersey.size == 0:
            return TeamType.UNKNOWN

        # Convert to HSV
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

        # Count pixels matching each team's color
        team_a_mask = cv2.inRange(hsv, self.team_a_lower, self.team_a_upper)
        team_b_mask = cv2.inRange(hsv, self.team_b_lower, self.team_b_upper)

        total_pixels = jersey.shape[0] * jersey.shape[1]
        team_a_ratio = np.sum(team_a_mask > 0) / total_pixels
        team_b_ratio = np.sum(team_b_mask > 0) / total_pixels

        # Classify based on dominant color
        if team_a_ratio >= self.min_ratio and team_a_ratio > team_b_ratio:
            return TeamType.TEAM_A
        elif team_b_ratio >= self.min_ratio and team_b_ratio > team_a_ratio:
            return TeamType.TEAM_B

        return TeamType.UNKNOWN

    def set_team_colors(
        self,
        team_a_lower: Tuple[int, int, int],
        team_a_upper: Tuple[int, int, int],
        team_b_lower: Tuple[int, int, int],
        team_b_upper: Tuple[int, int, int]
    ) -> None:
        """
        Update team color ranges.

        Args:
            team_a_lower: Lower HSV bound for Team A
            team_a_upper: Upper HSV bound for Team A
            team_b_lower: Lower HSV bound for Team B
            team_b_upper: Upper HSV bound for Team B
        """
        self.team_a_lower = np.array(team_a_lower)
        self.team_a_upper = np.array(team_a_upper)
        self.team_b_lower = np.array(team_b_lower)
        self.team_b_upper = np.array(team_b_upper)

    def auto_calibrate(self, team_a_samples: list, team_b_samples: list) -> None:
        """
        Auto-calibrate HSV ranges from samples.

        Analyzes provided samples to determine optimal HSV ranges.

        Args:
            team_a_samples: List of Team A player crops
            team_b_samples: List of Team B player crops
        """
        def get_hsv_stats(samples):
            """Get HSV statistics from samples."""
            all_h, all_s, all_v = [], [], []

            for img in samples:
                if img is None or img.size == 0:
                    continue
                jersey = self.extract_jersey_region(img)
                if jersey is None or jersey.size == 0:
                    continue

                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                all_h.extend(hsv[:, :, 0].flatten())
                all_s.extend(hsv[:, :, 1].flatten())
                all_v.extend(hsv[:, :, 2].flatten())

            if not all_h:
                return None

            # Calculate percentiles for robust range
            h_low, h_high = np.percentile(all_h, [10, 90])
            s_low, s_high = np.percentile(all_s, [10, 90])
            v_low, v_high = np.percentile(all_v, [10, 90])

            return (
                (int(h_low), int(s_low), int(v_low)),
                (int(h_high), int(s_high), int(v_high))
            )

        # Calibrate Team A
        if team_a_samples:
            stats = get_hsv_stats(team_a_samples)
            if stats:
                self.team_a_lower = np.array(stats[0])
                self.team_a_upper = np.array(stats[1])

        # Calibrate Team B
        if team_b_samples:
            stats = get_hsv_stats(team_b_samples)
            if stats:
                self.team_b_lower = np.array(stats[0])
                self.team_b_upper = np.array(stats[1])

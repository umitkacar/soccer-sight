"""
Base Team Classifier - Abstract class for team classification approaches.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


class TeamType(Enum):
    """Team classification result."""
    UNKNOWN = "UNKNOWN"
    TEAM_A = "TEAM_A"  # First team (e.g., home team)
    TEAM_B = "TEAM_B"  # Second team (e.g., away team)
    REFEREE = "REFEREE"  # Referee (different uniform)
    GOALKEEPER = "GOALKEEPER"  # Goalkeeper (different jersey)


class TeamClassifier(ABC):
    """
    Abstract base class for team classifiers.

    Team classifiers analyze player crops to determine which team they belong to.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._initialized = False
        self._team_a_samples: List[np.ndarray] = []
        self._team_b_samples: List[np.ndarray] = []

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the classifier.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def classify(self, image: np.ndarray) -> TeamType:
        """
        Classify a player crop into a team.

        Args:
            image: BGR player crop image

        Returns:
            TeamType classification
        """
        pass

    def classify_batch(self, images: List[np.ndarray]) -> List[TeamType]:
        """
        Classify multiple player crops.

        Default implementation processes sequentially.
        Subclasses can override for batch processing.

        Args:
            images: List of BGR player crop images

        Returns:
            List of TeamType classifications
        """
        return [self.classify(img) for img in images]

    def add_team_sample(self, image: np.ndarray, team: TeamType) -> None:
        """
        Add a known sample for team learning.

        Some classifiers (like KMeans, SigLIP) can learn team colors
        from provided samples.

        Args:
            image: BGR player crop image
            team: Known team for this sample
        """
        if team == TeamType.TEAM_A:
            self._team_a_samples.append(image.copy())
        elif team == TeamType.TEAM_B:
            self._team_b_samples.append(image.copy())

    def clear_samples(self) -> None:
        """Clear all team samples."""
        self._team_a_samples.clear()
        self._team_b_samples.clear()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def has_samples(self) -> bool:
        return len(self._team_a_samples) > 0 or len(self._team_b_samples) > 0

    @staticmethod
    def extract_jersey_region(image: np.ndarray) -> np.ndarray:
        """
        Extract the jersey/torso region from a player crop.

        Args:
            image: Full player bounding box crop

        Returns:
            Cropped jersey region
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]

        # Jersey is roughly in the upper-middle of the bounding box
        # Skip head (top 15%) and legs (bottom 50%)
        top = int(h * 0.15)
        bottom = int(h * 0.50)

        # Use center 80% horizontally
        left = int(w * 0.10)
        right = int(w * 0.90)

        jersey = image[top:bottom, left:right]

        if jersey.size == 0:
            return image

        return jersey

    @staticmethod
    def get_dominant_color(image: np.ndarray) -> Tuple[int, int, int]:
        """
        Get the dominant color of an image.

        Args:
            image: BGR image

        Returns:
            (B, G, R) dominant color tuple
        """
        if image is None or image.size == 0:
            return (128, 128, 128)

        # Resize for speed
        small = cv2.resize(image, (50, 50))

        # Calculate mean color
        mean_color = np.mean(small, axis=(0, 1))

        return tuple(map(int, mean_color))


# Import cv2 for static methods
import cv2

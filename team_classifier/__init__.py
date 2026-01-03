"""
Team Classification Module for Football Player Tracking.

Provides multiple approaches to classify players into teams:
- HSV Color Analysis (legacy, ~75% accuracy)
- KMeans Clustering (RGB, ~80% accuracy)
- SigLIP + UMAP (state-of-the-art, ~95% accuracy)

Usage:
    from team_classifier import create_team_classifier

    # Simple HSV-based (fast)
    classifier = create_team_classifier('hsv')

    # SigLIP-based (accurate)
    classifier = create_team_classifier('siglip')

    # Classify players
    team = classifier.classify(player_crop)
"""

from .base import TeamClassifier, TeamType
from .hsv_classifier import HSVTeamClassifier
from .kmeans_classifier import KMeansTeamClassifier
from .siglip_classifier import SigLIPTeamClassifier
from .factory import create_team_classifier, create_best_available_classifier

__all__ = [
    'TeamClassifier',
    'TeamType',
    'HSVTeamClassifier',
    'KMeansTeamClassifier',
    'SigLIPTeamClassifier',
    'create_team_classifier',
    'create_best_available_classifier',
]

"""
Team Classifier Factory - Creates and configures team classifiers.
"""

from typing import Optional, Dict, Any
from .base import TeamClassifier
from .hsv_classifier import HSVTeamClassifier
from .kmeans_classifier import KMeansTeamClassifier
from .siglip_classifier import SigLIPTeamClassifier


CLASSIFIER_REGISTRY = {
    'hsv': HSVTeamClassifier,
    'kmeans': KMeansTeamClassifier,
    'siglip': SigLIPTeamClassifier,
}


def create_team_classifier(
    classifier_type: str,
    config: Dict[str, Any] = None,
    auto_init: bool = True
) -> Optional[TeamClassifier]:
    """
    Create a team classifier by type.

    Args:
        classifier_type: 'hsv', 'kmeans', or 'siglip'
        config: Optional configuration dict
        auto_init: Whether to auto-initialize

    Returns:
        TeamClassifier instance or None

    Examples:
        # Fast HSV-based
        classifier = create_team_classifier('hsv')

        # KMeans clustering
        classifier = create_team_classifier('kmeans', {
            'color_space': 'lab',
            'n_clusters': 2
        })

        # SigLIP (highest accuracy)
        classifier = create_team_classifier('siglip', {
            'device': 'cuda'
        })
    """
    config = config or {}

    if classifier_type not in CLASSIFIER_REGISTRY:
        print(f"Unknown classifier: {classifier_type}")
        print(f"Available: {list(CLASSIFIER_REGISTRY.keys())}")
        return None

    classifier_class = CLASSIFIER_REGISTRY[classifier_type]

    try:
        classifier = classifier_class(**config)

        if auto_init:
            if not classifier.initialize():
                print(f"Failed to initialize {classifier_type}")
                # For siglip, fall back to kmeans
                if classifier_type == 'siglip':
                    print("Falling back to KMeans classifier")
                    return create_team_classifier('kmeans', auto_init=True)
                return None

        return classifier

    except Exception as e:
        print(f"Error creating {classifier_type}: {e}")
        return None


def create_best_available_classifier(
    preferred_order: list = None,
    config: Dict[str, Any] = None
) -> Optional[TeamClassifier]:
    """
    Create the best available team classifier.

    Tries classifiers in order of preference.

    Args:
        preferred_order: List of classifier types to try
        config: Configuration dict

    Returns:
        Best available TeamClassifier
    """
    if preferred_order is None:
        preferred_order = ['siglip', 'kmeans', 'hsv']

    config = config or {}

    for classifier_type in preferred_order:
        classifier_config = config.get(classifier_type, {})
        classifier = create_team_classifier(
            classifier_type,
            classifier_config,
            auto_init=True
        )
        if classifier and classifier.is_initialized:
            print(f"Using {classifier_type} team classifier")
            return classifier

    print("No team classifier available!")
    return None

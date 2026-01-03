"""
Detection modules for football player tracking.

Available detectors:
- RoboflowDetector: Cloud-based detection with correct football classes
"""

from .roboflow_detector import RoboflowDetector, Detection, create_roboflow_detector

__all__ = [
    'RoboflowDetector',
    'Detection',
    'create_roboflow_detector'
]

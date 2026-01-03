"""
Detection modules for football player tracking.

Available detectors:
- YOLO11: Local inference for player detection
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

__all__ = [
    'Detection'
]

"""
Multi-Object Tracking modules for football player tracking.

Available trackers:
- ByteTrack: Fast, good baseline (current default)
- BoT-SORT: Better occlusion handling with ReID
"""

from .tracker_factory import create_tracker, TrackerType, AVAILABLE_TRACKERS

__all__ = [
    'create_tracker',
    'TrackerType',
    'AVAILABLE_TRACKERS'
]

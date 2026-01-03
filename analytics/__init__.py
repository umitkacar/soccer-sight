"""
Analytics Module for Football Player Tracking.

Provides speed/distance calculation and player statistics.

Usage:
    from analytics import SpeedCalculator, PlayerStats

    # Initialize speed calculator
    calculator = SpeedCalculator(fps=30.0, scale_factor=0.1)

    # Update with new position
    calculator.update(player_id=1, position=(500, 300), frame_num=100)

    # Get player stats
    stats = calculator.get_stats(player_id=1)
"""

from .speed_calculator import SpeedCalculator, MovementVector
from .player_stats import PlayerStats, PlayerAnalytics
from .radar_view import RADARView, PlayerMarker, PitchType, create_radar_view

__all__ = [
    'SpeedCalculator',
    'MovementVector',
    'PlayerStats',
    'PlayerAnalytics',
    'RADARView',
    'PlayerMarker',
    'PitchType',
    'create_radar_view',
]

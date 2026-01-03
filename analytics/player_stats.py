"""
Player Statistics and Analytics.

Comprehensive player statistics including:
- Movement metrics (speed, distance)
- Zone heatmap data
- Time-based statistics
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class PlayerStats:
    """Statistical summary for a single player."""
    player_id: int
    jersey_number: Optional[str] = None
    team: Optional[str] = None

    # Movement stats
    total_distance_m: float = 0.0
    current_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0

    # Time stats
    time_walking: float = 0.0  # seconds
    time_jogging: float = 0.0
    time_running: float = 0.0
    time_sprinting: float = 0.0
    time_standing: float = 0.0

    # Position stats
    avg_position_x: float = 0.0
    avg_position_y: float = 0.0
    position_spread: float = 0.0  # How much area covered

    # Sprint stats
    sprint_count: int = 0
    avg_sprint_distance: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'player_id': self.player_id,
            'jersey_number': self.jersey_number,
            'team': self.team,
            'total_distance_m': round(self.total_distance_m, 1),
            'current_speed_kmh': round(self.current_speed_kmh, 1),
            'max_speed_kmh': round(self.max_speed_kmh, 1),
            'avg_speed_kmh': round(self.avg_speed_kmh, 1),
            'time_walking': round(self.time_walking, 1),
            'time_jogging': round(self.time_jogging, 1),
            'time_running': round(self.time_running, 1),
            'time_sprinting': round(self.time_sprinting, 1),
            'sprint_count': self.sprint_count
        }


class PlayerAnalytics:
    """
    Comprehensive player analytics and statistics tracking.

    Aggregates data from SpeedCalculator and provides
    additional analytics like heatmaps and zone occupancy.
    """

    # Pitch zones (3x3 grid)
    ZONE_NAMES = [
        ['Defensive Left', 'Defensive Center', 'Defensive Right'],
        ['Midfield Left', 'Midfield Center', 'Midfield Right'],
        ['Attacking Left', 'Attacking Center', 'Attacking Right']
    ]

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        fps: float = 30.0
    ):
        """
        Initialize analytics tracker.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # Player data storage
        self._stats: Dict[int, PlayerStats] = {}
        self._zone_counts: Dict[int, np.ndarray] = {}  # 3x3 zone heatmap
        self._heatmap_data: Dict[int, np.ndarray] = {}  # Full resolution heatmap

        # Sprint tracking
        self._sprint_active: Dict[int, bool] = {}
        self._sprint_start_distance: Dict[int, float] = {}

        # Last movement type for time tracking
        self._last_movement: Dict[int, str] = {}

    def get_or_create_stats(self, player_id: int) -> PlayerStats:
        """Get or create stats for a player."""
        if player_id not in self._stats:
            self._stats[player_id] = PlayerStats(player_id=player_id)
            self._zone_counts[player_id] = np.zeros((3, 3))
            self._sprint_active[player_id] = False
            self._last_movement[player_id] = "standing"
        return self._stats[player_id]

    def update(
        self,
        player_id: int,
        position: Tuple[float, float],
        speed_ms: float,
        total_distance: float,
        movement_type: str,
        jersey_number: Optional[str] = None,
        team: Optional[str] = None
    ):
        """
        Update player analytics with new data.

        Args:
            player_id: Player identifier
            position: (x, y) pixel position
            speed_ms: Current speed in m/s
            total_distance: Total distance covered
            movement_type: 'standing', 'walking', 'jogging', 'running', 'sprinting'
            jersey_number: Jersey number if known
            team: Team name if known
        """
        stats = self.get_or_create_stats(player_id)

        # Update basic stats
        stats.total_distance_m = total_distance
        stats.current_speed_kmh = speed_ms * 3.6
        stats.max_speed_kmh = max(stats.max_speed_kmh, speed_ms * 3.6)

        if jersey_number:
            stats.jersey_number = jersey_number
        if team:
            stats.team = team

        # Update time stats
        self._update_time_stats(player_id, movement_type)

        # Update zone occupancy
        self._update_zone(player_id, position)

        # Track sprints
        self._track_sprint(player_id, movement_type, total_distance)

        # Update position average
        self._update_position_stats(player_id, position)

    def _update_time_stats(self, player_id: int, movement_type: str):
        """Update time spent in each movement type."""
        stats = self._stats[player_id]

        # Add time for previous movement type
        if movement_type == "standing":
            stats.time_standing += self.frame_interval
        elif movement_type == "walking":
            stats.time_walking += self.frame_interval
        elif movement_type == "jogging":
            stats.time_jogging += self.frame_interval
        elif movement_type == "running":
            stats.time_running += self.frame_interval
        elif movement_type == "sprinting":
            stats.time_sprinting += self.frame_interval

        self._last_movement[player_id] = movement_type

    def _update_zone(self, player_id: int, position: Tuple[float, float]):
        """Update zone occupancy grid."""
        x, y = position

        # Calculate zone (3x3 grid)
        zone_x = min(2, int(x / self.frame_width * 3))
        zone_y = min(2, int(y / self.frame_height * 3))

        zone_x = max(0, zone_x)
        zone_y = max(0, zone_y)

        self._zone_counts[player_id][zone_y, zone_x] += 1

    def _track_sprint(
        self,
        player_id: int,
        movement_type: str,
        total_distance: float
    ):
        """Track sprint events."""
        stats = self._stats[player_id]
        is_sprinting = movement_type == "sprinting"

        if is_sprinting and not self._sprint_active[player_id]:
            # Sprint started
            self._sprint_active[player_id] = True
            self._sprint_start_distance[player_id] = total_distance
        elif not is_sprinting and self._sprint_active[player_id]:
            # Sprint ended
            self._sprint_active[player_id] = False
            sprint_distance = total_distance - self._sprint_start_distance.get(player_id, total_distance)

            if sprint_distance > 0:
                stats.sprint_count += 1
                # Update average sprint distance
                total_sprints = stats.sprint_count
                stats.avg_sprint_distance = (
                    (stats.avg_sprint_distance * (total_sprints - 1) + sprint_distance)
                    / total_sprints
                )

    def _update_position_stats(
        self,
        player_id: int,
        position: Tuple[float, float]
    ):
        """Update position-related statistics."""
        stats = self._stats[player_id]
        x, y = position

        # Running average of position
        if stats.avg_position_x == 0:
            stats.avg_position_x = x
            stats.avg_position_y = y
        else:
            # Exponential moving average
            alpha = 0.01
            stats.avg_position_x = alpha * x + (1 - alpha) * stats.avg_position_x
            stats.avg_position_y = alpha * y + (1 - alpha) * stats.avg_position_y

    def get_zone_name(self, player_id: int) -> str:
        """Get most occupied zone name for player."""
        if player_id not in self._zone_counts:
            return "Unknown"

        zones = self._zone_counts[player_id]
        idx = np.unravel_index(np.argmax(zones), zones.shape)
        return self.ZONE_NAMES[idx[0]][idx[1]]

    def get_zone_heatmap(self, player_id: int) -> np.ndarray:
        """Get normalized zone heatmap (3x3)."""
        if player_id not in self._zone_counts:
            return np.zeros((3, 3))

        zones = self._zone_counts[player_id]
        total = zones.sum()
        if total > 0:
            return zones / total
        return zones

    def get_stats(self, player_id: int) -> Optional[PlayerStats]:
        """Get stats for a player."""
        return self._stats.get(player_id)

    def get_all_stats(self) -> Dict[int, PlayerStats]:
        """Get all player stats."""
        return self._stats.copy()

    def get_top_players_by_distance(self, n: int = 5) -> List[PlayerStats]:
        """Get top N players by distance covered."""
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.total_distance_m,
            reverse=True
        )
        return sorted_stats[:n]

    def get_top_players_by_speed(self, n: int = 5) -> List[PlayerStats]:
        """Get top N players by max speed."""
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.max_speed_kmh,
            reverse=True
        )
        return sorted_stats[:n]

    def get_team_stats(self, team: str) -> dict:
        """Get aggregated stats for a team."""
        team_players = [s for s in self._stats.values() if s.team == team]

        if not team_players:
            return {}

        return {
            'total_distance': sum(p.total_distance_m for p in team_players),
            'avg_distance': np.mean([p.total_distance_m for p in team_players]),
            'max_speed': max(p.max_speed_kmh for p in team_players),
            'total_sprints': sum(p.sprint_count for p in team_players),
            'player_count': len(team_players)
        }

    def reset_player(self, player_id: int):
        """Reset stats for a specific player."""
        if player_id in self._stats:
            del self._stats[player_id]
        if player_id in self._zone_counts:
            del self._zone_counts[player_id]
        if player_id in self._sprint_active:
            del self._sprint_active[player_id]

    def reset_all(self):
        """Reset all analytics."""
        self._stats.clear()
        self._zone_counts.clear()
        self._sprint_active.clear()
        self._sprint_start_distance.clear()
        self._last_movement.clear()

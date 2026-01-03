"""
Speed and Distance Calculator for Football Players.

Calculates player speed (km/h, m/s) and total distance covered
using position tracking data.

Coordinate System:
- Pixel coordinates from video frame
- Optional homography for real-world meters conversion
- Football pitch reference: 105m x 68m (FIFA standard)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import deque
import math


@dataclass
class MovementVector:
    """Movement data between two frames."""
    dx: float  # Horizontal movement (pixels or meters)
    dy: float  # Vertical movement (pixels or meters)
    distance: float  # Total distance
    speed: float  # Speed in units/second
    direction: float  # Direction in degrees (0=right, 90=up)
    frame_delta: int  # Frames between measurements

    @property
    def speed_kmh(self) -> float:
        """Speed in km/h (if distance is in meters)."""
        return self.speed * 3.6

    @property
    def is_moving(self) -> bool:
        """Check if significant movement occurred."""
        return self.speed > 0.5  # Threshold for noise


@dataclass
class PositionRecord:
    """A single position record for a player."""
    frame_num: int
    x: float
    y: float
    timestamp: float  # seconds from video start


class SpeedCalculator:
    """
    Calculate player speed and distance from position tracking.

    Uses a sliding window for smooth speed estimation.

    Attributes:
        fps: Video frames per second
        scale_factor: Pixels to meters conversion (meters per pixel)
        smoothing_window: Number of frames for speed smoothing
    """

    # Football pitch dimensions for reference
    PITCH_LENGTH_M = 105.0  # FIFA standard
    PITCH_WIDTH_M = 68.0

    # Speed thresholds (m/s)
    WALKING_THRESHOLD = 2.0  # Below this = walking
    JOGGING_THRESHOLD = 4.0  # Below this = jogging
    RUNNING_THRESHOLD = 6.0  # Below this = running
    SPRINTING_THRESHOLD = 8.0  # Above this = sprinting

    def __init__(
        self,
        fps: float = 30.0,
        scale_factor: float = 0.1,  # Default: 0.1 meters per pixel
        smoothing_window: int = 5,
        max_history: int = 1000
    ):
        """
        Initialize speed calculator.

        Args:
            fps: Video frames per second
            scale_factor: Meters per pixel (for real-world conversion)
            smoothing_window: Number of frames for speed averaging
            max_history: Maximum position history to keep per player
        """
        self.fps = fps
        self.scale_factor = scale_factor
        self.smoothing_window = smoothing_window
        self.max_history = max_history

        # Position history for each player
        self._positions: Dict[int, deque] = {}

        # Calculated stats
        self._total_distance: Dict[int, float] = {}
        self._max_speed: Dict[int, float] = {}
        self._recent_speeds: Dict[int, deque] = {}

    def update(
        self,
        player_id: int,
        position: Tuple[float, float],
        frame_num: int
    ) -> Optional[MovementVector]:
        """
        Update player position and calculate movement.

        Args:
            player_id: Unique player identifier
            position: (x, y) pixel coordinates
            frame_num: Current frame number

        Returns:
            MovementVector if movement can be calculated, None otherwise
        """
        # Initialize history for new player
        if player_id not in self._positions:
            self._positions[player_id] = deque(maxlen=self.max_history)
            self._total_distance[player_id] = 0.0
            self._max_speed[player_id] = 0.0
            self._recent_speeds[player_id] = deque(maxlen=self.smoothing_window)

        x, y = position
        timestamp = frame_num / self.fps

        # Create position record
        record = PositionRecord(
            frame_num=frame_num,
            x=x,
            y=y,
            timestamp=timestamp
        )

        history = self._positions[player_id]

        # Calculate movement if we have previous position
        movement = None
        if len(history) > 0:
            prev = history[-1]
            movement = self._calculate_movement(prev, record)

            if movement:
                # Update stats
                self._total_distance[player_id] += movement.distance
                self._max_speed[player_id] = max(
                    self._max_speed[player_id],
                    movement.speed
                )
                self._recent_speeds[player_id].append(movement.speed)

        # Add to history
        history.append(record)

        return movement

    def _calculate_movement(
        self,
        prev: PositionRecord,
        curr: PositionRecord
    ) -> Optional[MovementVector]:
        """Calculate movement between two positions."""
        frame_delta = curr.frame_num - prev.frame_num
        time_delta = curr.timestamp - prev.timestamp

        if time_delta <= 0 or frame_delta <= 0:
            return None

        # Pixel displacement
        dx_px = curr.x - prev.x
        dy_px = curr.y - prev.y
        distance_px = math.sqrt(dx_px**2 + dy_px**2)

        # Convert to meters
        dx_m = dx_px * self.scale_factor
        dy_m = dy_px * self.scale_factor
        distance_m = distance_px * self.scale_factor

        # Speed in m/s
        speed = distance_m / time_delta

        # Direction (0 = right, 90 = up, 180 = left, 270 = down)
        direction = math.degrees(math.atan2(-dy_px, dx_px)) % 360

        return MovementVector(
            dx=dx_m,
            dy=dy_m,
            distance=distance_m,
            speed=speed,
            direction=direction,
            frame_delta=frame_delta
        )

    def get_current_speed(self, player_id: int) -> float:
        """Get current (smoothed) speed for player in m/s."""
        if player_id not in self._recent_speeds:
            return 0.0

        speeds = self._recent_speeds[player_id]
        if not speeds:
            return 0.0

        return sum(speeds) / len(speeds)

    def get_current_speed_kmh(self, player_id: int) -> float:
        """Get current speed in km/h."""
        return self.get_current_speed(player_id) * 3.6

    def get_total_distance(self, player_id: int) -> float:
        """Get total distance covered in meters."""
        return self._total_distance.get(player_id, 0.0)

    def get_max_speed(self, player_id: int) -> float:
        """Get maximum speed recorded in m/s."""
        return self._max_speed.get(player_id, 0.0)

    def get_max_speed_kmh(self, player_id: int) -> float:
        """Get maximum speed in km/h."""
        return self.get_max_speed(player_id) * 3.6

    def get_average_speed(self, player_id: int) -> float:
        """Get average speed over entire tracking period in m/s."""
        if player_id not in self._positions:
            return 0.0

        history = self._positions[player_id]
        if len(history) < 2:
            return 0.0

        total_distance = self._total_distance.get(player_id, 0.0)
        total_time = history[-1].timestamp - history[0].timestamp

        if total_time <= 0:
            return 0.0

        return total_distance / total_time

    def get_movement_classification(self, player_id: int) -> str:
        """Classify current movement type."""
        speed = self.get_current_speed(player_id)

        if speed < self.WALKING_THRESHOLD:
            return "standing"
        elif speed < self.JOGGING_THRESHOLD:
            return "walking"
        elif speed < self.RUNNING_THRESHOLD:
            return "jogging"
        elif speed < self.SPRINTING_THRESHOLD:
            return "running"
        else:
            return "sprinting"

    def get_position_history(
        self,
        player_id: int,
        last_n: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """Get position history for drawing trails."""
        if player_id not in self._positions:
            return []

        history = self._positions[player_id]
        positions = [(p.x, p.y) for p in history]

        if last_n:
            return positions[-last_n:]
        return positions

    def reset_player(self, player_id: int):
        """Reset stats for a player."""
        if player_id in self._positions:
            self._positions[player_id].clear()
        self._total_distance[player_id] = 0.0
        self._max_speed[player_id] = 0.0
        if player_id in self._recent_speeds:
            self._recent_speeds[player_id].clear()

    def reset_all(self):
        """Reset all player stats."""
        self._positions.clear()
        self._total_distance.clear()
        self._max_speed.clear()
        self._recent_speeds.clear()

    def set_scale_from_pitch(
        self,
        pitch_corners_px: List[Tuple[float, float]],
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0
    ):
        """
        Calibrate scale factor using visible pitch corners.

        Args:
            pitch_corners_px: List of 4 corner points in pixels
            pitch_length_m: Actual pitch length in meters
            pitch_width_m: Actual pitch width in meters
        """
        if len(pitch_corners_px) < 2:
            return

        # Calculate average pixel distance per meter
        # Using first two points as reference
        p1, p2 = pitch_corners_px[0], pitch_corners_px[1]
        pixel_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Estimate real distance (could be length or width)
        # Use the longer dimension as reference
        estimated_real_dist = max(pitch_length_m, pitch_width_m)

        if pixel_dist > 0:
            self.scale_factor = estimated_real_dist / pixel_dist

    def get_all_stats(self) -> Dict[int, dict]:
        """Get stats for all tracked players."""
        stats = {}
        for player_id in self._positions.keys():
            stats[player_id] = {
                'current_speed_ms': self.get_current_speed(player_id),
                'current_speed_kmh': self.get_current_speed_kmh(player_id),
                'total_distance_m': self.get_total_distance(player_id),
                'max_speed_ms': self.get_max_speed(player_id),
                'max_speed_kmh': self.get_max_speed_kmh(player_id),
                'avg_speed_ms': self.get_average_speed(player_id),
                'movement_type': self.get_movement_classification(player_id)
            }
        return stats

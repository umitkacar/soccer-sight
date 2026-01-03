"""
RADAR View - 2D Top-Down Football Pitch Visualization.

Renders a bird's eye view of the football pitch with:
- Player positions (color-coded by team)
- Movement trails
- Speed indicators
- Jersey numbers

Reference pitch dimensions:
- FIFA standard: 105m x 68m
- Display aspect ratio: ~1.54:1
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class PitchType(Enum):
    """Pitch rendering styles."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    MINIMAL = "minimal"


@dataclass
class PlayerMarker:
    """Player position for radar display."""
    player_id: int
    x: float  # Normalized position (0-1)
    y: float  # Normalized position (0-1)
    team: str  # 'red', 'blue', 'unknown'
    jersey_number: Optional[str] = None
    speed_kmh: float = 0.0
    trail: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.trail is None:
            self.trail = []


class RADARView:
    """
    Generate 2D top-down visualization of football pitch.

    Converts pixel coordinates from video to pitch coordinates
    and renders players on a standardized pitch view.
    """

    # Standard pitch dimensions (meters)
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    # Default radar size
    DEFAULT_WIDTH = 400
    DEFAULT_HEIGHT = 260

    # Colors (BGR)
    GRASS_COLOR = (34, 139, 34)  # Forest green
    LINE_COLOR = (255, 255, 255)  # White
    TEAM_RED_COLOR = (0, 0, 200)  # Red
    TEAM_BLUE_COLOR = (200, 100, 0)  # Cyan/Turquoise
    UNKNOWN_COLOR = (128, 128, 128)  # Gray
    TRAIL_COLOR = (200, 200, 200)  # Light gray

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        pitch_type: PitchType = PitchType.DETAILED,
        source_width: int = 1920,
        source_height: int = 1080,
        show_jersey_numbers: bool = True,
        show_trails: bool = True,
        trail_length: int = 20
    ):
        """
        Initialize RADAR view.

        Args:
            width: Radar image width
            height: Radar image height
            pitch_type: Rendering style
            source_width: Source video width (for coordinate mapping)
            source_height: Source video height
            show_jersey_numbers: Whether to display jersey numbers
            show_trails: Whether to show movement trails
            trail_length: Number of positions to keep in trail
        """
        self.width = width
        self.height = height
        self.pitch_type = pitch_type
        self.source_width = source_width
        self.source_height = source_height
        self.show_jersey_numbers = show_jersey_numbers
        self.show_trails = show_trails
        self.trail_length = trail_length

        # Player trails storage
        self._trails: Dict[int, List[Tuple[float, float]]] = {}

        # Create base pitch image
        self._base_pitch = self._create_pitch()

    def _create_pitch(self) -> np.ndarray:
        """Create base pitch image with markings."""
        # Create grass background
        pitch = np.full((self.height, self.width, 3), self.GRASS_COLOR, dtype=np.uint8)

        # Add grass texture (subtle stripes)
        for i in range(0, self.width, 20):
            shade = 5 if (i // 20) % 2 == 0 else -5
            cv2.rectangle(
                pitch,
                (i, 0),
                (min(i + 20, self.width), self.height),
                tuple(max(0, min(255, c + shade)) for c in self.GRASS_COLOR),
                -1
            )

        if self.pitch_type == PitchType.MINIMAL:
            # Just the outline
            self._draw_outline(pitch)
            return pitch

        # Draw pitch markings
        self._draw_outline(pitch)
        self._draw_center_circle(pitch)
        self._draw_penalty_areas(pitch)
        self._draw_goal_areas(pitch)
        self._draw_center_line(pitch)

        if self.pitch_type == PitchType.DETAILED:
            self._draw_corner_arcs(pitch)
            self._draw_penalty_spots(pitch)

        return pitch

    def _draw_outline(self, pitch: np.ndarray):
        """Draw pitch outline."""
        margin = 10
        cv2.rectangle(
            pitch,
            (margin, margin),
            (self.width - margin, self.height - margin),
            self.LINE_COLOR,
            2
        )

    def _draw_center_circle(self, pitch: np.ndarray):
        """Draw center circle."""
        center_x = self.width // 2
        center_y = self.height // 2
        radius = int(self.height * 0.15)  # ~9.15m relative to pitch width

        cv2.circle(pitch, (center_x, center_y), radius, self.LINE_COLOR, 2)
        cv2.circle(pitch, (center_x, center_y), 3, self.LINE_COLOR, -1)

    def _draw_center_line(self, pitch: np.ndarray):
        """Draw center line."""
        center_x = self.width // 2
        margin = 10
        cv2.line(
            pitch,
            (center_x, margin),
            (center_x, self.height - margin),
            self.LINE_COLOR,
            2
        )

    def _draw_penalty_areas(self, pitch: np.ndarray):
        """Draw penalty areas (18-yard boxes)."""
        margin = 10
        box_width = int(self.width * 0.16)  # ~16.5m / 105m
        box_height = int(self.height * 0.6)  # ~40.3m / 68m
        box_top = (self.height - box_height) // 2

        # Left penalty area
        cv2.rectangle(
            pitch,
            (margin, box_top),
            (margin + box_width, box_top + box_height),
            self.LINE_COLOR,
            2
        )

        # Right penalty area
        cv2.rectangle(
            pitch,
            (self.width - margin - box_width, box_top),
            (self.width - margin, box_top + box_height),
            self.LINE_COLOR,
            2
        )

    def _draw_goal_areas(self, pitch: np.ndarray):
        """Draw goal areas (6-yard boxes)."""
        margin = 10
        box_width = int(self.width * 0.055)  # ~5.5m / 105m
        box_height = int(self.height * 0.27)  # ~18.3m / 68m
        box_top = (self.height - box_height) // 2

        # Left goal area
        cv2.rectangle(
            pitch,
            (margin, box_top),
            (margin + box_width, box_top + box_height),
            self.LINE_COLOR,
            2
        )

        # Right goal area
        cv2.rectangle(
            pitch,
            (self.width - margin - box_width, box_top),
            (self.width - margin, box_top + box_height),
            self.LINE_COLOR,
            2
        )

    def _draw_corner_arcs(self, pitch: np.ndarray):
        """Draw corner arcs."""
        margin = 10
        radius = 5

        # Draw arcs at each corner
        corners = [
            (margin, margin, 0, 90),
            (self.width - margin, margin, 90, 180),
            (self.width - margin, self.height - margin, 180, 270),
            (margin, self.height - margin, 270, 360)
        ]

        for x, y, start, end in corners:
            cv2.ellipse(
                pitch,
                (x, y),
                (radius, radius),
                0,
                start,
                end,
                self.LINE_COLOR,
                1
            )

    def _draw_penalty_spots(self, pitch: np.ndarray):
        """Draw penalty spots."""
        margin = 10
        spot_x_offset = int(self.width * 0.105)  # ~11m / 105m

        # Left penalty spot
        cv2.circle(
            pitch,
            (margin + spot_x_offset, self.height // 2),
            3,
            self.LINE_COLOR,
            -1
        )

        # Right penalty spot
        cv2.circle(
            pitch,
            (self.width - margin - spot_x_offset, self.height // 2),
            3,
            self.LINE_COLOR,
            -1
        )

    def video_to_pitch_coords(
        self,
        x: float,
        y: float
    ) -> Tuple[float, float]:
        """
        Convert video pixel coordinates to pitch coordinates.

        Simple linear mapping - for accuracy, use homography transform.

        Args:
            x: Video x coordinate
            y: Video y coordinate

        Returns:
            (pitch_x, pitch_y) normalized 0-1
        """
        # Simple linear normalization
        pitch_x = x / self.source_width
        pitch_y = y / self.source_height
        return (pitch_x, pitch_y)

    def pitch_to_radar_coords(
        self,
        pitch_x: float,
        pitch_y: float
    ) -> Tuple[int, int]:
        """
        Convert normalized pitch coordinates to radar pixel coordinates.

        Args:
            pitch_x: Normalized x (0-1)
            pitch_y: Normalized y (0-1)

        Returns:
            (radar_x, radar_y) pixel coordinates
        """
        margin = 10
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin

        radar_x = int(margin + pitch_x * usable_width)
        radar_y = int(margin + pitch_y * usable_height)

        return (radar_x, radar_y)

    def update_player_position(
        self,
        player_id: int,
        video_x: float,
        video_y: float
    ):
        """Update player position for trail tracking."""
        pitch_x, pitch_y = self.video_to_pitch_coords(video_x, video_y)

        if player_id not in self._trails:
            self._trails[player_id] = []

        trail = self._trails[player_id]
        trail.append((pitch_x, pitch_y))

        # Trim trail to max length
        if len(trail) > self.trail_length:
            self._trails[player_id] = trail[-self.trail_length:]

    def render(
        self,
        players: List[PlayerMarker],
        show_speed: bool = True
    ) -> np.ndarray:
        """
        Render RADAR view with player positions.

        Args:
            players: List of PlayerMarker objects
            show_speed: Whether to show speed indicators

        Returns:
            BGR image of radar view
        """
        # Start with base pitch
        radar = self._base_pitch.copy()

        # Draw trails first (behind players)
        if self.show_trails:
            for player in players:
                self._draw_trail(radar, player)

        # Draw players
        for player in players:
            self._draw_player(radar, player, show_speed)

        return radar

    def _draw_trail(self, radar: np.ndarray, player: PlayerMarker):
        """Draw movement trail for a player."""
        trail = self._trails.get(player.player_id, [])

        if len(trail) < 2:
            return

        # Convert trail to radar coordinates
        radar_trail = [
            self.pitch_to_radar_coords(x, y)
            for x, y in trail
        ]

        # Draw trail with fading opacity
        for i in range(1, len(radar_trail)):
            alpha = i / len(radar_trail)  # Fade from old to new
            color = tuple(int(c * alpha) for c in self.TRAIL_COLOR)
            cv2.line(
                radar,
                radar_trail[i-1],
                radar_trail[i],
                color,
                1
            )

    def _draw_player(
        self,
        radar: np.ndarray,
        player: PlayerMarker,
        show_speed: bool
    ):
        """Draw a single player marker."""
        # Get radar coordinates
        radar_x, radar_y = self.pitch_to_radar_coords(player.x, player.y)

        # Choose color based on team
        if player.team.lower() == 'red':
            color = self.TEAM_RED_COLOR
        elif player.team.lower() in ['blue', 'turquoise', 'cyan']:
            color = self.TEAM_BLUE_COLOR
        else:
            color = self.UNKNOWN_COLOR

        # Player circle size based on speed
        base_radius = 6
        if show_speed and player.speed_kmh > 20:
            radius = base_radius + 2  # Faster = larger
        else:
            radius = base_radius

        # Draw player circle
        cv2.circle(radar, (radar_x, radar_y), radius, color, -1)
        cv2.circle(radar, (radar_x, radar_y), radius, self.LINE_COLOR, 1)

        # Draw jersey number
        if self.show_jersey_numbers and player.jersey_number:
            text_size = cv2.getTextSize(
                player.jersey_number,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                1
            )[0]

            text_x = radar_x - text_size[0] // 2
            text_y = radar_y - radius - 3

            cv2.putText(
                radar,
                player.jersey_number,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                self.LINE_COLOR,
                1
            )

        # Draw speed indicator (small bar below player)
        if show_speed and player.speed_kmh > 0:
            # Speed bar (0-35 km/h range)
            max_speed = 35.0
            bar_width = int((min(player.speed_kmh, max_speed) / max_speed) * 15)

            if bar_width > 0:
                bar_y = radar_y + radius + 2
                cv2.rectangle(
                    radar,
                    (radar_x - bar_width // 2, bar_y),
                    (radar_x + bar_width // 2, bar_y + 2),
                    (0, 255, 0) if player.speed_kmh > 20 else (0, 200, 200),
                    -1
                )

    def render_with_stats(
        self,
        players: List[PlayerMarker],
        team_stats: Dict[str, dict] = None
    ) -> np.ndarray:
        """
        Render RADAR view with team statistics overlay.

        Args:
            players: List of PlayerMarker objects
            team_stats: Optional team statistics to display

        Returns:
            BGR image with stats overlay
        """
        radar = self.render(players)

        if team_stats:
            # Add stats panel at bottom
            stats_height = 30
            expanded = np.zeros((self.height + stats_height, self.width, 3), dtype=np.uint8)
            expanded[:self.height] = radar
            expanded[self.height:] = (40, 40, 40)  # Dark gray background

            # Display stats
            y_pos = self.height + 20
            x_pos = 10

            for team, stats in team_stats.items():
                color = self.TEAM_RED_COLOR if 'red' in team.lower() else self.TEAM_BLUE_COLOR
                text = f"{team}: {stats.get('total_distance', 0):.0f}m"
                cv2.putText(
                    expanded,
                    text,
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
                x_pos += 150

            return expanded

        return radar

    def clear_trails(self):
        """Clear all movement trails."""
        self._trails.clear()

    def clear_player_trail(self, player_id: int):
        """Clear trail for a specific player."""
        if player_id in self._trails:
            del self._trails[player_id]


def create_radar_view(
    style: str = 'detailed',
    size: str = 'medium',
    source_dimensions: Tuple[int, int] = (1920, 1080)
) -> RADARView:
    """
    Factory function to create RADAR view with presets.

    Args:
        style: 'simple', 'detailed', or 'minimal'
        size: 'small' (300x195), 'medium' (400x260), 'large' (600x390)
        source_dimensions: (width, height) of source video

    Returns:
        Configured RADARView instance
    """
    sizes = {
        'small': (300, 195),
        'medium': (400, 260),
        'large': (600, 390)
    }

    width, height = sizes.get(size, sizes['medium'])

    pitch_types = {
        'simple': PitchType.SIMPLE,
        'detailed': PitchType.DETAILED,
        'minimal': PitchType.MINIMAL
    }

    return RADARView(
        width=width,
        height=height,
        pitch_type=pitch_types.get(style, PitchType.DETAILED),
        source_width=source_dimensions[0],
        source_height=source_dimensions[1]
    )

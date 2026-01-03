"""
Tests for the analytics module (speed/distance calculation).
"""

import pytest
import numpy as np
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMovementVector:
    """Test MovementVector dataclass."""

    def test_movement_vector_creation(self):
        """Test creating a movement vector."""
        from analytics import MovementVector

        mv = MovementVector(
            dx=1.0, dy=2.0, distance=2.236,
            speed=5.0, direction=63.4, frame_delta=1
        )
        assert mv.dx == 1.0
        assert mv.dy == 2.0
        assert mv.speed == 5.0

    def test_movement_vector_speed_kmh(self):
        """Test speed conversion to km/h."""
        from analytics import MovementVector

        mv = MovementVector(
            dx=1.0, dy=0.0, distance=1.0,
            speed=10.0, direction=0, frame_delta=1
        )
        assert mv.speed_kmh == 36.0  # 10 m/s = 36 km/h

    def test_movement_vector_is_moving(self):
        """Test movement threshold detection."""
        from analytics import MovementVector

        # Fast movement
        mv_fast = MovementVector(
            dx=1.0, dy=0.0, distance=1.0,
            speed=2.0, direction=0, frame_delta=1
        )
        assert mv_fast.is_moving is True

        # Slow/static
        mv_slow = MovementVector(
            dx=0.01, dy=0.0, distance=0.01,
            speed=0.1, direction=0, frame_delta=1
        )
        assert mv_slow.is_moving is False


class TestSpeedCalculator:
    """Test SpeedCalculator class."""

    def test_speed_calculator_init(self):
        """Test initialization."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0, scale_factor=0.1)
        assert calc.fps == 30.0
        assert calc.scale_factor == 0.1

    def test_update_new_player(self):
        """Test updating position for new player."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0)
        result = calc.update(player_id=1, position=(100, 100), frame_num=1)

        # First update should return None (no previous position)
        assert result is None

    def test_update_movement(self):
        """Test movement calculation between frames."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0, scale_factor=0.1)

        # First position
        calc.update(player_id=1, position=(100, 100), frame_num=1)

        # Second position (moved 30 pixels right)
        result = calc.update(player_id=1, position=(130, 100), frame_num=2)

        assert result is not None
        assert result.dx == 3.0  # 30 pixels * 0.1 scale = 3 meters
        assert result.dy == 0.0
        assert result.distance == 3.0

    def test_speed_calculation(self):
        """Test speed calculation."""
        from analytics import SpeedCalculator

        # 30 fps, scale 0.1 m/pixel
        calc = SpeedCalculator(fps=30.0, scale_factor=0.1)

        # Move 30 pixels in 1 frame = 3 meters in 1/30 second
        calc.update(player_id=1, position=(0, 0), frame_num=0)
        result = calc.update(player_id=1, position=(30, 0), frame_num=1)

        # Speed should be 3m / (1/30s) = 90 m/s
        assert result is not None
        assert abs(result.speed - 90.0) < 0.1

    def test_direction_calculation(self):
        """Test movement direction calculation."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0)

        # Move right
        calc.update(player_id=1, position=(0, 0), frame_num=0)
        result = calc.update(player_id=1, position=(10, 0), frame_num=1)
        assert abs(result.direction - 0.0) < 1.0  # Right = 0 degrees

        # Move up
        calc.update(player_id=2, position=(0, 10), frame_num=0)
        result = calc.update(player_id=2, position=(0, 0), frame_num=1)
        assert abs(result.direction - 90.0) < 1.0  # Up = 90 degrees

    def test_total_distance(self):
        """Test total distance accumulation."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0, scale_factor=1.0)  # 1 pixel = 1 meter

        calc.update(player_id=1, position=(0, 0), frame_num=0)
        calc.update(player_id=1, position=(10, 0), frame_num=1)
        calc.update(player_id=1, position=(10, 10), frame_num=2)

        total = calc.get_total_distance(player_id=1)
        assert total == 20.0  # 10 + 10 meters

    def test_max_speed(self):
        """Test max speed tracking."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0, scale_factor=0.1)

        calc.update(player_id=1, position=(0, 0), frame_num=0)
        calc.update(player_id=1, position=(10, 0), frame_num=1)  # Slow
        calc.update(player_id=1, position=(60, 0), frame_num=2)  # Fast (50 pixels)
        calc.update(player_id=1, position=(70, 0), frame_num=3)  # Slow again

        max_speed = calc.get_max_speed(player_id=1)
        # 50 pixels * 0.1 = 5m, at 30fps = 5 * 30 = 150 m/s
        assert max_speed == 150.0

    def test_movement_classification(self):
        """Test movement type classification."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0, scale_factor=0.01)

        # Standing
        calc.update(player_id=1, position=(0, 0), frame_num=0)
        calc.update(player_id=1, position=(1, 0), frame_num=1)  # 0.01m in 1/30s = 0.3 m/s

        classification = calc.get_movement_classification(player_id=1)
        assert classification == "standing"

    def test_position_history(self):
        """Test position history retrieval."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0)

        for i in range(5):
            calc.update(player_id=1, position=(i*10, i*5), frame_num=i)

        history = calc.get_position_history(player_id=1)
        assert len(history) == 5
        assert history[0] == (0, 0)
        assert history[-1] == (40, 20)

        # Test last_n
        last_2 = calc.get_position_history(player_id=1, last_n=2)
        assert len(last_2) == 2

    def test_reset_player(self):
        """Test resetting a single player."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0)

        calc.update(player_id=1, position=(0, 0), frame_num=0)
        calc.update(player_id=1, position=(100, 0), frame_num=1)

        assert calc.get_total_distance(player_id=1) > 0

        calc.reset_player(player_id=1)
        assert calc.get_total_distance(player_id=1) == 0.0

    def test_get_all_stats(self):
        """Test getting all player stats."""
        from analytics import SpeedCalculator

        calc = SpeedCalculator(fps=30.0, scale_factor=0.1)

        calc.update(player_id=1, position=(0, 0), frame_num=0)
        calc.update(player_id=1, position=(10, 0), frame_num=1)
        calc.update(player_id=2, position=(100, 100), frame_num=0)
        calc.update(player_id=2, position=(110, 100), frame_num=1)

        stats = calc.get_all_stats()
        assert 1 in stats
        assert 2 in stats
        assert 'current_speed_kmh' in stats[1]
        assert 'total_distance_m' in stats[1]


class TestPlayerStats:
    """Test PlayerStats dataclass."""

    def test_player_stats_creation(self):
        """Test creating player stats."""
        from analytics import PlayerStats

        stats = PlayerStats(
            player_id=1,
            jersey_number="10",
            team="Red"
        )
        assert stats.player_id == 1
        assert stats.jersey_number == "10"
        assert stats.total_distance_m == 0.0

    def test_player_stats_to_dict(self):
        """Test converting stats to dict."""
        from analytics import PlayerStats

        stats = PlayerStats(player_id=1, max_speed_kmh=25.5)
        d = stats.to_dict()

        assert d['player_id'] == 1
        assert d['max_speed_kmh'] == 25.5


class TestPlayerAnalytics:
    """Test PlayerAnalytics class."""

    def test_analytics_init(self):
        """Test analytics initialization."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics(
            frame_width=1920,
            frame_height=1080,
            fps=30.0
        )
        assert analytics.frame_width == 1920
        assert analytics.fps == 30.0

    def test_update_player(self):
        """Test updating player analytics."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics()

        analytics.update(
            player_id=1,
            position=(500, 300),
            speed_ms=5.0,
            total_distance=100.0,
            movement_type="running",
            jersey_number="10"
        )

        stats = analytics.get_stats(player_id=1)
        assert stats is not None
        assert stats.current_speed_kmh == 18.0  # 5 m/s * 3.6
        assert stats.total_distance_m == 100.0
        assert stats.jersey_number == "10"

    def test_time_tracking(self):
        """Test time spent in movement types."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics(fps=30.0)

        # Simulate 30 frames of running (1 second)
        for i in range(30):
            analytics.update(
                player_id=1,
                position=(i*10, 0),
                speed_ms=5.0,
                total_distance=i*0.5,
                movement_type="running"
            )

        stats = analytics.get_stats(player_id=1)
        assert stats.time_running >= 0.9  # ~1 second

    def test_sprint_tracking(self):
        """Test sprint event counting."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics()

        # Start standing
        analytics.update(1, (0, 0), 0.0, 0.0, "standing")

        # Sprint
        for i in range(10):
            analytics.update(1, (i*20, 0), 10.0, i*5.0, "sprinting")

        # Stop
        analytics.update(1, (200, 0), 0.5, 50.0, "standing")

        stats = analytics.get_stats(player_id=1)
        assert stats.sprint_count == 1

    def test_zone_tracking(self):
        """Test zone occupancy tracking."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics(frame_width=300, frame_height=300)

        # Stay in center zone (100-200 x 100-200)
        for _ in range(10):
            analytics.update(1, (150, 150), 1.0, 0.0, "walking")

        zone = analytics.get_zone_name(player_id=1)
        assert "Center" in zone

    def test_team_stats(self):
        """Test team-aggregated stats."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics()

        # Add players to different teams
        analytics.update(1, (0, 0), 5.0, 100.0, "running", team="Red")
        analytics.update(2, (100, 0), 4.0, 80.0, "running", team="Red")
        analytics.update(3, (200, 0), 6.0, 120.0, "running", team="Blue")

        red_stats = analytics.get_team_stats("Red")
        assert red_stats['total_distance'] == 180.0  # 100 + 80
        assert red_stats['player_count'] == 2

    def test_top_players(self):
        """Test getting top players by metric."""
        from analytics import PlayerAnalytics

        analytics = PlayerAnalytics()

        analytics.update(1, (0, 0), 5.0, 100.0, "running")
        analytics.update(2, (0, 0), 4.0, 200.0, "running")
        analytics.update(3, (0, 0), 6.0, 50.0, "running")

        top_distance = analytics.get_top_players_by_distance(n=2)
        assert top_distance[0].player_id == 2  # 200m
        assert top_distance[1].player_id == 1  # 100m


class TestIntegration:
    """Integration tests for analytics module."""

    def test_speed_to_analytics(self):
        """Test SpeedCalculator feeding into PlayerAnalytics."""
        from analytics import SpeedCalculator, PlayerAnalytics

        calc = SpeedCalculator(fps=30.0, scale_factor=0.1)
        analytics = PlayerAnalytics(fps=30.0)

        # Simulate 30 frames of movement
        for frame in range(30):
            movement = calc.update(
                player_id=1,
                position=(frame * 10, 0),
                frame_num=frame
            )

            if movement:
                analytics.update(
                    player_id=1,
                    position=(frame * 10, 0),
                    speed_ms=calc.get_current_speed(1),
                    total_distance=calc.get_total_distance(1),
                    movement_type=calc.get_movement_classification(1)
                )

        # Verify data flows correctly
        stats = analytics.get_stats(player_id=1)
        assert stats is not None
        assert stats.total_distance_m > 0


class TestRadarView:
    """Test RADAR view visualization."""

    def test_radar_view_init(self):
        """Test RADAR view initialization."""
        from analytics import RADARView

        radar = RADARView(width=400, height=260)
        assert radar.width == 400
        assert radar.height == 260

    def test_radar_view_pitch_creation(self):
        """Test pitch image creation."""
        from analytics import RADARView

        radar = RADARView()
        # Should have created base pitch
        assert radar._base_pitch is not None
        assert radar._base_pitch.shape == (260, 400, 3)

    def test_video_to_pitch_coords(self):
        """Test coordinate conversion."""
        from analytics import RADARView

        radar = RADARView(source_width=1920, source_height=1080)

        # Center of video = center of pitch
        px, py = radar.video_to_pitch_coords(960, 540)
        assert abs(px - 0.5) < 0.01
        assert abs(py - 0.5) < 0.01

        # Corner
        px, py = radar.video_to_pitch_coords(0, 0)
        assert px == 0.0
        assert py == 0.0

    def test_pitch_to_radar_coords(self):
        """Test pitch to radar conversion."""
        from analytics import RADARView

        radar = RADARView(width=400, height=260)

        # Center should be near center
        rx, ry = radar.pitch_to_radar_coords(0.5, 0.5)
        assert 180 < rx < 220
        assert 120 < ry < 140

    def test_render_empty(self):
        """Test rendering with no players."""
        from analytics import RADARView

        radar = RADARView()
        image = radar.render([])

        assert image is not None
        assert image.shape == (260, 400, 3)

    def test_render_with_players(self):
        """Test rendering with players."""
        from analytics import RADARView, PlayerMarker

        radar = RADARView()

        players = [
            PlayerMarker(player_id=1, x=0.3, y=0.5, team='red', jersey_number='10'),
            PlayerMarker(player_id=2, x=0.7, y=0.5, team='blue', jersey_number='7'),
        ]

        image = radar.render(players)

        assert image is not None
        assert image.shape == (260, 400, 3)

    def test_player_marker_creation(self):
        """Test PlayerMarker dataclass."""
        from analytics import PlayerMarker

        marker = PlayerMarker(
            player_id=1,
            x=0.5,
            y=0.5,
            team='red',
            jersey_number='10',
            speed_kmh=25.0
        )

        assert marker.player_id == 1
        assert marker.jersey_number == '10'
        assert marker.speed_kmh == 25.0

    def test_update_player_position(self):
        """Test trail tracking."""
        from analytics import RADARView

        radar = RADARView(trail_length=5)

        for i in range(10):
            radar.update_player_position(1, i * 100, 500)

        # Should have 5 positions (max trail length)
        assert len(radar._trails[1]) == 5

    def test_clear_trails(self):
        """Test clearing trails."""
        from analytics import RADARView

        radar = RADARView()
        radar.update_player_position(1, 100, 100)
        radar.update_player_position(2, 200, 200)

        radar.clear_trails()
        assert len(radar._trails) == 0

    def test_pitch_types(self):
        """Test different pitch rendering styles."""
        from analytics import RADARView, PitchType

        for pitch_type in PitchType:
            radar = RADARView(pitch_type=pitch_type)
            assert radar._base_pitch is not None

    def test_create_radar_view_factory(self):
        """Test factory function."""
        from analytics import create_radar_view

        radar = create_radar_view(style='minimal', size='small')
        assert radar.width == 300
        assert radar.height == 195

    def test_render_with_stats(self):
        """Test rendering with team stats."""
        from analytics import RADARView, PlayerMarker

        radar = RADARView()
        players = [
            PlayerMarker(player_id=1, x=0.5, y=0.5, team='red')
        ]

        team_stats = {
            'Red': {'total_distance': 1500.0},
            'Blue': {'total_distance': 1450.0}
        }

        image = radar.render_with_stats(players, team_stats)

        # Should be taller due to stats panel
        assert image.shape[0] > 260


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

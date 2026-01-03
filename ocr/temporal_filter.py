"""
Temporal Consistency Filter for Jersey Number Recognition.

This filter maintains a history of OCR results per player and
ensures temporal consistency across frames, reducing flickering
and improving accuracy through temporal voting.
"""

from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass, field
from .base import OCRResult


@dataclass
class PlayerOCRHistory:
    """Maintains OCR history for a single player."""
    player_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    locked_number: Optional[str] = None
    locked_confidence: float = 0.0
    lock_count: int = 0  # Frames since locked


class TemporalConsistencyFilter:
    """
    Filters OCR results using temporal consistency.

    Maintains a sliding window of recent OCR results per player
    and requires consistent detection before locking a number.

    Features:
    - Sliding window voting across frames
    - Confidence-weighted temporal voting
    - Hysteresis to prevent flickering
    - Lock mechanism for stable numbers
    """

    def __init__(
        self,
        window_size: int = 60,
        lock_threshold: int = 7,
        unlock_threshold: int = 10,
        min_confidence: float = 0.4,
        confidence_decay: float = 0.95,
        majority_threshold: float = 0.6
    ):
        """
        Initialize temporal filter with majority voting for fisheye cameras.

        Args:
            window_size: Number of frames to consider (60 = 2 sec at 30fps)
            lock_threshold: Consistent detections needed to lock
            unlock_threshold: Consecutive mismatches to unlock
            min_confidence: Minimum confidence to consider
            confidence_decay: Decay factor for older results
            majority_threshold: Required vote percentage to lock (0.6 = 60%)
        """
        self.window_size = window_size
        self.lock_threshold = lock_threshold
        self.unlock_threshold = unlock_threshold
        self.min_confidence = min_confidence
        self.confidence_decay = confidence_decay
        self.majority_threshold = majority_threshold

        # Per-player history
        self.players: Dict[int, PlayerOCRHistory] = {}

    def get_or_create_player(self, player_id: int) -> PlayerOCRHistory:
        """Get or create player history."""
        if player_id not in self.players:
            self.players[player_id] = PlayerOCRHistory(
                player_id=player_id,
                history=deque(maxlen=self.window_size)
            )
        return self.players[player_id]

    def update(self, player_id: int, result: OCRResult) -> OCRResult:
        """
        Update filter with new OCR result and return filtered result.

        Args:
            player_id: Player slot ID (1-8)
            result: Raw OCR result from engine

        Returns:
            Filtered OCRResult (may be different from input)
        """
        player = self.get_or_create_player(player_id)

        # Add to history
        player.history.append(result)

        # If player is locked, check if we should unlock
        if player.locked_number is not None:
            return self._handle_locked_player(player, result)

        # Not locked - try to achieve consensus
        return self._try_lock(player, result)

    def _handle_locked_player(
        self,
        player: PlayerOCRHistory,
        result: OCRResult
    ) -> OCRResult:
        """Handle OCR result for a player with locked number."""
        player.lock_count += 1

        # Check if new result matches locked number
        if result.is_valid_jersey() and result.text == player.locked_number:
            # Match - reinforce lock
            player.locked_confidence = min(
                1.0,
                player.locked_confidence * 0.9 + result.confidence * 0.1
            )
            return OCRResult(
                text=player.locked_number,
                confidence=player.locked_confidence,
                raw_text=f"locked ({player.lock_count} frames)",
                engine="temporal_filter"
            )

        # Check for consistent mismatch (possible ID swap)
        if result.is_valid_jersey() and result.confidence >= self.min_confidence:
            # Count recent mismatches
            recent_mismatches = sum(
                1 for r in list(player.history)[-self.unlock_threshold:]
                if r.is_valid_jersey() and r.text == result.text
            )

            if recent_mismatches >= self.unlock_threshold:
                # Unlock and switch to new number
                old_number = player.locked_number
                player.locked_number = result.text
                player.locked_confidence = result.confidence
                player.lock_count = 0
                return OCRResult(
                    text=result.text,
                    confidence=result.confidence,
                    raw_text=f"switched from {old_number}",
                    engine="temporal_filter"
                )

        # Keep locked number (ignore transient misreads)
        return OCRResult(
            text=player.locked_number,
            confidence=player.locked_confidence * 0.99,  # Slight decay
            raw_text=f"locked, ignoring noise",
            engine="temporal_filter"
        )

    def _try_lock(self, player: PlayerOCRHistory, result: OCRResult) -> OCRResult:
        """Try to achieve consensus and lock a number using majority voting."""
        if len(player.history) < self.lock_threshold:
            # Not enough history
            return result

        # Collect recent valid results
        valid_results = [
            r for r in player.history
            if r.is_valid_jersey() and r.confidence >= self.min_confidence
        ]

        if len(valid_results) < self.lock_threshold:
            return result

        # Count occurrences with confidence weighting
        weighted_votes: Dict[str, float] = {}
        vote_counts: Dict[str, int] = {}

        for i, r in enumerate(valid_results):
            # Apply temporal decay (recent results weighted more)
            decay = self.confidence_decay ** (len(valid_results) - 1 - i)
            weight = r.confidence * decay

            weighted_votes[r.text] = weighted_votes.get(r.text, 0) + weight
            vote_counts[r.text] = vote_counts.get(r.text, 0) + 1

        # Find candidate with most votes
        if not vote_counts:
            return result

        best_number = max(vote_counts, key=lambda k: (vote_counts[k], weighted_votes[k]))
        best_count = vote_counts[best_number]
        best_weight = weighted_votes[best_number]
        total_valid = len(valid_results)

        # Calculate vote percentage for majority voting
        vote_percentage = best_count / total_valid if total_valid > 0 else 0

        # Check if we can lock - requires BOTH count threshold AND majority
        if best_count >= self.lock_threshold and vote_percentage >= self.majority_threshold:
            # Lock the number - majority agrees!
            player.locked_number = best_number
            player.locked_confidence = best_weight / best_count
            player.lock_count = 0

            return OCRResult(
                text=best_number,
                confidence=player.locked_confidence,
                raw_text=f"locked: {best_count}/{total_valid} votes ({vote_percentage:.0%})",
                engine="temporal_voting"
            )

        # Not enough consistency yet - return current best guess but don't lock
        if best_count >= 3 and vote_percentage >= 0.4:
            # Return tentative result without locking
            return OCRResult(
                text=best_number,
                confidence=best_weight / best_count * 0.8,  # Lower confidence for tentative
                raw_text=f"tentative: {best_count}/{total_valid} ({vote_percentage:.0%})",
                engine="temporal_voting"
            )

        return result

    def get_locked_number(self, player_id: int) -> Optional[str]:
        """Get locked number for a player, if any."""
        if player_id in self.players:
            return self.players[player_id].locked_number
        return None

    def reset_player(self, player_id: int) -> None:
        """Reset history for a player (e.g., when lost)."""
        if player_id in self.players:
            player = self.players[player_id]
            player.history.clear()
            player.locked_number = None
            player.locked_confidence = 0.0
            player.lock_count = 0

    def reset_all(self) -> None:
        """Reset all player histories."""
        self.players.clear()

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            'total_players': len(self.players),
            'locked_players': sum(
                1 for p in self.players.values()
                if p.locked_number is not None
            ),
            'players': {
                pid: {
                    'locked_number': p.locked_number,
                    'confidence': p.locked_confidence,
                    'history_size': len(p.history)
                }
                for pid, p in self.players.items()
            }
        }

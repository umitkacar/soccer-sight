"""
Tests for OCR module.
"""

import pytest
import numpy as np


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_ocr_result_defaults(self):
        """OCRResult should have sensible defaults."""
        from ocr.base import OCRResult

        result = OCRResult()
        assert result.text is None
        assert result.confidence == 0.0
        assert result.engine == ""

    def test_is_valid_jersey_valid(self):
        """is_valid_jersey should accept numbers 1-99."""
        from ocr.base import OCRResult

        for num in [1, 10, 23, 99]:
            result = OCRResult(text=str(num))
            assert result.is_valid_jersey() is True

    def test_is_valid_jersey_invalid(self):
        """is_valid_jersey should reject invalid numbers."""
        from ocr.base import OCRResult

        invalid_cases = [None, "", "0", "100", "abc", "-1"]
        for text in invalid_cases:
            result = OCRResult(text=text)
            assert result.is_valid_jersey() is False


class TestOCREngineBase:
    """Tests for base OCR engine functionality."""

    def test_extract_jersey_number_valid(self):
        """extract_jersey_number should extract valid numbers."""
        from ocr.base import OCREngine

        test_cases = [
            ("7", "7"),
            ("23", "23"),
            ("07", "7"),  # Leading zero removed
            ("99", "99"),
            ("1", "1"),
        ]

        for input_text, expected in test_cases:
            result = OCREngine.extract_jersey_number(input_text)
            assert result == expected, f"Failed for {input_text}"

    def test_extract_jersey_number_with_noise(self):
        """extract_jersey_number should handle noise."""
        from ocr.base import OCREngine

        test_cases = [
            ("Player 7", "7"),
            ("#23", "23"),
            ("No.10", "10"),
        ]

        for input_text, expected in test_cases:
            result = OCREngine.extract_jersey_number(input_text)
            assert result == expected, f"Failed for {input_text}"

    def test_extract_jersey_number_invalid(self):
        """extract_jersey_number should return None for invalid."""
        from ocr.base import OCREngine

        # Note: "100" returns "10" (first 2 digits) - this is by design
        invalid_cases = [None, "", "abc"]
        for text in invalid_cases:
            result = OCREngine.extract_jersey_number(text)
            assert result is None, f"Should be None for {text}"

    def test_extract_jersey_truncates_to_two_digits(self):
        """extract_jersey_number should truncate to 2 digits."""
        from ocr.base import OCREngine

        # Numbers > 99 are truncated to first 2 digits
        assert OCREngine.extract_jersey_number("100") == "10"
        assert OCREngine.extract_jersey_number("999") == "99"


class TestTemporalFilter:
    """Tests for temporal consistency filter."""

    def test_filter_initialization(self):
        """Filter should initialize properly."""
        from ocr.temporal_filter import TemporalConsistencyFilter

        filter = TemporalConsistencyFilter()
        assert filter.window_size == 30
        assert filter.lock_threshold == 3

    def test_filter_locks_after_consistent_detections(self):
        """Filter should lock after consistent detections."""
        from ocr.temporal_filter import TemporalConsistencyFilter
        from ocr.base import OCRResult

        filter = TemporalConsistencyFilter(lock_threshold=3)

        # Simulate consistent detections
        for i in range(3):
            result = OCRResult(text="7", confidence=0.9)
            filtered = filter.update(player_id=1, result=result)

        # Should be locked now
        assert filter.get_locked_number(1) == "7"

    def test_filter_ignores_noise_when_locked(self):
        """Filter should ignore transient misreads when locked."""
        from ocr.temporal_filter import TemporalConsistencyFilter
        from ocr.base import OCRResult

        filter = TemporalConsistencyFilter(lock_threshold=2, unlock_threshold=3)

        # Lock number 7
        for _ in range(2):
            filter.update(1, OCRResult(text="7", confidence=0.9))

        # Send noise
        filtered = filter.update(1, OCRResult(text="8", confidence=0.5))

        # Should still return 7
        assert filtered.text == "7"

    def test_filter_reset_player(self):
        """reset_player should clear history."""
        from ocr.temporal_filter import TemporalConsistencyFilter
        from ocr.base import OCRResult

        filter = TemporalConsistencyFilter(lock_threshold=2)

        # Lock number 7
        for _ in range(2):
            filter.update(1, OCRResult(text="7", confidence=0.9))

        assert filter.get_locked_number(1) == "7"

        # Reset
        filter.reset_player(1)
        assert filter.get_locked_number(1) is None


class TestEnsembleEngine:
    """Tests for ensemble OCR engine."""

    def test_ensemble_voting_strategies(self):
        """Ensemble should support different voting strategies."""
        from ocr.ensemble_engine import EnsembleOCREngine

        # Test creation with different strategies
        for strategy in ['majority', 'confidence_weighted', 'highest_confidence']:
            engine = EnsembleOCREngine(voting_strategy=strategy)
            assert engine.voting_strategy == strategy

    def test_ensemble_empty_engines(self):
        """Ensemble should handle empty engine list."""
        from ocr.ensemble_engine import EnsembleOCREngine

        engine = EnsembleOCREngine(engines=[])
        result = engine.initialize()
        assert result is False


class TestOCRFactory:
    """Tests for OCR factory functions."""

    def test_get_available_engines(self):
        """get_available_engines should return known engines."""
        from ocr.factory import get_available_engines

        engines = get_available_engines()
        assert 'easyocr' in engines
        assert 'paddleocr' in engines
        assert 'ensemble' in engines

    def test_create_unknown_engine(self):
        """create_ocr_engine should return None for unknown."""
        from ocr.factory import create_ocr_engine

        result = create_ocr_engine('unknown_engine')
        assert result is None

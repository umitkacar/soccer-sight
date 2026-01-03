"""
Ensemble OCR Engine - Combines multiple OCR engines with voting.

This engine runs multiple OCR engines and combines their results
using confidence-weighted voting for improved accuracy.

Typical ensemble combinations:
- EasyOCR + PaddleOCR: Fast, good baseline
- EasyOCR + MMOCR: Better accuracy
- EasyOCR + PARSeq: Best accuracy
- All four: Maximum reliability
"""

import numpy as np
from typing import Optional, List, Dict
from collections import Counter
from .base import OCREngine, OCRResult


class EnsembleOCREngine(OCREngine):
    """
    Ensemble OCR engine combining multiple engines.

    Uses confidence-weighted voting to determine the final result.
    Can use majority voting or weighted average.
    """

    def __init__(
        self,
        engines: List[OCREngine] = None,
        voting_strategy: str = 'confidence_weighted',
        min_agreement: int = 2,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize ensemble engine.

        Args:
            engines: List of OCR engines to use
            voting_strategy: 'majority', 'confidence_weighted', or 'highest_confidence'
            min_agreement: Minimum engines that must agree for valid result
            confidence_threshold: Minimum confidence for a vote to count
        """
        super().__init__(name="ensemble")
        self.engines = engines or []
        self.voting_strategy = voting_strategy
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold

    def add_engine(self, engine: OCREngine) -> None:
        """Add an engine to the ensemble."""
        self.engines.append(engine)

    def initialize(self) -> bool:
        """Initialize all engines in the ensemble."""
        if not self.engines:
            print("No engines configured for ensemble")
            return False

        success_count = 0
        for engine in self.engines:
            if not engine.is_initialized:
                if engine.initialize():
                    success_count += 1
                else:
                    print(f"Failed to initialize {engine.name}")
            else:
                success_count += 1

        self._initialized = success_count > 0
        print(f"Ensemble initialized with {success_count}/{len(self.engines)} engines")
        return self._initialized

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize jersey number using all engines and vote.

        Args:
            image: BGR or grayscale numpy array

        Returns:
            OCRResult with consensus result
        """
        if not self._initialized or not self.engines:
            return OCRResult(engine=self.name)

        if image is None or image.size == 0:
            return OCRResult(engine=self.name)

        # Collect results from all engines
        results: List[OCRResult] = []
        for engine in self.engines:
            if engine.is_initialized:
                result = engine.recognize(image)
                results.append(result)

        if not results:
            return OCRResult(engine=self.name)

        # Apply voting strategy
        if self.voting_strategy == 'highest_confidence':
            return self._highest_confidence_vote(results)
        elif self.voting_strategy == 'majority':
            return self._majority_vote(results)
        else:  # confidence_weighted
            return self._confidence_weighted_vote(results)

    def _highest_confidence_vote(self, results: List[OCRResult]) -> OCRResult:
        """Simply return the result with highest confidence."""
        valid_results = [r for r in results if r.is_valid_jersey()]

        if valid_results:
            best = max(valid_results, key=lambda r: r.confidence)
            return OCRResult(
                text=best.text,
                confidence=best.confidence,
                raw_text=f"from {best.engine}",
                engine=self.name
            )

        # No valid results, return best confidence anyway
        best = max(results, key=lambda r: r.confidence)
        return OCRResult(
            text=best.text,
            confidence=best.confidence * 0.5,  # Penalize non-consensus
            raw_text=f"no consensus, best from {best.engine}",
            engine=self.name
        )

    def _majority_vote(self, results: List[OCRResult]) -> OCRResult:
        """Use majority voting (each engine gets 1 vote)."""
        valid_results = [r for r in results if r.is_valid_jersey()
                        and r.confidence >= self.confidence_threshold]

        if len(valid_results) < self.min_agreement:
            return OCRResult(
                text=None,
                confidence=0.0,
                raw_text="no consensus",
                engine=self.name
            )

        # Count votes
        votes = Counter(r.text for r in valid_results)
        most_common = votes.most_common(1)[0]
        winner_text, vote_count = most_common

        # Check if winner has minimum agreement
        if vote_count < self.min_agreement:
            return OCRResult(
                text=None,
                confidence=0.0,
                raw_text=f"no consensus (max votes: {vote_count})",
                engine=self.name
            )

        # Calculate average confidence for winner
        winner_confs = [r.confidence for r in valid_results if r.text == winner_text]
        avg_conf = sum(winner_confs) / len(winner_confs)

        # Boost confidence based on agreement
        agreement_boost = vote_count / len(self.engines)
        final_conf = min(1.0, avg_conf * (1 + agreement_boost * 0.2))

        return OCRResult(
            text=winner_text,
            confidence=final_conf,
            raw_text=f"{vote_count}/{len(valid_results)} engines agree",
            engine=self.name
        )

    def _confidence_weighted_vote(self, results: List[OCRResult]) -> OCRResult:
        """
        Use confidence-weighted voting.

        Each engine's vote is weighted by its confidence.
        """
        valid_results = [r for r in results if r.is_valid_jersey()
                        and r.confidence >= self.confidence_threshold]

        if len(valid_results) < self.min_agreement:
            # Fall back to highest confidence
            return self._highest_confidence_vote(results)

        # Accumulate weighted votes
        weighted_votes: Dict[str, float] = {}
        vote_counts: Dict[str, int] = {}

        for result in valid_results:
            text = result.text
            weighted_votes[text] = weighted_votes.get(text, 0) + result.confidence
            vote_counts[text] = vote_counts.get(text, 0) + 1

        # Find winner by weighted score
        winner_text = max(weighted_votes, key=weighted_votes.get)
        winner_weight = weighted_votes[winner_text]
        winner_count = vote_counts[winner_text]

        # Check minimum agreement
        if winner_count < self.min_agreement:
            return OCRResult(
                text=None,
                confidence=0.0,
                raw_text=f"no consensus (max agreement: {winner_count})",
                engine=self.name
            )

        # Normalize confidence
        total_weight = sum(weighted_votes.values())
        normalized_conf = winner_weight / total_weight if total_weight > 0 else 0

        # Boost for high agreement
        agreement_ratio = winner_count / len(valid_results)
        final_conf = min(1.0, normalized_conf * (0.8 + agreement_ratio * 0.4))

        return OCRResult(
            text=winner_text,
            confidence=final_conf,
            raw_text=f"{winner_count}/{len(valid_results)} engines, weight={winner_weight:.2f}",
            engine=self.name
        )

    def get_all_results(self, image: np.ndarray) -> Dict[str, OCRResult]:
        """
        Get results from all engines without voting.

        Useful for debugging and analysis.

        Args:
            image: Input image

        Returns:
            Dict mapping engine name to result
        """
        results = {}
        for engine in self.engines:
            if engine.is_initialized:
                results[engine.name] = engine.recognize(image)
        return results

"""
Base OCR Engine - Abstract class for all OCR implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class OCRResult:
    """Result from OCR recognition."""
    text: Optional[str] = None  # Recognized text (jersey number)
    confidence: float = 0.0  # Recognition confidence (0-1)
    raw_text: Optional[str] = None  # Original OCR output before filtering
    engine: str = ""  # Which engine produced this result
    bbox: Optional[Tuple[int, int, int, int]] = None  # Bounding box if available

    def is_valid_jersey(self) -> bool:
        """Check if result is a valid jersey number (1-99)."""
        if not self.text:
            return False
        try:
            num = int(self.text)
            return 1 <= num <= 99
        except ValueError:
            return False

    def __repr__(self):
        return f"OCRResult(text='{self.text}', conf={self.confidence:.3f}, engine='{self.engine}')"


class OCREngine(ABC):
    """
    Abstract base class for OCR engines.

    All OCR engines must implement the recognize() method.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the OCR engine.

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize text (jersey number) from image.

        Args:
            image: BGR or grayscale numpy array

        Returns:
            OCRResult with recognized text and confidence
        """
        pass

    def recognize_multiple(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Recognize text from multiple images.

        Default implementation processes images sequentially.
        Subclasses can override for batch processing.

        Args:
            images: List of BGR or grayscale numpy arrays

        Returns:
            List of OCRResults
        """
        return [self.recognize(img) for img in images]

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    @staticmethod
    def extract_jersey_number(text: str) -> Optional[str]:
        """
        Extract valid jersey number from OCR text.

        Args:
            text: Raw OCR output

        Returns:
            Jersey number string (1-99) or None if invalid
        """
        if not text:
            return None

        # Extract only digits
        digits = ''.join(c for c in str(text) if c.isdigit())

        if not digits:
            return None

        # Take first 2 digits max
        digits = digits[:2]

        try:
            num = int(digits)
            if 1 <= num <= 99:
                return digits.lstrip('0') or '0'  # Remove leading zeros
        except ValueError:
            pass

        return None

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Common preprocessing for OCR.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Preprocessed image
        """
        import cv2

        if image is None or image.size == 0:
            return image

        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    def __repr__(self):
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"

"""
EasyOCR Engine - Lightweight and fast OCR.

EasyOCR is a good baseline for jersey number recognition.
Accuracy: ~52% on SoccerNet benchmark
Speed: Fast (CPU friendly)
"""

import numpy as np
import cv2
from typing import Optional, List
from .base import OCREngine, OCRResult


class EasyOCREngine(OCREngine):
    """
    EasyOCR-based jersey number recognition.

    Pros:
    - Fast and lightweight
    - Works on CPU
    - Easy to install

    Cons:
    - Lower accuracy than MMOCR/PARSeq
    - Struggles with stylized fonts
    """

    def __init__(self, gpu: bool = False, languages: List[str] = None):
        """
        Initialize EasyOCR engine.

        Args:
            gpu: Whether to use GPU acceleration
            languages: List of language codes (default: ['en'])
        """
        super().__init__(name="easyocr")
        self.gpu = gpu
        self.languages = languages or ['en']
        self.reader = None

    def initialize(self) -> bool:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            self._initialized = True
            return True
        except ImportError:
            print("EasyOCR not installed. Install with: pip install easyocr")
            return False
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            return False

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize jersey number using EasyOCR.

        Args:
            image: BGR or grayscale numpy array

        Returns:
            OCRResult with recognized number
        """
        if not self._initialized or self.reader is None:
            return OCRResult(engine=self.name)

        if image is None or image.size == 0:
            return OCRResult(engine=self.name)

        try:
            # Convert to RGB (EasyOCR expects RGB)
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run OCR
            results = self.reader.readtext(
                rgb_image,
                detail=1,
                paragraph=False,
                allowlist='0123456789'  # Only recognize digits
            )

            # Find best jersey number
            best_number = None
            best_conf = 0.0
            best_raw = None
            best_bbox = None

            for detection in results:
                if len(detection) >= 3:
                    bbox, text, conf = detection

                    # Extract jersey number
                    number = self.extract_jersey_number(text)

                    if number and conf > best_conf:
                        best_number = number
                        best_conf = float(conf)
                        best_raw = text
                        # Convert bbox to tuple
                        if bbox:
                            x_coords = [p[0] for p in bbox]
                            y_coords = [p[1] for p in bbox]
                            best_bbox = (
                                int(min(x_coords)),
                                int(min(y_coords)),
                                int(max(x_coords)),
                                int(max(y_coords))
                            )

            return OCRResult(
                text=best_number,
                confidence=best_conf,
                raw_text=best_raw,
                engine=self.name,
                bbox=best_bbox
            )

        except Exception as e:
            print(f"EasyOCR recognition error: {e}")
            return OCRResult(engine=self.name)

    def recognize_with_preprocessing(
        self,
        image: np.ndarray,
        preprocessed_versions: List[np.ndarray]
    ) -> OCRResult:
        """
        Try OCR on original and preprocessed versions, return best result.

        Args:
            image: Original image
            preprocessed_versions: List of preprocessed images

        Returns:
            Best OCRResult across all versions
        """
        results = []

        # Try original first
        results.append(self.recognize(image))

        # Try preprocessed versions
        for prep in preprocessed_versions:
            results.append(self.recognize(prep))

        # Return result with highest confidence
        valid_results = [r for r in results if r.is_valid_jersey()]

        if valid_results:
            return max(valid_results, key=lambda r: r.confidence)

        # Return best non-valid result if no valid ones
        return max(results, key=lambda r: r.confidence)

"""
PaddleOCR Engine - Good accuracy OCR with multi-language support.

PaddleOCR provides good accuracy and is well-maintained.
Accuracy: ~58% on SoccerNet benchmark
Speed: Medium
"""

import numpy as np
import cv2
from typing import Optional, List
from .base import OCREngine, OCRResult


class PaddleOCREngine(OCREngine):
    """
    PaddleOCR-based jersey number recognition.

    Pros:
    - Good accuracy
    - Multi-language support
    - Well maintained

    Cons:
    - Larger model size
    - Slower than EasyOCR
    """

    def __init__(self, use_gpu: bool = False, lang: str = 'en'):
        """
        Initialize PaddleOCR engine.

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language code (default: 'en')
        """
        super().__init__(name="paddleocr")
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None

    def initialize(self) -> bool:
        """Initialize PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=False,
                use_gpu=self.use_gpu,
                show_log=False
            )
            self._initialized = True
            return True
        except ImportError:
            print("PaddleOCR not installed. Install with: pip install paddleocr")
            return False
        except Exception as e:
            print(f"PaddleOCR initialization failed: {e}")
            return False

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize jersey number using PaddleOCR.

        Args:
            image: BGR or grayscale numpy array

        Returns:
            OCRResult with recognized number
        """
        if not self._initialized or self.ocr is None:
            return OCRResult(engine=self.name)

        if image is None or image.size == 0:
            return OCRResult(engine=self.name)

        try:
            # Convert grayscale to BGR if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Run OCR
            results = self.ocr.ocr(image, cls=False)

            if not results or not isinstance(results, list):
                return OCRResult(engine=self.name)

            best_number = None
            best_conf = 0.0
            best_raw = None
            best_bbox = None

            # Handle different PaddleOCR output formats
            for result in results:
                if result is None:
                    continue

                # New format: list of [bbox, (text, conf)]
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            bbox = item[0] if isinstance(item[0], list) else None
                            text_conf = item[1] if len(item) > 1 else item[0]

                            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                                text, conf = text_conf[0], text_conf[1]
                            elif isinstance(text_conf, str):
                                text, conf = text_conf, 0.5
                            else:
                                continue

                            number = self.extract_jersey_number(text)

                            if number and conf > best_conf:
                                best_number = number
                                best_conf = float(conf)
                                best_raw = text

                                if bbox and len(bbox) >= 4:
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
            print(f"PaddleOCR recognition error: {e}")
            return OCRResult(engine=self.name)

"""
MMOCR Engine - High accuracy scene text recognition.

MMOCR's SAR model provides excellent accuracy for sports jersey numbers.
Accuracy: ~78% on SoccerNet benchmark (SAR model)
Speed: Medium (requires GPU for real-time)

Install: pip install mmocr mmcv mmdet
"""

import numpy as np
import cv2
from typing import Optional, List
from .base import OCREngine, OCRResult


class MMOCREngine(OCREngine):
    """
    MMOCR-based jersey number recognition using SAR model.

    Pros:
    - High accuracy on scene text
    - Multiple model options (SAR, ABINet, etc.)
    - Part of OpenMMLab ecosystem

    Cons:
    - Heavier dependencies (mmcv, mmdet)
    - Requires GPU for real-time performance
    """

    def __init__(
        self,
        det_config: str = None,
        det_weights: str = None,
        rec_config: str = None,
        rec_weights: str = None,
        device: str = 'cpu'
    ):
        """
        Initialize MMOCR engine.

        Args:
            det_config: Detection model config path (or use default)
            det_weights: Detection model weights path (or download)
            rec_config: Recognition model config path (or use default)
            rec_weights: Recognition model weights path (or download)
            device: 'cpu' or 'cuda:0'
        """
        super().__init__(name="mmocr")
        self.det_config = det_config
        self.det_weights = det_weights
        self.rec_config = rec_config
        self.rec_weights = rec_weights
        self.device = device
        self.ocr = None

    def initialize(self) -> bool:
        """Initialize MMOCR with SAR model."""
        try:
            from mmocr.apis import MMOCRInferencer

            # Use default SAR model if no config specified
            self.ocr = MMOCRInferencer(
                det='dbnet',  # Text detection
                rec='sar',    # SAR recognition (good for scene text)
                device=self.device
            )
            self._initialized = True
            return True

        except ImportError:
            print("MMOCR not installed. Install with: pip install mmocr mmcv mmdet")
            return False
        except Exception as e:
            print(f"MMOCR initialization failed: {e}")
            # Try alternative initialization
            try:
                from mmocr.apis import TextRecInferencer
                self.ocr = TextRecInferencer(model='sar', device=self.device)
                self._initialized = True
                return True
            except Exception as e2:
                print(f"MMOCR alternative init also failed: {e2}")
                return False

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize jersey number using MMOCR.

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

            # Run inference
            result = self.ocr(image)

            best_number = None
            best_conf = 0.0
            best_raw = None

            # Parse MMOCR results
            if hasattr(result, 'predictions') and result.predictions:
                for pred in result.predictions:
                    if hasattr(pred, 'rec_texts') and hasattr(pred, 'rec_scores'):
                        for text, score in zip(pred.rec_texts, pred.rec_scores):
                            number = self.extract_jersey_number(text)
                            if number and score > best_conf:
                                best_number = number
                                best_conf = float(score)
                                best_raw = text

            # Alternative result format
            elif isinstance(result, dict):
                texts = result.get('rec_texts', result.get('text', []))
                scores = result.get('rec_scores', result.get('score', []))

                if isinstance(texts, str):
                    texts = [texts]
                if isinstance(scores, (int, float)):
                    scores = [scores]

                for text, score in zip(texts, scores):
                    number = self.extract_jersey_number(text)
                    if number and score > best_conf:
                        best_number = number
                        best_conf = float(score)
                        best_raw = text

            return OCRResult(
                text=best_number,
                confidence=best_conf,
                raw_text=best_raw,
                engine=self.name
            )

        except Exception as e:
            print(f"MMOCR recognition error: {e}")
            return OCRResult(engine=self.name)

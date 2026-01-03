"""
SoccerNet Jersey Number OCR Engine.

Uses PARSeq model fine-tuned on SoccerNet dataset for jersey number recognition.
This is the state-of-the-art solution from CVPR 2024 Workshop.

Reference:
- Paper: "A General Framework for Jersey Number Recognition in Sports Video"
- GitHub: https://github.com/mkoshkina/jersey-number-pipeline
- CVPR 2024 Workshop (CVSports)

Accuracy: ~92% on SoccerNet benchmark (SOTA)
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

# Add jersey-number-pipeline to path for PARSeq imports
PIPELINE_ROOT = Path(__file__).parent.parent / 'jersey-number-pipeline' / 'str' / 'parseq'
if PIPELINE_ROOT.exists():
    sys.path.insert(0, str(PIPELINE_ROOT))

from .base import OCREngine, OCRResult


class SoccerNetOCR(OCREngine):
    """
    SoccerNet fine-tuned PARSeq OCR for jersey numbers.

    This is the BEST available model for football jersey number recognition.
    Trained specifically on SoccerNet dataset with 2853 player tracklets.

    Advantages:
    - Fine-tuned specifically for soccer/football jerseys
    - Handles various jersey styles, lighting, occlusion
    - CVPR 2024 SOTA results
    - Fast inference (GPU/CPU)
    """

    def __init__(
        self,
        weights_path: str = None,
        device: str = None
    ):
        """
        Initialize SoccerNet OCR engine.

        Args:
            weights_path: Path to SoccerNet fine-tuned PARSeq weights
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        super().__init__(name="soccernet")

        # Find weights
        if weights_path is None:
            default_path = Path(__file__).parent.parent / 'models' / 'soccernet_parseq.pt'
            if default_path.exists():
                weights_path = str(default_path)
            else:
                print(f"SoccerNet weights not found at {default_path}")
                return

        self.weights_path = weights_path

        # Auto-detect device
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.transform = None

    def initialize(self) -> bool:
        """Initialize the PARSeq model for jersey number recognition."""
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as T

            print(f"Loading PARSeq model...")

            # Load torch hub pretrained PARSeq architecture
            self.model = torch.hub.load(
                'baudm/parseq',
                'parseq',
                pretrained=True,
                trust_repo=True
            )

            # Try to load SoccerNet fine-tuned weights (92% accuracy on jersey numbers!)
            soccernet_loaded = False
            if self.weights_path and Path(self.weights_path).exists():
                try:
                    print(f"Loading SoccerNet fine-tuned weights from {self.weights_path}...")
                    checkpoint = torch.load(
                        self.weights_path,
                        map_location='cpu',
                        weights_only=False  # Required for PyTorch 2.6+
                    )

                    # Add 'model.' prefix to match torch.hub model structure
                    sn_state_dict = checkpoint['state_dict']
                    new_state_dict = {}
                    for key, value in sn_state_dict.items():
                        new_key = 'model.' + key
                        new_state_dict[new_key] = value

                    # Load SoccerNet weights
                    self.model.load_state_dict(new_state_dict, strict=True)
                    soccernet_loaded = True
                    print(f"✅ SoccerNet fine-tuned weights loaded successfully!")

                except Exception as e:
                    print(f"⚠️ SoccerNet weights loading failed: {e}")
                    print("Falling back to pretrained weights...")

            self.model = self.model.to(self.device)
            self.model.train(False)

            # Setup transform (PARSeq standard: 32x128)
            self.transform = T.Compose([
                T.Resize((32, 128), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])

            self._initialized = True
            mode = "SoccerNet fine-tuned" if soccernet_loaded else "pretrained"
            print(f"PARSeq OCR initialized on {self.device} ({mode})")
            return True

        except Exception as e:
            print(f"PARSeq OCR initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _preprocess(self, image: np.ndarray) -> 'torch.Tensor':
        """
        Preprocess image for PARSeq model.

        Args:
            image: BGR numpy array

        Returns:
            Preprocessed tensor
        """
        from PIL import Image
        import torch

        # Convert BGR to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Upscale small images for better recognition
        h, w = image.shape[:2]
        if h < 64 or w < 128:
            scale = max(64 / h, 128 / w, 2.0)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Apply transform
        tensor = self.transform(pil_image)

        return tensor.unsqueeze(0)

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize jersey number from image.

        Args:
            image: BGR numpy array of torso/jersey crop

        Returns:
            OCRResult with recognized number
        """
        if not self._initialized or self.model is None:
            return OCRResult(engine=self.name)

        if image is None or image.size == 0:
            return OCRResult(engine=self.name)

        try:
            import torch

            # Preprocess
            input_tensor = self._preprocess(image).to(self.device)

            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)

                # Use full model output - let tokenizer handle decoding properly
                # Previous bug: truncating to [:3, :11] caused misreads like "3" -> "11"
                preds = self.model.tokenizer.decode(logits.softmax(-1))

                # Handle both tuple (text, confidence) and single text output
                if isinstance(preds, tuple):
                    preds, confidence = preds
                else:
                    confidence = None

            # Extract text and confidence
            text = preds[0] if preds else ""

            # Handle confidence tensor
            try:
                conf = float(confidence[0].mean()) if confidence is not None else 0.0
            except:
                conf = 0.0

            # Extract jersey number
            number = self.extract_jersey_number(text)

            if number:
                return OCRResult(
                    text=number,
                    confidence=conf,
                    raw_text=text,
                    engine=self.name
                )

            return OCRResult(engine=self.name, raw_text=text)

        except Exception as e:
            print(f"SoccerNet OCR error: {e}")
            return OCRResult(engine=self.name)

    def recognize_with_variants(self, image: np.ndarray) -> OCRResult:
        """
        Recognize with multiple preprocessing variants for better accuracy.

        Args:
            image: BGR numpy array

        Returns:
            Best OCRResult from all variants
        """
        if not self._initialized:
            return OCRResult(engine=self.name)

        best_result = OCRResult(engine=self.name)
        best_conf = 0.0

        # Preprocessing variants
        variants = self._get_variants(image)

        for name, variant in variants:
            result = self.recognize(variant)
            if result.confidence > best_conf and result.is_valid_jersey():
                best_conf = result.confidence
                best_result = result
                best_result.raw_text = f"{result.text} (via {name})"

        return best_result

    def _get_variants(self, image: np.ndarray) -> list:
        """
        Create preprocessing variants for better OCR.

        Returns:
            List of (name, image) tuples
        """
        variants = [("original", image)]

        # 1. CLAHE enhancement
        try:
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                variants.append(("clahe", enhanced))
        except:
            pass

        # 2. Bilateral filter
        try:
            bilateral = cv2.bilateralFilter(image, 9, 75, 75)
            variants.append(("bilateral", bilateral))
        except:
            pass

        # 3. Sharpen
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            variants.append(("sharpen", sharpened))
        except:
            pass

        return variants


def create_soccernet_ocr(weights_path: str = None) -> Optional[SoccerNetOCR]:
    """
    Factory function to create SoccerNet OCR engine.

    Args:
        weights_path: Optional path to weights file

    Returns:
        Initialized SoccerNetOCR or None if initialization fails
    """
    engine = SoccerNetOCR(weights_path=weights_path)
    if engine.initialize():
        return engine
    return None

"""
PARSeq Engine - State-of-the-art scene text recognition.

PARSeq (Permuted Autoregressive Sequence Models) achieves SOTA on multiple benchmarks.
Accuracy: ~85-92% on SoccerNet benchmark (with fine-tuning)
Speed: Fast inference (GPU recommended)

Paper: "Scene Text Recognition with Permuted Autoregressive Sequence Models"
GitHub: https://github.com/baudm/parseq

Install: pip install torch torchvision
         (model downloaded automatically from HuggingFace)
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from .base import OCREngine, OCRResult


def get_preprocessing_variants(img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Create multiple preprocessed versions for better OCR.

    Based on benchmark results:
    - CLAHE: 24 detections (best)
    - Bilateral: 23 detections
    - White threshold: 19 detections
    """
    variants = []

    # 1. Original with 2x upscale
    up2 = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_2x", up2))

    # 2. CLAHE enhanced (best performer!)
    gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))

    # 3. Bilateral filter (2nd best)
    bilateral = cv2.bilateralFilter(up2, 9, 75, 75)
    variants.append(("bilateral", bilateral))

    # 4. White threshold (3rd best - good for white numbers)
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    variants.append(("white_thresh", cv2.cvtColor(white_thresh, cv2.COLOR_GRAY2BGR)))

    return variants


class PARSeqEngine(OCREngine):
    """
    PARSeq-based jersey number recognition.

    State-of-the-art scene text recognition model.

    Pros:
    - Highest accuracy available
    - Fast inference
    - Handles various text styles well

    Cons:
    - Requires PyTorch
    - GPU recommended for real-time
    """

    def __init__(
        self,
        model_name: str = 'parseq',
        device: str = None,
        pretrained: bool = True,
        use_preprocessing: bool = True
    ):
        """
        Initialize PARSeq engine.

        Args:
            model_name: Model variant ('parseq', 'parseq_tiny')
            device: 'cpu', 'cuda', or None (auto-detect)
            pretrained: Use pretrained weights
            use_preprocessing: Use multi-variant preprocessing (85% vs 57% accuracy!)
        """
        super().__init__(name="parseq")
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.transform = None
        self.use_preprocessing = use_preprocessing

        # Auto-detect device
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def initialize(self) -> bool:
        """Initialize PARSeq model from torch hub."""
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as T

            # Load model from torch hub (downloads automatically)
            self.model = torch.hub.load(
                'baudm/parseq',
                self.model_name,
                pretrained=self.pretrained,
                trust_repo=True
            ).to(self.device)

            # Set model to inference mode
            self.model.train(False)

            # Get model's image transform
            self.transform = T.Compose([
                T.Resize((32, 128), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])

            self._initialized = True
            return True

        except ImportError as e:
            print(f"PARSeq dependencies missing: {e}")
            print("Install with: pip install torch torchvision pillow")
            return False
        except Exception as e:
            print(f"PARSeq initialization failed: {e}")
            return False

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize jersey number using PARSeq with preprocessing variants.

        Uses multiple preprocessing techniques (CLAHE, bilateral, threshold)
        to achieve 85% detection rate vs 57% without preprocessing.

        Args:
            image: BGR or grayscale numpy array

        Returns:
            OCRResult with recognized number
        """
        if not self._initialized or self.model is None:
            return OCRResult(engine=self.name)

        if image is None or image.size == 0:
            return OCRResult(engine=self.name)

        try:
            import torch
            from PIL import Image

            # Convert grayscale to BGR if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Get preprocessing variants or just use upscaled original
            if self.use_preprocessing:
                variants = get_preprocessing_variants(image)
            else:
                up2 = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                variants = [("upscale_2x", up2)]

            best_number = None
            best_conf = 0.0
            best_raw = ""
            best_variant = ""

            # Try each preprocessing variant
            for variant_name, variant_img in variants:
                try:
                    # Convert to RGB PIL Image
                    img_rgb = cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)

                    # Transform image
                    img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                    # Run inference
                    with torch.no_grad():
                        logits = self.model(img_tensor)
                        pred = logits.softmax(-1)
                        label, confidence = self.model.tokenizer.decode(pred)

                    # Extract result
                    if label and len(label) > 0:
                        text = label[0]
                        # Handle confidence tensor
                        try:
                            conf = float(confidence[0].mean()) if confidence is not None else 0.0
                        except:
                            conf = 0.0

                        number = self.extract_jersey_number(text)

                        # Keep best result (highest confidence with valid number)
                        if number and conf > best_conf:
                            best_number = number
                            best_conf = conf
                            best_raw = text
                            best_variant = variant_name

                except Exception:
                    continue

            # Return best result
            if best_number:
                return OCRResult(
                    text=best_number,
                    confidence=best_conf,
                    raw_text=f"{best_raw} (via {best_variant})",
                    engine=self.name
                )

            return OCRResult(engine=self.name)

        except Exception as e:
            print(f"PARSeq recognition error: {e}")
            return OCRResult(engine=self.name)

    def recognize_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Batch recognition for efficiency.

        Args:
            images: List of BGR or grayscale numpy arrays

        Returns:
            List of OCRResults
        """
        if not self._initialized or self.model is None:
            return [OCRResult(engine=self.name) for _ in images]

        if not images:
            return []

        try:
            import torch
            from PIL import Image

            # Prepare batch
            tensors = []
            for img in images:
                if img is None or img.size == 0:
                    tensors.append(None)
                    continue

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(img)
                tensors.append(self.transform(pil_image))

            # Filter out None values and track indices
            valid_tensors = [(i, t) for i, t in enumerate(tensors) if t is not None]

            if not valid_tensors:
                return [OCRResult(engine=self.name) for _ in images]

            indices, batch_tensors = zip(*valid_tensors)
            batch = torch.stack(batch_tensors).to(self.device)

            # Run batch inference
            with torch.no_grad():
                logits = self.model(batch)
                pred = logits.softmax(-1)
                labels, confidences = self.model.tokenizer.decode(pred)

            # Build results
            results = [OCRResult(engine=self.name) for _ in images]

            for idx, (label, conf) in zip(indices, zip(labels, confidences)):
                number = self.extract_jersey_number(label)
                results[idx] = OCRResult(
                    text=number,
                    confidence=float(conf),
                    raw_text=label,
                    engine=self.name
                )

            return results

        except Exception as e:
            print(f"PARSeq batch recognition error: {e}")
            return [OCRResult(engine=self.name) for _ in images]

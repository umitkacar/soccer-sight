#!/usr/bin/env python3
"""
TrOCR Benchmark with Preprocessing Variants.

Tests multiple preprocessing techniques to improve TrOCR performance.
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

def get_preprocessing_variants(img_bgr):
    """Create multiple preprocessed versions of an image."""
    variants = []
    h, w = img_bgr.shape[:2]

    # 1. Original with 2x upscale
    up2 = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_2x", up2))

    # 2. 3x upscale (more detail)
    up3 = cv2.resize(img_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_3x", up3))

    # 3. 4x upscale (maximum detail)
    up4 = cv2.resize(img_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_4x", up4))

    # 4. CLAHE enhanced (contrast)
    gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))

    # 5. Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(up2, -1, kernel)
    variants.append(("sharpened", sharp))

    # 6. Bilateral filter (denoise while keeping edges)
    bilateral = cv2.bilateralFilter(up2, 9, 75, 75)
    variants.append(("bilateral", bilateral))

    # 7. White threshold (for white numbers on dark)
    gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    variants.append(("white_thresh", cv2.cvtColor(white_thresh, cv2.COLOR_GRAY2BGR)))

    # 8. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive_thresh", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # 9. Inverted (for dark numbers on light)
    _, inv_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    variants.append(("inverted", cv2.cvtColor(inv_thresh, cv2.COLOR_GRAY2BGR)))

    # 10. Morphological operations (clean up)
    kernel_morph = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(white_thresh, cv2.MORPH_CLOSE, kernel_morph)
    variants.append(("morphology", cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)))

    return variants


def test_trocr_with_preprocessing(crops_dir: str, max_crops: int = 107):
    """Test TrOCR with multiple preprocessing variants."""
    print("=" * 60)
    print("TrOCR BENCHMARK WITH PREPROCESSING")
    print("=" * 60)

    # Load TrOCR
    print("\n[1/4] Loading TrOCR model...")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch

        model_name = "microsoft/trocr-base-str"
        print(f"  Model: {model_name}")

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"  Device: {device}")
        print("  TrOCR loaded successfully!")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load crops
    print(f"\n[2/4] Loading crops from {crops_dir}...")
    crops_path = Path(crops_dir)
    crop_files = sorted(crops_path.glob("*.jpg"))[:max_crops]
    print(f"  Found {len(crop_files)} crops")

    # Test
    print(f"\n[3/4] Running TrOCR with 10 preprocessing variants...")
    print("-" * 60)

    results = []
    detected_original = 0
    detected_any = 0
    variant_wins = {}
    total_time = 0

    for i, crop_file in enumerate(crop_files):
        img_bgr = cv2.imread(str(crop_file))
        if img_bgr is None:
            continue

        best_number = ""
        best_conf = 0.0
        best_variant = ""
        best_raw = ""

        for variant_name, variant_img in get_preprocessing_variants(img_bgr):
            try:
                start = time.time()

                # Convert to RGB PIL Image
                img_rgb = cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)

                # Process
                pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

                with torch.no_grad():
                    generated_ids = model.generate(pixel_values, max_new_tokens=10)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                elapsed = time.time() - start

                # Extract jersey number (1-2 digits)
                clean = ''.join(c for c in text if c.isdigit())

                if clean and 1 <= len(clean) <= 2:
                    # Track original detection
                    if variant_name == "upscale_2x" and not best_number:
                        detected_original += 1

                    # Use simple heuristic: first valid detection wins
                    # (Could use confidence but TrOCR doesn't expose it easily)
                    if not best_number:
                        best_number = clean
                        best_variant = variant_name
                        best_raw = text

            except Exception as e:
                continue

        if best_number:
            detected_any += 1
            results.append({
                'file': crop_file.name,
                'number': best_number,
                'variant': best_variant,
                'raw': best_raw
            })
            variant_wins[best_variant] = variant_wins.get(best_variant, 0) + 1
            print(f"  {crop_file.name}: #{best_number} (via: {best_variant}, raw: '{best_raw}')")

        if (i + 1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(crop_files)}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    orig_rate = detected_original / len(crop_files) * 100 if crop_files else 0
    any_rate = detected_any / len(crop_files) * 100 if crop_files else 0

    print(f"\n  Original (2x) only: {detected_original}/{len(crop_files)} ({orig_rate:.1f}%)")
    print(f"  With preprocessing: {detected_any}/{len(crop_files)} ({any_rate:.1f}%)")
    print(f"  Improvement: +{any_rate - orig_rate:.1f}%")

    if variant_wins:
        print("\n  Best preprocessing variants:")
        for v, c in sorted(variant_wins.items(), key=lambda x: -x[1]):
            print(f"    {v}: {c} detections")

    if results:
        # Number frequency
        print("\n  Detected numbers:")
        number_counts = {}
        for r in results:
            n = r['number']
            number_counts[n] = number_counts.get(n, 0) + 1

        for num, count in sorted(number_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    #{num}: {count} times")

    return results


if __name__ == "__main__":
    crops_dir = sys.argv[1] if len(sys.argv) > 1 else "ocr_benchmark/crops"
    max_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 107

    test_trocr_with_preprocessing(crops_dir, max_crops)

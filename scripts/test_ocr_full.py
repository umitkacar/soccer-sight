#!/usr/bin/env python3
"""
Full OCR test with preprocessing variants.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def preprocess_variants(img):
    """Create multiple preprocessed versions."""
    variants = [img]  # Original

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    # 2. Threshold (for white numbers on dark)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # 3. Inverted threshold (for dark numbers)
    _, inv_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    variants.append(cv2.cvtColor(inv_thresh, cv2.COLOR_GRAY2BGR))

    # 4. Upscaled 2x
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)

    return variants


def test_with_preprocessing(crops_dir: str, max_crops: int = 107):
    """Test EasyOCR with multiple preprocessing variants."""
    try:
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return original_load(*args, **kwargs)
        torch.load = patched_load

        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("EasyOCR initialized!")

        crops_path = Path(crops_dir)
        crop_files = sorted(crops_path.glob("*.jpg"))[:max_crops]

        print(f"\nTesting {len(crop_files)} crops with preprocessing variants...")
        print("=" * 60)

        detected_original = 0
        detected_any = 0
        results = []

        for crop_file in crop_files:
            img = cv2.imread(str(crop_file))
            if img is None:
                continue

            # Test original
            orig_result = reader.readtext(img, detail=1)
            orig_number = ""
            orig_conf = 0.0
            for bbox, text, c in orig_result:
                clean = ''.join(ch for ch in text if ch.isdigit())
                if clean and 1 <= len(clean) <= 2:
                    orig_number = clean
                    orig_conf = c
                    break

            if orig_number:
                detected_original += 1

            # Test all variants
            best_number = orig_number
            best_conf = orig_conf
            best_variant = "original" if orig_number else ""

            variants = preprocess_variants(img)
            variant_names = ["original", "enhanced", "threshold", "inv_threshold", "upscaled"]

            for variant, name in zip(variants[1:], variant_names[1:]):  # Skip original (already tested)
                ocr_results = reader.readtext(variant, detail=1)
                for bbox, text, c in ocr_results:
                    clean = ''.join(ch for ch in text if ch.isdigit())
                    if clean and 1 <= len(clean) <= 2:
                        if c > best_conf:
                            best_number = clean
                            best_conf = c
                            best_variant = name
                        break

            if best_number:
                detected_any += 1
                results.append({
                    'file': crop_file.name,
                    'number': best_number,
                    'conf': best_conf,
                    'variant': best_variant
                })
                print(f"  {crop_file.name}: #{best_number} (conf: {best_conf:.2f}, via: {best_variant})")

        print("=" * 60)
        print(f"\n📊 RESULTS:")
        print(f"  Original only: {detected_original}/{len(crop_files)} ({detected_original/len(crop_files)*100:.1f}%)")
        print(f"  With preproc:  {detected_any}/{len(crop_files)} ({detected_any/len(crop_files)*100:.1f}%)")

        if results:
            avg_conf = sum(r['conf'] for r in results) / len(results)
            print(f"  Average conf:  {avg_conf:.2f}")

            # Variant breakdown
            print("\n📈 Best variant breakdown:")
            variant_counts = {}
            for r in results:
                v = r['variant']
                variant_counts[v] = variant_counts.get(v, 0) + 1
            for v, c in sorted(variant_counts.items(), key=lambda x: -x[1]):
                print(f"    {v}: {c} detections")

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    crops_dir = sys.argv[1] if len(sys.argv) > 1 else "ocr_benchmark/crops"
    max_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 107
    test_with_preprocessing(crops_dir, max_crops)

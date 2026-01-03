#!/usr/bin/env python3
"""
Simple OCR test on extracted crops.
"""

import cv2
import sys
import os
from pathlib import Path

def test_easyocr(crops_dir: str):
    """Test EasyOCR on crops."""
    try:
        # Workaround for torch.load weights_only issue
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
        crop_files = sorted(crops_path.glob("*.jpg"))[:30]  # Test first 30

        results = []
        detected = 0

        print(f"\nTesting {len(crop_files)} crops...")
        print("-" * 50)

        for crop_file in crop_files:
            img = cv2.imread(str(crop_file))
            if img is None:
                continue

            ocr_results = reader.readtext(img, detail=1)

            # Find jersey numbers (1-2 digits)
            number = ""
            conf = 0.0
            for bbox, text, c in ocr_results:
                clean = ''.join(ch for ch in text if ch.isdigit())
                if clean and 1 <= len(clean) <= 2:
                    number = clean
                    conf = c
                    break

            if number:
                detected += 1
                print(f"  {crop_file.name}: #{number} (conf: {conf:.2f})")
                results.append({'file': crop_file.name, 'number': number, 'conf': conf})
            else:
                # Check if any text was found
                all_text = [t for _, t, _ in ocr_results]
                if all_text:
                    print(f"  {crop_file.name}: No number, found: {all_text}")

        print("-" * 50)
        print(f"\nDetection rate: {detected}/{len(crop_files)} ({detected/len(crop_files)*100:.1f}%)")

        if results:
            avg_conf = sum(r['conf'] for r in results) / len(results)
            print(f"Average confidence: {avg_conf:.2f}")

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    crops_dir = sys.argv[1] if len(sys.argv) > 1 else "ocr_benchmark/crops"
    test_easyocr(crops_dir)

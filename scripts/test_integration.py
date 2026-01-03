#!/usr/bin/env python3
"""
Test PARSeq integration in OCR module.
"""

import sys
sys.path.insert(0, '/home/umit/github-umitkacar/project-13-futbl')

from pathlib import Path
import cv2

def test_parseq_engine():
    """Test PARSeq engine directly."""
    print("=" * 60)
    print("PARSEQ ENGINE INTEGRATION TEST")
    print("=" * 60)

    try:
        from ocr import create_ocr_engine, create_best_available_engine

        # Test create_best_available_engine (should pick parseq)
        print("\n[1/3] Testing create_best_available_engine...")
        engine = create_best_available_engine(
            preferred_order=['parseq', 'easyocr', 'paddleocr']
        )

        if engine:
            print(f"  Engine created: {engine.name}")
            print(f"  Initialized: {engine.is_initialized}")
        else:
            print("  ERROR: No engine created!")
            return False

        # Test on sample crops
        print("\n[2/3] Testing on sample crops...")
        crops_dir = Path("ocr_benchmark/crops")
        crop_files = sorted(crops_dir.glob("*.jpg"))[:10]

        if not crop_files:
            print("  No crops found!")
            return False

        detected = 0
        for crop_file in crop_files:
            img = cv2.imread(str(crop_file))
            if img is None:
                continue

            result = engine.recognize(img)

            if result.text:
                detected += 1
                print(f"  {crop_file.name}: #{result.text} (conf: {result.confidence:.2f})")
            else:
                print(f"  {crop_file.name}: No detection")

        print(f"\n[3/3] Results: {detected}/{len(crop_files)} detected")

        return detected > 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_parseq_engine()
    print("\n" + "=" * 60)
    print(f"TEST {'PASSED' if success else 'FAILED'}")
    print("=" * 60)

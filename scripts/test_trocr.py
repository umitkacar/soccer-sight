#!/usr/bin/env python3
"""
TrOCR Benchmark - State-of-the-art scene text recognition.

TrOCR is similar to PARSeq in architecture (Transformer-based).
microsoft/trocr-base-str is trained on scene text.
"""

import cv2
import sys
import time
from pathlib import Path

def test_trocr(crops_dir: str, max_crops: int = 107):
    """Test TrOCR on extracted crops."""
    print("=" * 60)
    print("TrOCR BENCHMARK (PARSeq-style Transformer OCR)")
    print("=" * 60)

    # Load TrOCR
    print("\n[1/3] Loading TrOCR model...")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch

        # Use scene text recognition model
        model_name = "microsoft/trocr-base-str"
        print(f"  Model: {model_name}")

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Use GPU if available
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
    print(f"\n[2/3] Loading crops from {crops_dir}...")
    crops_path = Path(crops_dir)
    crop_files = sorted(crops_path.glob("*.jpg"))[:max_crops]
    print(f"  Found {len(crop_files)} crops")

    # Test
    print(f"\n[3/3] Running TrOCR on {len(crop_files)} crops...")
    print("-" * 60)

    results = []
    detected = 0
    total_time = 0

    for i, crop_file in enumerate(crop_files):
        img_bgr = cv2.imread(str(crop_file))
        if img_bgr is None:
            continue

        # Upscale 2x for better recognition
        img_bgr = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to RGB PIL Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Run TrOCR
        start = time.time()
        try:
            pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=10)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elapsed = (time.time() - start) * 1000
            total_time += elapsed

            # Extract jersey number (1-2 digits)
            clean = ''.join(c for c in text if c.isdigit())

            if clean and 1 <= len(clean) <= 2:
                detected += 1
                results.append({
                    'file': crop_file.name,
                    'number': clean,
                    'raw': text,
                    'time_ms': elapsed
                })
                print(f"  {crop_file.name}: #{clean} (raw: '{text}', {elapsed:.0f}ms)")
            elif text.strip():
                # Log non-numeric detections
                print(f"  {crop_file.name}: No number, raw: '{text}'")

        except Exception as e:
            print(f"  {crop_file.name}: ERROR - {e}")

        if (i + 1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(crop_files)}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    det_rate = detected / len(crop_files) * 100 if crop_files else 0
    avg_time = total_time / len(crop_files) if crop_files else 0

    print(f"\n  Detection Rate: {detected}/{len(crop_files)} ({det_rate:.1f}%)")
    print(f"  Average Time:   {avg_time:.0f}ms per crop")

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


def compare_easyocr_vs_trocr(crops_dir: str, max_crops: int = 50):
    """Direct comparison of EasyOCR vs TrOCR."""
    print("\n" + "=" * 60)
    print("COMPARISON: EasyOCR vs TrOCR")
    print("=" * 60)

    crops_path = Path(crops_dir)
    crop_files = sorted(crops_path.glob("*.jpg"))[:max_crops]

    # Initialize both
    print("\nInitializing engines...")

    # EasyOCR
    easyocr_reader = None
    try:
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return original_load(*args, **kwargs)
        torch.load = patched_load

        import easyocr
        easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("  EasyOCR: OK")
    except Exception as e:
        print(f"  EasyOCR: FAILED - {e}")

    # TrOCR
    trocr = None
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-str")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        trocr = {'processor': processor, 'model': model, 'device': device}
        print("  TrOCR: OK")
    except Exception as e:
        print(f"  TrOCR: FAILED - {e}")

    if not easyocr_reader and not trocr:
        print("No engines available!")
        return

    # Compare
    print(f"\nTesting {len(crop_files)} crops...")
    print("-" * 60)

    easy_det = 0
    trocr_det = 0
    both_det = 0
    agreements = 0
    disagreements = []

    for crop_file in crop_files:
        img = cv2.imread(str(crop_file))
        if img is None:
            continue

        img_up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        easy_num = ""
        trocr_num = ""

        # EasyOCR
        if easyocr_reader:
            try:
                results = easyocr_reader.readtext(img_up, detail=1)
                for bbox, text, conf in results:
                    clean = ''.join(c for c in text if c.isdigit())
                    if clean and 1 <= len(clean) <= 2:
                        easy_num = clean
                        break
            except:
                pass

        # TrOCR
        if trocr:
            try:
                from PIL import Image
                import torch

                rgb = cv2.cvtColor(img_up, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pixel_values = trocr['processor'](images=pil_img, return_tensors="pt").pixel_values.to(trocr['device'])

                with torch.no_grad():
                    gen_ids = trocr['model'].generate(pixel_values, max_new_tokens=10)

                text = trocr['processor'].batch_decode(gen_ids, skip_special_tokens=True)[0]
                clean = ''.join(c for c in text if c.isdigit())
                if clean and 1 <= len(clean) <= 2:
                    trocr_num = clean
            except:
                pass

        # Count
        if easy_num:
            easy_det += 1
        if trocr_num:
            trocr_det += 1
        if easy_num and trocr_num:
            both_det += 1
            if easy_num == trocr_num:
                agreements += 1
            else:
                disagreements.append({
                    'file': crop_file.name,
                    'easyocr': easy_num,
                    'trocr': trocr_num
                })

    # Results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    total = len(crop_files)
    print(f"\n  {'Engine':<12} {'Detections':<12} {'Rate':<10}")
    print(f"  {'-'*34}")
    print(f"  {'EasyOCR':<12} {easy_det:<12} {easy_det/total*100:.1f}%")
    print(f"  {'TrOCR':<12} {trocr_det:<12} {trocr_det/total*100:.1f}%")

    if both_det > 0:
        print(f"\n  Both detected: {both_det}")
        print(f"  Agreements:    {agreements} ({agreements/both_det*100:.1f}%)")

    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)}):")
        for d in disagreements[:5]:
            print(f"    {d['file']}: EasyOCR=#{d['easyocr']}, TrOCR=#{d['trocr']}")


if __name__ == "__main__":
    crops_dir = sys.argv[1] if len(sys.argv) > 1 else "ocr_benchmark/crops"
    max_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 107

    # Run TrOCR benchmark
    test_trocr(crops_dir, max_crops)

    # Compare if requested
    if "--compare" in sys.argv:
        compare_easyocr_vs_trocr(crops_dir, min(50, max_crops))

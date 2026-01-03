#!/usr/bin/env python3
"""
PARSeq Benchmark - State-of-the-art scene text recognition.

PARSeq (Permutation Language Model) is currently SOTA on most STR benchmarks.
Uses iterative refinement with permutation language modeling.
"""

import cv2
import sys
import time
from pathlib import Path

def test_parseq(crops_dir: str, max_crops: int = 107):
    """Test PARSeq on extracted crops."""
    print("=" * 60)
    print("PARSeq BENCHMARK (SOTA Scene Text Recognition)")
    print("=" * 60)

    # Load PARSeq
    print("\n[1/3] Loading PARSeq model...")
    try:
        import torch
        from PIL import Image
        from torchvision import transforms as T

        # PARSeq from torch hub
        print("  Loading from torch hub...")
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
        parseq.train(False)  # Set to eval mode

        # Force CPU mode
        device = "cpu"
        parseq = parseq.to(device)
        print(f"  Device: {device}")

        # PARSeq uses specific image preprocessing (from strhub)
        # Input: 32x128 grayscale-like normalized image
        img_size = (32, 128)
        img_transform = T.Compose([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

        print("  PARSeq loaded successfully!")

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
    print(f"\n[3/3] Running PARSeq on {len(crop_files)} crops...")
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

        # Run PARSeq
        start = time.time()
        try:
            # Transform and batch
            img_tensor = img_transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                # Get prediction logits
                logits = parseq(img_tensor)
                # Greedy decode
                pred = logits.softmax(-1)
                # Use tokenizer to decode
                label, confidence = parseq.tokenizer.decode(pred)

            text = label[0] if label else ""
            # Confidence is a tensor, just use mean
            try:
                conf = float(confidence[0].mean()) if confidence is not None else 0.0
            except:
                conf = 0.0
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
                    'conf': conf,
                    'time_ms': elapsed
                })
                print(f"  {crop_file.name}: #{clean} (raw: '{text}', conf: {conf:.2f}, {elapsed:.0f}ms)")
            elif text.strip():
                # Log non-numeric detections
                print(f"  {crop_file.name}: No number, raw: '{text}'")

        except Exception as e:
            print(f"  {crop_file.name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            break

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
        avg_conf = sum(r['conf'] for r in results) / len(results)
        print(f"  Average Conf:   {avg_conf:.2f}")

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

    # Run PARSeq benchmark
    test_parseq(crops_dir, max_crops)

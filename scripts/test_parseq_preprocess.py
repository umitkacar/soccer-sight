#!/usr/bin/env python3
"""
PARSeq Benchmark with Preprocessing Variants.
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

def get_preprocessing_variants(img_bgr):
    """Create multiple preprocessed versions."""
    variants = []

    # 1. Original with 2x upscale
    up2 = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_2x", up2))

    # 2. 3x upscale
    up3 = cv2.resize(img_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_3x", up3))

    # 3. CLAHE enhanced
    gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))

    # 4. Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(up2, -1, kernel)
    variants.append(("sharpened", sharp))

    # 5. Bilateral filter
    bilateral = cv2.bilateralFilter(up2, 9, 75, 75)
    variants.append(("bilateral", bilateral))

    # 6. White threshold
    gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    variants.append(("white_thresh", cv2.cvtColor(white_thresh, cv2.COLOR_GRAY2BGR)))

    # 7. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    return variants


def test_parseq_with_preprocessing(crops_dir: str, max_crops: int = 107):
    """Test PARSeq with preprocessing variants."""
    print("=" * 60)
    print("PARSeq + PREPROCESSING BENCHMARK")
    print("=" * 60)

    # Load PARSeq
    print("\n[1/4] Loading PARSeq model...")
    try:
        import torch
        from PIL import Image
        from torchvision import transforms as T

        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
        parseq.train(False)
        device = "cpu"
        parseq = parseq.to(device)

        img_size = (32, 128)
        img_transform = T.Compose([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        print(f"  PARSeq loaded on {device}")

    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Load crops
    print(f"\n[2/4] Loading crops from {crops_dir}...")
    crops_path = Path(crops_dir)
    crop_files = sorted(crops_path.glob("*.jpg"))[:max_crops]
    print(f"  Found {len(crop_files)} crops")

    # Test
    print(f"\n[3/4] Running PARSeq with 7 preprocessing variants...")
    print("-" * 60)

    results = []
    detected_original = 0
    detected_any = 0
    variant_wins = {}

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
                img_rgb = cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                img_tensor = img_transform(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = parseq(img_tensor)
                    pred = logits.softmax(-1)
                    label, confidence = parseq.tokenizer.decode(pred)

                text = label[0] if label else ""
                try:
                    conf = float(confidence[0].mean()) if confidence is not None else 0.0
                except:
                    conf = 0.0

                clean = ''.join(c for c in text if c.isdigit())

                if clean and 1 <= len(clean) <= 2:
                    if variant_name == "upscale_2x":
                        detected_original += 1

                    if conf > best_conf:
                        best_number = clean
                        best_conf = conf
                        best_variant = variant_name
                        best_raw = text

            except Exception as e:
                continue

        if best_number:
            detected_any += 1
            results.append({
                'file': crop_file.name,
                'number': best_number,
                'conf': best_conf,
                'variant': best_variant,
                'raw': best_raw
            })
            variant_wins[best_variant] = variant_wins.get(best_variant, 0) + 1
            print(f"  {crop_file.name}: #{best_number} (conf: {best_conf:.2f}, via: {best_variant})")

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
        avg_conf = sum(r['conf'] for r in results) / len(results)
        print(f"\n  Average Confidence: {avg_conf:.2f}")

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
    test_parseq_with_preprocessing(crops_dir, max_crops)

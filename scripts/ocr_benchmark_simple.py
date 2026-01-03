#!/usr/bin/env python3
"""
Simple OCR Benchmark - No tracking, just detection and OCR comparison.

Usage:
    python scripts/ocr_benchmark_simple.py videos/test_video.mp4 --frames 20
"""

import cv2
import numpy as np
import sys
import os
import time
import json
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_crops_simple(video_path: str, num_frames: int = 20, skip_frames: int = 50):
    """Extract player crops using YOLO without tracking."""
    from ultralytics import YOLO

    print(f"Loading YOLO model...")
    model = YOLO("yolo11l.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames, sampling {num_frames}")

    all_crops = []
    frame_count = 0
    sampled = 0

    while sampled < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Detect persons
        results = model(frame, classes=[0], conf=0.4, verbose=False)[0]

        if len(results.boxes) == 0:
            continue

        crops_in_frame = []
        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            height = y2 - y1
            width = x2 - x1

            # Skip small detections
            if height < 60 or width < 25:
                continue

            # Fixed % crop (8% top, 60% bottom)
            torso_top = y1 + int(height * 0.08)
            torso_bottom = y1 + int(height * 0.60)

            crop = frame[torso_top:torso_bottom, x1:x2]
            if crop.size > 0 and crop.shape[0] >= 20:
                crops_in_frame.append({
                    'frame': frame_count,
                    'idx': i,
                    'crop': crop,
                    'bbox': (x1, y1, x2, y2)
                })

        if crops_in_frame:
            all_crops.extend(crops_in_frame)
            sampled += 1
            print(f"  Frame {frame_count}: {len(crops_in_frame)} players")

    cap.release()
    print(f"Total crops: {len(all_crops)}")
    return all_crops


def run_easyocr(reader, image):
    """Run EasyOCR."""
    try:
        results = reader.readtext(image, detail=1)
        for bbox, text, conf in results:
            clean = ''.join(c for c in text if c.isdigit())
            if clean and 1 <= len(clean) <= 2:
                return clean, conf
        return "", 0.0
    except:
        return "", 0.0


def run_paddleocr(ocr, image):
    """Run PaddleOCR."""
    try:
        result = ocr.ocr(image, cls=True)
        if result and result[0]:
            for line in result[0]:
                text, conf = line[1]
                clean = ''.join(c for c in text if c.isdigit())
                if clean and 1 <= len(clean) <= 2:
                    return clean, conf
        return "", 0.0
    except:
        return "", 0.0


def run_trocr(processor, model, image):
    """Run TrOCR."""
    try:
        from PIL import Image
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=5)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        clean = ''.join(c for c in text if c.isdigit())

        if clean and 1 <= len(clean) <= 2:
            return clean, 0.8
        return "", 0.0
    except Exception as e:
        print(f"TrOCR error: {e}")
        return "", 0.0


def main(video_path: str, num_frames: int = 20, output_dir: str = "ocr_benchmark"):
    print("=" * 60)
    print("OCR ENGINE BENCHMARK (Simple)")
    print("=" * 60)

    # Output setup
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    crops_path = out_path / "crops"
    crops_path.mkdir(exist_ok=True)

    # Extract crops
    print("\n[1/4] Extracting crops...")
    crops = extract_crops_simple(video_path, num_frames)

    if not crops:
        print("No crops extracted!")
        return

    # Initialize engines
    print("\n[2/4] Initializing OCR engines...")
    engines = {}

    # EasyOCR
    try:
        import easyocr
        engines['easyocr'] = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("  EasyOCR: OK")
    except Exception as e:
        print(f"  EasyOCR: FAILED - {e}")

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        os.environ['FLAGS_use_mkldnn'] = '0'
        engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
        print("  PaddleOCR: OK")
    except Exception as e:
        print(f"  PaddleOCR: FAILED - {e}")

    # TrOCR
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print("  Loading TrOCR model (this may take a moment)...")
        trocr_proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
        trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
        engines['trocr'] = {'processor': trocr_proc, 'model': trocr_model}
        print("  TrOCR: OK")
    except Exception as e:
        print(f"  TrOCR: FAILED - {e}")

    if not engines:
        print("No OCR engines available!")
        return

    # Run benchmark
    print(f"\n[3/4] Running OCR on {len(crops)} crops...")

    stats = {name: {'detections': 0, 'total': 0, 'confs': [], 'times': []}
             for name in engines}

    results = []

    for i, crop_data in enumerate(crops):
        crop = crop_data['crop']
        frame_num = crop_data['frame']
        idx = crop_data['idx']

        # Save crop
        crop_file = f"f{frame_num:04d}_p{idx}.jpg"
        cv2.imwrite(str(crops_path / crop_file), crop)

        crop_results = {'frame': frame_num, 'idx': idx, 'file': crop_file}

        # Test each engine
        for name, engine in engines.items():
            start = time.time()

            if name == 'easyocr':
                text, conf = run_easyocr(engine, crop)
            elif name == 'paddleocr':
                text, conf = run_paddleocr(engine, crop)
            elif name == 'trocr':
                text, conf = run_trocr(engine['processor'], engine['model'], crop)
            else:
                continue

            elapsed = (time.time() - start) * 1000

            stats[name]['total'] += 1
            stats[name]['times'].append(elapsed)
            if text:
                stats[name]['detections'] += 1
                stats[name]['confs'].append(conf)

            crop_results[name] = {'text': text, 'conf': conf, 'time_ms': elapsed}

        results.append(crop_results)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(crops)} crops")

    # Print results
    print("\n" + "=" * 60)
    print("[4/4] RESULTS")
    print("=" * 60)

    print(f"\nTotal crops: {len(crops)}")
    print("\n{:<12} {:>12} {:>12} {:>12} {:>8}".format(
        "Engine", "Detection%", "Avg Conf", "Avg Time", "Found"))
    print("-" * 58)

    for name, s in stats.items():
        det_pct = (s['detections'] / s['total'] * 100) if s['total'] > 0 else 0
        avg_conf = np.mean(s['confs']) if s['confs'] else 0
        avg_time = np.mean(s['times']) if s['times'] else 0

        print("{:<12} {:>11.1f}% {:>12.3f} {:>10.1f}ms {:>8}".format(
            name, det_pct, avg_conf, avg_time, s['detections']))

    # Agreement analysis
    print("\n" + "-" * 58)
    print("AGREEMENT ANALYSIS")

    agree = 0
    disagree = 0
    disagree_examples = []

    for r in results:
        detected = {}
        for name in engines:
            if name in r and r[name]['text']:
                detected[name] = r[name]['text']

        if len(detected) >= 2:
            vals = list(detected.values())
            if all(v == vals[0] for v in vals):
                agree += 1
            else:
                disagree += 1
                if len(disagree_examples) < 5:
                    disagree_examples.append({
                        'file': r['file'],
                        'results': detected
                    })

    total_multi = agree + disagree
    if total_multi > 0:
        print(f"\nWhen 2+ engines detect: {agree}/{total_multi} agree ({agree/total_multi*100:.1f}%)")

    if disagree_examples:
        print("\nSample disagreements:")
        for ex in disagree_examples:
            print(f"  {ex['file']}: {ex['results']}")

    # Save results
    results_file = out_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {name: {
                'detection_rate': s['detections'] / s['total'] * 100 if s['total'] > 0 else 0,
                'avg_confidence': float(np.mean(s['confs'])) if s['confs'] else 0,
                'avg_time_ms': float(np.mean(s['times'])) if s['times'] else 0,
                'detections': s['detections'],
                'total': s['total']
            } for name, s in stats.items()},
            'results': results[:100]
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Crops saved to: {crops_path}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Video path")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output", default="ocr_benchmark")
    args = parser.parse_args()

    main(args.video, args.frames, args.output)

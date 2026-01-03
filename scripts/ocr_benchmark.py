#!/usr/bin/env python3
"""
OCR Engine Benchmark Script

Compares different OCR engines on actual player crops from video.
Measures: detection rate, confidence, consistency across frames.

Usage:
    python scripts/ocr_benchmark.py videos/test_video.mp4 --frames 50
"""

import cv2
import numpy as np
import sys
import os
import time
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class OCRResult:
    """Single OCR result."""
    text: str
    confidence: float
    engine: str
    time_ms: float


@dataclass
class CropResult:
    """Results for a single crop across all engines."""
    frame_num: int
    player_id: int
    crop_path: str
    results: Dict[str, OCRResult]


def extract_player_crops(video_path: str, num_frames: int = 50, skip_frames: int = 30):
    """
    Extract player crops from video using YOLO detection.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        skip_frames: Frames to skip between samples

    Returns:
        List of (frame_num, crops) tuples where crops is list of (player_id, crop_image)
    """
    from ultralytics import YOLO
    import supervision as sv

    print(f"Loading YOLO model...")
    model = YOLO("yolo11l.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames, sampling {num_frames} frames")

    # Setup tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )

    results = []
    frame_count = 0
    sampled = 0

    while sampled < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % skip_frames != 0:
            continue

        # Detect
        detections = model(frame, classes=[0], conf=0.3, verbose=False)[0]

        if len(detections.boxes) == 0:
            continue

        # Track
        sv_detections = sv.Detections(
            xyxy=detections.boxes.xyxy.cpu().numpy(),
            confidence=detections.boxes.conf.cpu().numpy(),
            class_id=detections.boxes.cls.cpu().numpy().astype(int)
        )
        tracked = tracker.update_with_detections(sv_detections)

        if len(tracked) == 0:
            continue

        # Extract crops
        crops = []
        for i, (bbox, tracker_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            height = y2 - y1
            width = x2 - x1

            # Skip too small detections
            if height < 50 or width < 20:
                continue

            # Fixed % crop (current strategy)
            torso_top = y1 + int(height * 0.08)
            torso_bottom = y1 + int(height * 0.60)

            crop = frame[torso_top:torso_bottom, x1:x2]

            if crop.size == 0 or crop.shape[0] < 20:
                continue

            crops.append((int(tracker_id), crop))

        if crops:
            results.append((frame_count, crops))
            sampled += 1
            print(f"  Frame {frame_count}: {len(crops)} players detected")

    cap.release()
    print(f"Extracted crops from {sampled} frames")
    return results


def init_ocr_engines():
    """Initialize available OCR engines."""
    engines = {}

    # EasyOCR
    try:
        import easyocr
        engines['easyocr'] = easyocr.Reader(['en'], gpu=True, verbose=False)
        print("  EasyOCR initialized (GPU)")
    except Exception as e:
        try:
            import easyocr
            engines['easyocr'] = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("  EasyOCR initialized (CPU)")
        except Exception as e2:
            print(f"  EasyOCR failed: {e2}")

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("  PaddleOCR initialized")
    except Exception as e:
        print(f"  PaddleOCR failed: {e}")

    # PARSeq (via transformers)
    try:
        from transformers import AutoTokenizer, VisionEncoderDecoderModel, TrOCRProcessor
        # Try PARSeq-style model
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            # Use TrOCR as proxy (similar architecture)
            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
            engines['trocr'] = {'processor': processor, 'model': model}
            print("  TrOCR (PARSeq-style) initialized")
        except Exception as e:
            print(f"  TrOCR failed: {e}")
    except Exception as e:
        print(f"  Transformers OCR failed: {e}")

    return engines


def run_easyocr(engine, image: np.ndarray) -> Tuple[str, float]:
    """Run EasyOCR on image."""
    try:
        results = engine.readtext(image, detail=1)
        if results:
            # Filter for jersey numbers (1-2 digits)
            for bbox, text, conf in results:
                clean = ''.join(c for c in text if c.isdigit())
                if clean and 1 <= len(clean) <= 2:
                    return clean, conf
        return "", 0.0
    except Exception:
        return "", 0.0


def run_paddleocr(engine, image: np.ndarray) -> Tuple[str, float]:
    """Run PaddleOCR on image."""
    try:
        result = engine.ocr(image, cls=True)
        if result and result[0]:
            for line in result[0]:
                text, conf = line[1]
                clean = ''.join(c for c in text if c.isdigit())
                if clean and 1 <= len(clean) <= 2:
                    return clean, conf
        return "", 0.0
    except Exception:
        return "", 0.0


def run_trocr(engine, image: np.ndarray) -> Tuple[str, float]:
    """Run TrOCR on image."""
    try:
        from PIL import Image
        import torch

        processor = engine['processor']
        model = engine['model']

        # Convert to PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Process
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        clean = ''.join(c for c in text if c.isdigit())

        if clean and 1 <= len(clean) <= 2:
            return clean, 0.8  # TrOCR doesn't provide confidence
        return "", 0.0
    except Exception as e:
        return "", 0.0


def run_benchmark(video_path: str, num_frames: int = 50, output_dir: str = "ocr_benchmark"):
    """Run full OCR benchmark."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    crops_path = output_path / "crops"
    crops_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("OCR ENGINE BENCHMARK")
    print("=" * 60)

    # Extract crops
    print("\n[1/3] Extracting player crops from video...")
    frame_crops = extract_player_crops(video_path, num_frames)

    # Initialize engines
    print("\n[2/3] Initializing OCR engines...")
    engines = init_ocr_engines()

    if not engines:
        print("ERROR: No OCR engines available!")
        return

    print(f"\nAvailable engines: {list(engines.keys())}")

    # Run OCR
    print("\n[3/3] Running OCR benchmark...")

    engine_runners = {
        'easyocr': run_easyocr,
        'paddleocr': run_paddleocr,
        'trocr': run_trocr,
    }

    all_results: List[CropResult] = []

    # Stats
    stats = {name: {'detections': 0, 'total': 0, 'confidences': [], 'times': []}
             for name in engines.keys()}

    total_crops = sum(len(crops) for _, crops in frame_crops)
    processed = 0

    for frame_num, crops in frame_crops:
        for player_id, crop in crops:
            processed += 1

            # Save crop for inspection
            crop_filename = f"frame{frame_num:05d}_player{player_id}.jpg"
            crop_path = str(crops_path / crop_filename)
            cv2.imwrite(crop_path, crop)

            # Test each engine
            results = {}

            for engine_name, engine in engines.items():
                runner = engine_runners.get(engine_name)
                if not runner:
                    continue

                start = time.time()
                text, conf = runner(engine, crop)
                elapsed = (time.time() - start) * 1000

                results[engine_name] = OCRResult(
                    text=text,
                    confidence=conf,
                    engine=engine_name,
                    time_ms=elapsed
                )

                # Update stats
                stats[engine_name]['total'] += 1
                stats[engine_name]['times'].append(elapsed)
                if text:
                    stats[engine_name]['detections'] += 1
                    stats[engine_name]['confidences'].append(conf)

            all_results.append(CropResult(
                frame_num=frame_num,
                player_id=player_id,
                crop_path=crop_path,
                results={k: asdict(v) for k, v in results.items()}
            ))

            if processed % 20 == 0:
                print(f"  Processed {processed}/{total_crops} crops...")

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nTotal crops tested: {total_crops}")
    print("\n{:<12} {:>10} {:>10} {:>10} {:>10}".format(
        "Engine", "Detection%", "Avg Conf", "Avg Time", "Count"))
    print("-" * 54)

    for engine_name, s in stats.items():
        det_rate = (s['detections'] / s['total'] * 100) if s['total'] > 0 else 0
        avg_conf = np.mean(s['confidences']) if s['confidences'] else 0
        avg_time = np.mean(s['times']) if s['times'] else 0

        print("{:<12} {:>9.1f}% {:>10.2f} {:>8.1f}ms {:>10}".format(
            engine_name, det_rate, avg_conf, avg_time, s['detections']))

    # Agreement analysis
    print("\n" + "-" * 54)
    print("AGREEMENT ANALYSIS (when multiple engines detect)")

    agreements = 0
    disagreements = 0

    for result in all_results:
        detected = {k: v['text'] for k, v in result.results.items() if v['text']}
        if len(detected) >= 2:
            values = list(detected.values())
            if all(v == values[0] for v in values):
                agreements += 1
            else:
                disagreements += 1
                # Log disagreement
                print(f"  Disagreement frame {result.frame_num}: {detected}")

    total_multi = agreements + disagreements
    if total_multi > 0:
        print(f"\nAgreement rate: {agreements}/{total_multi} ({agreements/total_multi*100:.1f}%)")

    # Save detailed results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'stats': {k: {**v, 'confidences': v['confidences'][:10], 'times': v['times'][:10]}
                     for k, v in stats.items()},
            'sample_results': [asdict(r) if hasattr(r, '__dict__') else r for r in all_results[:50]]
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")
    print(f"Crop images saved to: {crops_path}")

    return stats, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR Engine Benchmark")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to sample")
    parser.add_argument("--output", default="ocr_benchmark", help="Output directory")

    args = parser.parse_args()

    run_benchmark(args.video, args.frames, args.output)

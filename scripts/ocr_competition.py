#!/usr/bin/env python3
"""
OCR Competition Script - ULTRATHINK

Tests ALL OCR engines (active + deprecated) on jersey images.
Winner will be integrated into the main pipeline.

Ground Truth Structure:
    test_images/
    ├── 3/          <- jersey number
    │   ├── jersey_3_i_1.jpeg
    │   └── ...
    ├── 5/
    ├── 7/
    ├── 10/
    └── 19/

Usage:
    python scripts/ocr_competition.py
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix deprecated imports - they expect 'base' in deprecated folder
# but base.py is in ocr/ folder
import importlib.util
ocr_path = Path(__file__).parent.parent / "ocr"

# Create module alias for deprecated imports
if (ocr_path / "base.py").exists():
    # Make ocr.deprecated.base point to ocr.base
    import ocr.base
    sys.modules['ocr.deprecated.base'] = ocr.base


@dataclass
class TestResult:
    """Single test result."""
    engine: str
    preprocessing: str
    image_path: str
    ground_truth: str
    detected: str
    confidence: float
    time_ms: float
    correct: bool


# Preprocessing functions
def preprocess_raw(img: np.ndarray) -> np.ndarray:
    """No preprocessing - return as is."""
    return img


def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)


def preprocess_bilateral(img: np.ndarray) -> np.ndarray:
    """Apply bilateral filter for noise reduction."""
    return cv2.bilateralFilter(img, 9, 75, 75)


def preprocess_upscale(img: np.ndarray) -> np.ndarray:
    """Upscale 2x for better OCR."""
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def preprocess_sharpen(img: np.ndarray) -> np.ndarray:
    """Apply sharpening kernel."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def preprocess_threshold(img: np.ndarray) -> np.ndarray:
    """High contrast black/white threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


PREPROCESSORS = {
    'RAW': preprocess_raw,
    'CLAHE': preprocess_clahe,
    'BILATERAL': preprocess_bilateral,
    'UPSCALE_2X': preprocess_upscale,
    'SHARPEN': preprocess_sharpen,
    'THRESHOLD': preprocess_threshold,
}


def init_engines() -> Dict[str, object]:
    """Initialize all OCR engines."""
    engines = {}

    print("\n" + "=" * 60)
    print("INITIALIZING OCR ENGINES")
    print("=" * 60)

    # 1. SoccerNetOCR (Active)
    try:
        from ocr.soccernet_ocr import SoccerNetOCR
        engine = SoccerNetOCR()
        if engine.initialize():
            engines['SoccerNetOCR'] = engine
            print("  [OK] SoccerNetOCR (ACTIVE - SOTA 92%)")
    except Exception as e:
        print(f"  [FAIL] SoccerNetOCR: {e}")

    # 2. EasyOCR (Deprecated)
    try:
        from ocr.deprecated.easyocr_engine import EasyOCREngine
        engine = EasyOCREngine(gpu=False)
        if engine.initialize():
            engines['EasyOCR'] = engine
            print("  [OK] EasyOCR (Deprecated - 52%)")
    except Exception as e:
        print(f"  [FAIL] EasyOCR: {e}")

    # 3. PARSeq (Deprecated)
    try:
        from ocr.deprecated.parseq_engine import PARSeqEngine
        engine = PARSeqEngine()
        if engine.initialize():
            engines['PARSeq'] = engine
            print("  [OK] PARSeq (Deprecated - 85-92%)")
    except Exception as e:
        print(f"  [FAIL] PARSeq: {e}")

    # 4. PaddleOCR (Deprecated)
    try:
        from ocr.deprecated.paddleocr_engine import PaddleOCREngine
        engine = PaddleOCREngine(use_gpu=False)
        if engine.initialize():
            engines['PaddleOCR'] = engine
            print("  [OK] PaddleOCR (Deprecated - 58%)")
    except Exception as e:
        print(f"  [FAIL] PaddleOCR: {e}")

    # 5. MMOCR (Deprecated)
    try:
        from ocr.deprecated.mmocr_engine import MMOCREngine
        engine = MMOCREngine()
        if engine.initialize():
            engines['MMOCR'] = engine
            print("  [OK] MMOCR (Deprecated - 78%)")
    except Exception as e:
        print(f"  [FAIL] MMOCR: {e}")

    # 6. Direct EasyOCR (as fallback)
    if 'EasyOCR' not in engines:
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            engines['EasyOCR_Direct'] = reader
            print("  [OK] EasyOCR_Direct (Fallback)")
        except Exception as e:
            print(f"  [FAIL] EasyOCR_Direct: {e}")

    print(f"\nTotal engines initialized: {len(engines)}")
    return engines


def run_ocr(engine, engine_name: str, image: np.ndarray) -> Tuple[str, float]:
    """Run OCR on image and return (detected_number, confidence)."""
    try:
        # Handle different engine types
        if hasattr(engine, 'recognize'):
            # OCREngine interface
            result = engine.recognize(image)
            if result and result.text:
                return result.text, result.confidence
            return "", 0.0
        elif hasattr(engine, 'readtext'):
            # Direct EasyOCR
            results = engine.readtext(image, detail=1, allowlist='0123456789')
            if results:
                for bbox, text, conf in results:
                    clean = ''.join(c for c in text if c.isdigit())
                    if clean and 1 <= len(clean) <= 2:
                        return clean, conf
            return "", 0.0
        else:
            return "", 0.0
    except Exception as e:
        return "", 0.0


def load_test_images(base_path: Path) -> List[Tuple[str, str, np.ndarray]]:
    """Load test images with ground truth.

    Returns: List of (ground_truth, image_path, image_array)
    """
    images = []

    for jersey_dir in sorted(base_path.iterdir()):
        if jersey_dir.is_dir() and jersey_dir.name.isdigit():
            ground_truth = jersey_dir.name

            for img_file in sorted(jersey_dir.glob("*.jpeg")) + sorted(jersey_dir.glob("*.jpg")) + sorted(jersey_dir.glob("*.png")):
                img = cv2.imread(str(img_file))
                if img is not None:
                    images.append((ground_truth, str(img_file), img))

    return images


def run_competition(test_images_path: str = "scripts/test_images"):
    """Run the OCR competition."""

    base_path = Path(test_images_path)
    if not base_path.exists():
        base_path = Path(__file__).parent / "test_images"

    print("\n" + "=" * 70)
    print("      OCR COMPETITION - ULTRATHINK")
    print("=" * 70)

    # Load test images
    print("\n[1/3] Loading test images...")
    images = load_test_images(base_path)
    print(f"  Loaded {len(images)} test images")

    # Show ground truth distribution
    gt_counts = defaultdict(int)
    for gt, _, _ in images:
        gt_counts[gt] += 1
    print(f"  Ground truth distribution: {dict(gt_counts)}")

    # Initialize engines
    print("\n[2/3] Initializing OCR engines...")
    engines = init_engines()

    if not engines:
        print("\nERROR: No OCR engines available!")
        return

    # Run competition
    print("\n[3/3] Running OCR competition...")
    print("-" * 70)

    results: List[TestResult] = []

    # Stats per engine
    engine_stats = {name: {
        'correct': 0,
        'total': 0,
        'confidences': [],
        'times': [],
        'by_preprocess': defaultdict(lambda: {'correct': 0, 'total': 0})
    } for name in engines.keys()}

    total_tests = len(images) * len(engines) * len(PREPROCESSORS)
    current = 0

    for ground_truth, img_path, img in images:
        for engine_name, engine in engines.items():
            for preprocess_name, preprocess_fn in PREPROCESSORS.items():
                current += 1

                # Preprocess image
                processed = preprocess_fn(img.copy())

                # Run OCR
                start = time.time()
                detected, confidence = run_ocr(engine, engine_name, processed)
                elapsed_ms = (time.time() - start) * 1000

                # Check correctness
                correct = (detected == ground_truth)

                # Record result
                result = TestResult(
                    engine=engine_name,
                    preprocessing=preprocess_name,
                    image_path=img_path,
                    ground_truth=ground_truth,
                    detected=detected if detected else "-",
                    confidence=confidence,
                    time_ms=elapsed_ms,
                    correct=correct
                )
                results.append(result)

                # Update stats
                stats = engine_stats[engine_name]
                stats['total'] += 1
                stats['times'].append(elapsed_ms)
                if detected:
                    stats['confidences'].append(confidence)
                if correct:
                    stats['correct'] += 1

                # Per-preprocessing stats
                stats['by_preprocess'][preprocess_name]['total'] += 1
                if correct:
                    stats['by_preprocess'][preprocess_name]['correct'] += 1

                # Progress
                if current % 50 == 0:
                    print(f"  Progress: {current}/{total_tests} ({current*100/total_tests:.0f}%)")

    # Print Results
    print("\n" + "=" * 70)
    print("      COMPETITION RESULTS")
    print("=" * 70)

    # Overall ranking
    print("\n### OVERALL RANKING ###")
    print("-" * 60)
    print(f"{'Rank':<5} {'Engine':<20} {'Correct':<10} {'Accuracy':<10} {'Avg Conf':<10}")
    print("-" * 60)

    rankings = []
    for engine_name, stats in engine_stats.items():
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
        rankings.append((engine_name, stats['correct'], stats['total'], accuracy, avg_conf))

    rankings.sort(key=lambda x: (-x[3], -x[4]))  # Sort by accuracy, then confidence

    for rank, (name, correct, total, accuracy, avg_conf) in enumerate(rankings, 1):
        medal = ""
        if rank == 1:
            medal = " [WINNER]"
        elif rank == 2:
            medal = " [2nd]"
        elif rank == 3:
            medal = " [3rd]"
        print(f"{rank:<5} {name:<20} {correct}/{total:<8} {accuracy:>6.1f}% {avg_conf:>9.2f}{medal}")

    # Best preprocessing per engine
    print("\n\n### BEST PREPROCESSING PER ENGINE ###")
    print("-" * 70)

    for engine_name in engines.keys():
        stats = engine_stats[engine_name]
        best_preprocess = None
        best_accuracy = 0

        for prep_name, prep_stats in stats['by_preprocess'].items():
            acc = prep_stats['correct'] / prep_stats['total'] * 100 if prep_stats['total'] > 0 else 0
            if acc > best_accuracy:
                best_accuracy = acc
                best_preprocess = prep_name

        if best_preprocess:
            print(f"  {engine_name:<20}: {best_preprocess:<12} ({best_accuracy:.1f}%)")
        else:
            print(f"  {engine_name:<20}: NO DETECTIONS")

    # Detailed per-jersey results for winner
    winner = rankings[0][0]
    print(f"\n\n### WINNER DETAILS: {winner} ###")
    print("-" * 70)

    winner_results = [r for r in results if r.engine == winner]

    # Group by ground truth
    by_gt = defaultdict(list)
    for r in winner_results:
        by_gt[r.ground_truth].append(r)

    for gt in sorted(by_gt.keys(), key=int):
        gt_results = by_gt[gt]
        correct_count = sum(1 for r in gt_results if r.correct)
        total_count = len(gt_results)

        # Find best detection for this jersey
        best_result = max((r for r in gt_results if r.correct),
                          key=lambda x: x.confidence,
                          default=None)

        status = "" if best_result else " [FAILED]"
        best_prep = best_result.preprocessing if best_result else "-"
        best_conf = best_result.confidence if best_result else 0

        print(f"  Jersey #{gt}: {correct_count}/{total_count} correct | Best: {best_prep} (conf: {best_conf:.2f}){status}")

    # Failed detections
    print("\n\n### FAILED DETECTIONS (All Engines) ###")
    print("-" * 70)

    # Find images where no engine got it right with RAW preprocessing
    for gt, img_path, _ in images:
        raw_results = [r for r in results
                       if r.image_path == img_path and r.preprocessing == 'RAW']
        all_failed = all(not r.correct for r in raw_results)

        if all_failed:
            img_name = Path(img_path).name
            detections = {r.engine: r.detected for r in raw_results}
            print(f"  {img_name} (GT: {gt})")
            for eng, det in detections.items():
                print(f"    {eng}: {det}")

    print("\n" + "=" * 70)
    print(f"  WINNER: {rankings[0][0]} with {rankings[0][3]:.1f}% accuracy")
    print("=" * 70)

    return results, engine_stats


if __name__ == "__main__":
    run_competition()

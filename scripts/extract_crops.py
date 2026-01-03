#!/usr/bin/env python3
"""
Step 1: Extract player crops from video using YOLO.
"""

import cv2
import sys
from pathlib import Path

def main(video_path: str, num_frames: int = 15, skip_frames: int = 50, output_dir: str = "ocr_benchmark"):
    from ultralytics import YOLO

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    crops_path = out_path / "crops"
    crops_path.mkdir(exist_ok=True)

    print(f"Loading YOLO model...")
    model = YOLO("yolo11l.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames")

    frame_count = 0
    sampled = 0
    total_crops = 0

    while sampled < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        results = model(frame, classes=[0], conf=0.4, verbose=False)[0]
        if len(results.boxes) == 0:
            continue

        crops_in_frame = 0
        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            height = y2 - y1
            width = x2 - x1

            if height < 60 or width < 25:
                continue

            # Fixed % crop
            torso_top = y1 + int(height * 0.08)
            torso_bottom = y1 + int(height * 0.60)
            crop = frame[torso_top:torso_bottom, x1:x2]

            if crop.size > 0 and crop.shape[0] >= 20:
                crop_file = f"f{frame_count:04d}_p{i}.jpg"
                cv2.imwrite(str(crops_path / crop_file), crop)
                crops_in_frame += 1
                total_crops += 1

        if crops_in_frame > 0:
            sampled += 1
            print(f"  Frame {frame_count}: {crops_in_frame} crops")

    cap.release()
    print(f"\nTotal: {total_crops} crops saved to {crops_path}/")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/test_video.mp4"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    main(video, frames)

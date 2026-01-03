# Soccer Sight - System Architecture

## Overview

Soccer Sight is a modular computer vision pipeline for real-time football player tracking.

---

## System Components

### 1. Video Input Layer

```
┌─────────────────────────────────────┐
│           Video Source              │
│  (File Upload / Stream / Camera)    │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         VideoCamera Class           │
│  - Frame extraction                 │
│  - FPS control                      │
│  - Seek/pause functionality         │
└─────────────────────────────────────┘
```

### 2. Detection Layer

```
┌─────────────────────────────────────┐
│        YOLO11 Player Detector       │
│         (Local Inference)           │
├─────────────────────────────────────┤
│  Input:  BGR frame (1920x1080)      │
│  Output: List of detections         │
│          - bbox (x1, y1, x2, y2)    │
│          - confidence               │
│          - class_id                 │
└─────────────────────────────────────┘
```

### 3. Tracking Layer

```
┌─────────────────────────────────────┐
│          BoT-SORT Tracker           │
├─────────────────────────────────────┤
│  Features:                          │
│  - Deep appearance features         │
│  - Camera motion compensation       │
│  - Re-identification                │
│                                     │
│  Output:                            │
│  - Track ID (persistent)            │
│  - Bounding box                     │
│  - Track state                      │
└─────────────────────────────────────┘
```

### 4. Recognition Layer

#### 4a. Jersey Number OCR

```
┌─────────────────────────────────────┐
│        SoccerNet OCR Engine         │
├─────────────────────────────────────┤
│  Model: PARSeq (fine-tuned)         │
│  Input: Player crop (BGR)           │
│  Output:                            │
│  - text: "10"                       │
│  - confidence: 0.92                 │
│                                     │
│  Preprocessing:                     │
│  - CLAHE contrast enhancement       │
│  - Jersey region extraction         │
│  - Upscaling (2x)                   │
└─────────────────────────────────────┘
```

#### 4b. Team Classification

```
┌─────────────────────────────────────┐
│      SigLIP Team Classifier         │
├─────────────────────────────────────┤
│  Model: google/siglip-base-patch16  │
│  Pipeline:                          │
│  1. Extract SigLIP embeddings       │
│  2. UMAP dimensionality reduction   │
│  3. K-Means clustering (k=2)        │
│                                     │
│  Output: TEAM_A / TEAM_B / UNKNOWN  │
└─────────────────────────────────────┘
```

### 5. Analytics Layer

```
┌─────────────────────────────────────┐
│         Player Analytics            │
├─────────────────────────────────────┤
│  Modules:                           │
│  - PlayerStats: Per-player metrics  │
│  - SpeedCalculator: Velocity (m/s)  │
│  - RadarView: 2D pitch projection   │
└─────────────────────────────────────┘
```

### 6. Presentation Layer

```
┌─────────────────────────────────────┐
│        Flask Web Dashboard          │
├─────────────────────────────────────┤
│  Features:                          │
│  - MJPEG video streaming            │
│  - Player thumbnail panel           │
│  - Real-time statistics             │
│  - Play/pause/seek controls         │
│                                     │
│  Endpoints:                         │
│  - /video_feed (MJPEG)              │
│  - /status (JSON)                   │
│  - /player_crops (base64)           │
└─────────────────────────────────────┘
```

---

## Data Flow

```
Video Frame
    │
    ├──► YOLO11 Detection
    │         │
    │         ▼
    │    BoT-SORT Tracking
    │         │
    │         ├──► Player Crop Extraction
    │         │         │
    │         │         ├──► SoccerNet OCR ──► Jersey Number
    │         │         │
    │         │         └──► SigLIP ──► Team Assignment
    │         │
    │         └──► Analytics Processing
    │
    └──► Frame Rendering
              │
              ▼
         MJPEG Stream
```

---

## Module Dependencies

```
app.py
  └── camera.py
        ├── detectors/__init__.py
        ├── trackers/tracker_factory.py
        ├── ocr/soccernet_ocr.py
        │     └── ocr/base.py
        ├── team_classifier/siglip_classifier.py
        │     └── team_classifier/base.py
        └── analytics/player_stats.py
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `YOLO_MODEL` | YOLO model path | No (default: yolo11l.pt) |
| `DETECTION_CONFIDENCE` | Detection threshold | No (default: 0.5) |
| `FLASK_SECRET_KEY` | Session secret | No |

### Config Classes

```python
# config.py
class DetectorConfig:
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

class TrackerConfig:
    MAX_AGE = 30
    MIN_HITS = 3

class OCRConfig:
    MODEL_PATH = "models/soccernet_parseq.pt"
    CONFIDENCE_THRESHOLD = 0.3
```

---

## Performance Characteristics

| Component | GPU (RTX 3080) | CPU |
|-----------|----------------|-----|
| Detection | ~15ms | ~100ms |
| Tracking | ~5ms | ~10ms |
| OCR | ~30ms | ~150ms |
| Team Class. | ~50ms | ~200ms |
| **Total** | **~100ms (10 FPS)** | **~460ms (2 FPS)** |

---

## Extensibility

### Adding New Detector

```python
from detectors.base import BaseDetector

class YourDetector(BaseDetector):
    def detect(self, frame):
        # Your implementation
        return detections
```

### Adding New OCR Engine

```python
from ocr.base import OCREngine

class YourOCR(OCREngine):
    def recognize(self, image):
        # Your implementation
        return OCRResult(text, confidence)
```

### Adding New Team Classifier

```python
from team_classifier.base import TeamClassifier

class YourClassifier(TeamClassifier):
    def classify(self, image):
        # Your implementation
        return TeamType.TEAM_A
```

---

## Future Architecture

### Planned Enhancements

1. **Ball Detection Module**
   - Specialized small object detection
   - Trajectory prediction

2. **Event Detection**
   - Pass recognition
   - Shot detection
   - Foul detection

3. **Formation Analysis**
   - Real-time formation detection
   - Tactical pattern recognition

4. **Export Pipeline**
   - SPADL format
   - StatsBomb format
   - Custom JSON/CSV

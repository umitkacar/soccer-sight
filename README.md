# Soccer Sight

**Real-time football player tracking with jersey number recognition and team classification.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![YOLO11](https://img.shields.io/badge/YOLO-11-orange.svg)](https://docs.ultralytics.com/)

---

## Overview

Soccer Sight is a computer vision system that tracks football players in real-time video, recognizes jersey numbers using OCR, and classifies players into teams using deep learning embeddings.

### Key Features

| Feature | Technology | Accuracy |
|---------|------------|----------|
| **Player Detection** | YOLO11 + Roboflow | ~95% |
| **Multi-Object Tracking** | BoT-SORT | Robust |
| **Jersey Number OCR** | SoccerNet PARSeq | 92% |
| **Team Classification** | SigLIP + UMAP | ~95% |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Video Input                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  YOLO11 Player Detection                     │
│                    (Roboflow API)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   BoT-SORT Tracking                          │
│              (Multi-Object Tracking)                         │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   SoccerNet OCR          │    │   SigLIP Classifier      │
│   (Jersey Numbers)       │    │   (Team Assignment)      │
└──────────────────────────┘    └──────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Flask Web Dashboard                         │
│              (MJPEG Streaming + Stats)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Roboflow API key

### Installation

```bash
# Clone repository
git clone https://github.com/umitkacar/soccer-sight.git
cd soccer-sight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_MODEL=football-players-detection-3zvbc/4
```

### Run

```bash
python app.py
```

Open browser: `http://localhost:5000`

---

## Project Structure

```
soccer-sight/
├── app.py                 # Flask application entry
├── camera.py              # Video processing pipeline
├── config.py              # Configuration management
│
├── detectors/             # Player detection
│   └── roboflow_detector.py
│
├── trackers/              # Multi-object tracking
│   └── tracker_factory.py
│
├── ocr/                   # Jersey number recognition
│   ├── soccernet_ocr.py   # Active (92% accuracy)
│   ├── base.py            # OCR interface
│   ├── temporal_filter.py # Temporal smoothing
│   └── deprecated/        # Legacy engines
│
├── team_classifier/       # Team classification
│   ├── siglip_classifier.py  # SigLIP + UMAP (~95%)
│   ├── hsv_classifier.py     # Color-based fallback
│   └── kmeans_classifier.py  # K-Means clustering
│
├── pose/                  # Pose estimation
│   ├── pose_estimator.py
│   └── roi_extractor.py
│
├── analytics/             # Player analytics
│   ├── player_stats.py
│   ├── speed_calculator.py
│   └── radar_view.py
│
├── templates/             # Web UI
│   ├── index.html         # Upload page
│   └── dashboard.html     # Live tracking view
│
├── scripts/               # Utilities
│   ├── ocr_competition.py # OCR benchmark
│   └── test_images/       # Ground truth data
│
├── tests/                 # Unit tests
│
└── docs/                  # Documentation
    ├── roadmap/
    ├── reviews/
    └── architecture/
```

---

## Components

### Detection (YOLO11)

Uses Roboflow-hosted YOLO11 model for player detection:

```python
from detectors.roboflow_detector import RoboflowDetector

detector = RoboflowDetector(
    api_key="your_key",
    model_id="football-players-detection-3zvbc/4"
)
detections = detector.detect(frame)
```

### Tracking (BoT-SORT)

Robust multi-object tracking with re-identification:

```python
from trackers.tracker_factory import create_tracker

tracker = create_tracker("botsort")
tracks = tracker.update(detections, frame)
```

### OCR (SoccerNet)

State-of-the-art jersey number recognition:

```python
from ocr.soccernet_ocr import SoccerNetOCR

ocr = SoccerNetOCR()
ocr.initialize()
result = ocr.recognize(player_crop)
print(f"Jersey: {result.text}, Confidence: {result.confidence}")
```

### Team Classification (SigLIP)

Vision-language model for team assignment:

```python
from team_classifier.siglip_classifier import SigLIPTeamClassifier

classifier = SigLIPTeamClassifier()
classifier.initialize()
classifier.fit(player_crops)  # Learn team clusters
team = classifier.classify(new_crop)  # TEAM_A or TEAM_B
```

---

## OCR Benchmark Results

Tested on 16 jersey images (numbers: 3, 5, 7, 10, 19):

| Engine | Accuracy | Best Preprocessing |
|--------|----------|-------------------|
| **SoccerNet** | 22.9% | BILATERAL |
| EasyOCR | 11.5% | RAW |
| PARSeq | 10.4% | CLAHE |

Run benchmark:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/ocr_competition.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload page |
| `/upload` | POST | Upload video |
| `/dashboard` | GET | Live tracking view |
| `/video_feed` | GET | MJPEG stream |
| `/status` | GET | Current frame stats |
| `/player_crops` | GET | Player thumbnails (base64) |
| `/toggle_play` | POST | Play/pause |
| `/seek` | POST | Seek to frame |
| `/health` | GET | Health check |

---

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black . --check
flake8 .
```

---

## Roadmap

See [docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md) for detailed development plans.

### Upcoming Features

- [ ] Real-time ball tracking
- [ ] Pass detection and visualization
- [ ] Heatmap generation
- [ ] Formation analysis
- [ ] Export to tracking data formats

---

## References

- [SoccerNet Jersey Number Recognition](https://github.com/SoccerNet/sn-jersey)
- [Roboflow Sports](https://github.com/roboflow/sports)
- [SigLIP Vision-Language Model](https://arxiv.org/abs/2303.15343)
- [BoT-SORT Tracker](https://github.com/NirAharon/BoT-SORT)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Dr. Umit Kacar**

- GitHub: [@umitkacar](https://github.com/umitkacar)
- Email: kacarumit.phd@gmail.com

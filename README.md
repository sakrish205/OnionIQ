# OnionIQ — AI-Powered Onion Quality Grading System

> **Smart India Hackathon 2026 · Problem Statement SIH26031**  
> Ministry of Consumer Affairs, Food & Public Distribution

OnionIQ is a real-time onion detection and counting system that runs on a roller conveyor belt. It uses a custom-trained YOLO11 model with ByteTrack for persistent tracking, served through a Streamlit dashboard with smooth MJPEG/WebRTC video streaming.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/sakrish205/OnionIQ.git
cd OnionIQ
pip install -r requirements.txt
```

For GPU (NVIDIA — recommended for 30+ FPS):

```bash
# Check your CUDA version first
nvidia-smi

# Install matching PyTorch (replace cu128 with your version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### 2. Run the dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

### 3. Test without UI (fastest way to validate your model)

```bash
python test_detection.py
```

Press `Q` to quit, `SPACE` to pause, `S` for slow-motion. Results saved to `test_result.txt`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Streamlit Dashboard                      │
│  Live Detection │ Analytics │ Disputes │ Configuration   │
│                                                          │
│  Video display: MJPEG server (port 5679) — no reruns     │
└────────────────────────┬────────────────────────────────┘
                         │ background thread
┌────────────────────────▼────────────────────────────────┐
│                  _PipelineWorker                         │
│                                                          │
│  Reader thread ──► frame_q (maxsize=4)                   │
│       │                │                                 │
│  cv2.resize(640×640)   │                                 │
│                        ▼                                 │
│              model.track() — YOLO11 + ByteTrack          │
│                        │                                 │
│              _keep_box() filter                          │
│              (rejects hands / people / giant blobs)      │
│                        │                                 │
│              annotate frame → _state["frame1"]           │
│                        │                                 │
│              MJPEG server (port 5679) ──► browser        │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
OnionIQ/
├── dashboard.py       # Streamlit app — all UI + pipeline worker
├── test_detection.py  # Standalone CV test (no UI, fastest validation)
├── model_wrapper.py   # OnionModel — YOLO loader, _keep() filter, TRT support
├── config.py          # Settings loader/saver, grade class definitions
├── database.py        # SQLite WAL schema + thread-safe CRUD
├── tracker.py         # ByteTrack wrapper
├── matcher.py         # Cross-camera IoU matcher (future multi-cam)
├── grader.py          # Grade decision engine
├── ejector.py         # Ejector controller (simulated or real serial/GPIO)
├── grad_cam.py        # EigenCAM explainability for disputed detections
├── sync.py            # Offline-first HTTP sync daemon
├── certificate.py     # ReportLab batch PDF certificate generator
├── calibration.py     # Camera calibration utility
├── mock_model.py      # MockYOLO fallback (no GPU/model available)
├── settings.json      # Runtime config (model path, confidence, overlap zones)
├── requirements.txt
└── README.md
```

---

## Loading Your Trained Model

1. Go to **Configuration** tab in the dashboard
2. Paste the full path to your `.pt` file in **Custom Model Path**
3. Paste the path to your `data.yaml` in **Dataset YAML**
4. Click **Save Configuration**
5. Click **Restart** in the Live Detection tab

The model loads automatically. With a custom model loaded:
- HSV heuristic is **disabled** (YOLO output trusted directly)
- ByteTrack tracking is **active**
- Box color is **green** (same as `test_detection.py`)

### TensorRT (optional — for 60+ FPS)

If you have CUDA and want maximum speed, export your model once:

```bash
python -c "
from ultralytics import YOLO
m = YOLO('models/testdataset/runs/runs/onion_v1-2/weights/best.pt')
m.export(format='engine', batch=1, device=0, half=True, imgsz=640)
"
```

The dashboard and test script auto-detect the `.engine` file and use it.

---

## Training Your Own Model

```bash
# Place your Roboflow YOLOv11 dataset in models/testdataset/
# Fix data.yaml to use absolute paths, then:

yolo train \
  model=yolo11s.pt \
  data=models/testdataset/data.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=models/testdataset/runs \
  name=onion_v1
```

Best weights are saved to `models/testdataset/runs/onion_v1/weights/best.pt`.

---

## Detection Parameters (tuned values)

| Parameter | Value | Notes |
|---|---|---|
| Confidence | 0.30 | Lower catches more onions; raise to 0.45+ if false positives appear |
| IOU (NMS) | 0.45 | Overlap threshold for duplicate suppression |
| Image size | 640×640 | All frames resized before inference |
| Min box size | 10 px | Rejects tiny noise |
| Max box size | 60% of frame | Rejects hands, people, large blobs |
| Aspect ratio | 0.25–4.0 | Rejects elongated shapes (arms, conveyor rails) |

---

## Video Display Modes

| Mode | FPS | How |
|---|---|---|
| **MJPEG** (primary) | ~20–25 FPS | Local HTTP server on port 5679 — browser `<img>` pulls frames; no Streamlit reruns, smooth playback |
| **Standalone test** | native FPS | `test_detection.py` — `cv2.imshow()` window, no browser involved |

> WebRTC via `streamlit-webrtc` is available as a fallback but is optimised for live camera sources, not MP4 files.

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| **Live Detection** | Video feed, FPS, frame counter, onion count, Start/Stop/Restart/Reset controls, confidence/IOU sliders |
| **Analytics** | Grade distribution bar chart, defect breakdown pie chart, recent detections table |
| **Dispute Review** | Detections where cameras disagreed — shows both camera frames + Grad-CAM overlay |
| **Configuration** | Model path, data.yaml path, overlap zones, belt speed, ejector delay, cross-camera matching |

---

## Two-Camera Setup (future)

Currently single-camera mode is the primary focus. Multi-camera architecture is designed but not yet tested:

- One YOLO+ByteTrack tracker per camera (local IDs)
- Overlap zone entry triggers Hungarian matching
- Worst-case grading: `final_grade = max(severity(cam1), severity(cam2))`

Enable Camera 2 in the dashboard sidebar when ready to test.

---

## Database Schema

```sql
onion_grades (
    global_id             INTEGER PRIMARY KEY,
    cam1_grade            TEXT,
    cam2_grade            TEXT,
    final_grade           TEXT,
    defect_type           TEXT,
    estimated_diameter_mm REAL,
    cam1_confidence       REAL,
    cam2_confidence       REAL,
    ejected               INTEGER DEFAULT 0,
    timestamp             REAL,
    batch_id              TEXT,
    farmer_name           TEXT,
    sync_status           TEXT DEFAULT 'pending'
)
```

---

## Requirements

```
Python >= 3.10
CUDA-capable GPU (RTX 4060 or better recommended for 30+ FPS)

ultralytics >= 8.3.0    # YOLO11 inference
torch >= 2.3.0
opencv-python >= 4.10.0
streamlit >= 1.37.0
streamlit-webrtc >= 0.47.0   # WebRTC video streaming
aiortc >= 1.9.0               # MP4 → WebRTC transport
plotly >= 5.22.0
pandas >= 2.2.0
pyyaml >= 6.0                 # data.yaml parsing
reportlab >= 4.2.0            # PDF certificates
pytorch-grad-cam >= 1.5.0     # EigenCAM explainability
pyserial >= 3.5               # Real ejector (optional)
```

---

## Team

Built for Smart India Hackathon 2026 — Problem SIH26031  
Ministry of Consumer Affairs, Food & Public Distribution

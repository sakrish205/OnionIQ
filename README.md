# OnionIQ — AI-Powered Onion Quality Grading System

> **Smart India Hackathon 2026 · Problem Statement SIH26031**  
> Ministry of Consumer Affairs, Food & Public Distribution

OnionIQ is a real-time onion detection and counting system for roller conveyor belts. It uses a custom-trained YOLO11 model with ByteTrack persistent tracking, served through a Streamlit dashboard with smooth MJPEG video streaming.

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

Press `Q` to quit, `SPACE` to pause, `S` for slow-motion.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Streamlit Dashboard                      │
│  Live Detection │ Analytics │ Configuration                │
│                                                          │
│  Video: MJPEG server (port 5679) — smooth, no reruns    │
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
│              cv2.putText HUD overlay                     │
│                        │                                 │
│              _state["frame1"] ──► MJPEG ──► browser      │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
OnionIQ/
├── dashboard.py       # Streamlit app — all UI + pipeline worker
├── test_detection.py  # Standalone CV validation (no UI)
├── model_wrapper.py   # YOLO loader, _keep() filter, TRT support
├── config.py          # Settings loader/saver (onioniq.db path, defaults)
├── database.py        # SQLite WAL schema + thread-safe CRUD
├── tracker.py         # ByteTrack wrapper
├── matcher.py         # Cross-camera Hungarian matcher (future multi-cam)
├── grader.py          # Grade decision engine
├── ejector.py         # Ejector controller (simulated or real serial/GPIO)
├── grad_cam.py        # EigenCAM explainability for disputed detections
├── sync.py            # Offline-first HTTP sync daemon
├── certificate.py     # ReportLab batch PDF certificate generator
├── calibration.py     # Camera calibration utility
├── mock_model.py      # MockYOLO fallback (no GPU/model available)
├── settings.json      # Runtime config — gitignored, auto-created
├── requirements.txt
└── README.md
```

---

## Loading Your Trained Model

1. Go to the **Configuration** tab
2. Paste the full path to your `.pt` file in **Custom Model Path**
3. Paste the path to your `data.yaml` in **Dataset YAML**
4. Click **Save Configuration**
5. Click **Restart** in the Live Detection tab

With a custom model loaded:
- HSV heuristic is **disabled** (YOLO output trusted directly)
- ByteTrack tracking is **active**
- Box colour is **green** (same as `test_detection.py`)

### TensorRT (optional — 60+ FPS)

Export your model once:

```bash
python -c "
from ultralytics import YOLO
m = YOLO('path/to/best.pt')
m.export(format='engine', batch=1, device=0, half=True, imgsz=640)
"
```

The dashboard and test script auto-detect the `.engine` file alongside the `.pt`.

---

## Training Your Own Model

```bash
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

Best weights saved to `models/testdataset/runs/onion_v1/weights/best.pt`.

---

## Detection Parameters

| Parameter | Value | Notes |
|---|---|---|
| Confidence | 0.50 (default) | Adjustable via slider in Live Detection tab |
| IOU (NMS) | 0.45 | Overlap threshold for duplicate suppression |
| Image size | 640×640 | Frames resized before inference |
| Min box side | 10 px | Rejects tiny noise detections |
| Max box side | 60% of frame | Rejects hands, people, large blobs |
| Aspect ratio | 0.25–4.0 | Rejects elongated shapes (arms, rails) |

---

## Dashboard

### Live Detection tab

| Element | Description |
|---|---|
| **Onions Counted** | Cumulative unique ByteTrack IDs since last reset |
| **Frame Rate** | Inference FPS (updated every frame) |
| **Frames** | Current frame / total frames in video |
| **Status** | Live / Stopped |
| **Camera selectbox** | Dropdown (Camera 0, Camera 1 …); Scan button finds active indices |
| **Video File** | Drag-and-drop or paste path to `.mp4 / .avi / .mov` |
| **Start / Stop / Restart** | Pipeline control |
| **Reset Count** | Zeros the track-ID set mid-run without stopping |
| **Confidence / NMS IoU** | Live sliders; Save applies on next Start (or Restart if running) |

### Other tabs

| Tab | Contents |
|---|---|
| **Analytics** | Grade distribution bar chart, defect breakdown pie, recent detections table |
| **Configuration** | Model path, data.yaml, belt speed, ejector delay, size calibration |

---

## Video Display

| Mode | FPS | Notes |
|---|---|---|
| **MJPEG** (active) | ~20–25 FPS | Local HTTP server port 5679; browser `<img>` streams frames directly — no Streamlit reruns, no flicker |
| **Standalone** | native FPS | `test_detection.py` — `cv2.imshow()`, no browser |

---

## Database

Active database: `onioniq.db` (SQLite WAL, thread-safe).

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
    sync_status           TEXT DEFAULT 'pending'
)
```

---

## Requirements

```
Python >= 3.10
CUDA GPU recommended (RTX 4060 or better for 30+ FPS)

ultralytics >= 8.3.0
torch >= 2.3.0
opencv-python >= 4.10.0
streamlit >= 1.37.0
plotly >= 5.22.0
pandas >= 2.2.0
pyyaml >= 6.0
reportlab >= 4.2.0
pytorch-grad-cam >= 1.5.0
pyserial >= 3.5          # optional — real ejector only
streamlit-webrtc >= 0.47.0
aiortc >= 1.9.0
```

---

## Team

Built for Smart India Hackathon 2026 — Problem SIH26031  
Ministry of Consumer Affairs, Food & Public Distribution

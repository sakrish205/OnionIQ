# 🧅 OnionIQ — AI-Powered Onion Quality Grading System

> **Smart India Hackathon 2026 · Problem Statement SIH26031**  
> Ministry of Consumer Affairs, Food & Public Distribution

OnionIQ is a real-time, dual-camera onion grading system that runs on a roller conveyor belt. It uses YOLO11n-seg for detection and segmentation, ByteTrack for per-camera tracking, cross-camera IoU matching to avoid counting the same onion twice, and worst-case grading logic to assign a final grade. All data is stored locally in SQLite and synced to a central server when WiFi is available — designed for rural APMC mandis with intermittent connectivity.

---

## Demo

| Live Demo | Analytics | Disputes |
|---|---|---|
| Real-time camera feed with bounding boxes, track IDs, grade labels | Nivo bar chart (grade distribution) + pie chart (defects), draggable | Cam1 vs Cam2 disagreement explorer with Grad-CAM overlay |

---

## Features

### Core Pipeline
- **YOLO11n-seg** — detection + segmentation on live camera or video file
- **ByteTrack** — per-camera persistent tracking (falls back to centroid tracker)
- **Cross-camera matching** — IoU-based global ID assignment across the overlap zone; each onion counted once
- **Worst-case grading** — `final_grade = max(severity(cam1), severity(cam2))`
- **Ejector control** — schedules ejection signal after configurable belt delay

### Dashboard (Streamlit + Material UI)
- **Live feed** with bounding boxes, track IDs, confidence scores, overlap zone overlay
- **Source switcher** — camera index or video file; 🔄 Restart switches instantly without Stop
- **📷 Scan cameras** — auto-detects available camera indices (0–4)
- **Detection controls** — Confidence / IoU sliders applied live to the running pipeline
- **Material UI metric cards** — Detected count, FPS, Active Tracks, Status
- **Draggable Nivo charts** — Grade Distribution (bar) + Defect Breakdown (pie)
- **⚙️ Settings panel** — all calibration constants editable at runtime, saved to `settings.json`

### Novelty Features (as per SIH spec)
| # | Feature | Implementation |
|---|---|---|
| 1 | Defect taxonomy | 10-class YOLO (grade_a/b/c + 7 defect types); defect field in DB + pie chart |
| 2 | Size estimation | Mask area × calibration factor → diameter in mm; shown per detection |
| 3 | Farmer fingerprint | Per-farmer aggregate stats (sessions, reject rate, grade-A rate) across batches |
| 4 | Grad-CAM explainability | EigenCAM heatmap on disputed onions (cam1 ≠ cam2 grade) |
| 5 | Offline-first sync | SQLite queue → HTTP POST when WiFi available; demo: disconnect WiFi, reconnect, watch sync |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                  │
│  Live Demo │ Analytics │ Disputes │ Settings             │
└────────────────────────┬────────────────────────────────┘
                         │ (background thread)
┌────────────────────────▼────────────────────────────────┐
│                   _PipelineWorker                        │
│  Camera/Video ──► YOLO11n-seg ──► ByteTrack             │
│                                      │                   │
│        CrossCameraMatcher ◄──────────┘                   │
│               │                                          │
│        GradeDecisionEngine ──► SQLite (WAL)              │
│               │                                          │
│        EjectorController    SyncWorker (daemon)          │
└─────────────────────────────────────────────────────────┘
```

### File Structure
```
OnionIQ/
├── config.py          # All calibration constants + settings.json loader/saver
├── utils.py           # DetectionEvent dataclass, bbox_iou, logging
├── database.py        # SQLite WAL schema + thread-safe CRUD
├── mock_model.py      # MockYOLO — same interface as ultralytics Results
├── model_wrapper.py   # OnionModel — auto-downloads yolo11n-seg.pt; custom model hot-swap
├── tracker.py         # ByteTrack wrapper + centroid tracker fallback
├── camera.py          # CameraThread — frame queue with mock fallback
├── matcher.py         # CrossCameraMatcher — IoU-based global ID assignment
├── grader.py          # GradeDecisionEngine — worst-case logic + size estimation
├── ejector.py         # EjectorController — simulated or real serial/GPIO
├── grad_cam.py        # EigenCAM explainability for disputed onions
├── sync.py            # SyncWorker — offline-first HTTP sync daemon
├── certificate.py     # ReportLab batch PDF certificate generator
├── calibration.py     # Interactive HoughCircles calibration + overlap zone marking
├── dashboard.py       # Streamlit app — self-contained, no separate main.py needed
├── main.py            # CLI entrypoint for headless / two-laptop deployment
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/sakrish205/OnionIQ.git
cd OnionIQ
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.

### 3. Start detection
1. Go to **📹 Live Demo** tab
2. Click **📷 Scan cameras** to find your camera index
3. Enter the index in **Camera 1 index** (usually `0`)
4. Click **▶ Start**

The system auto-downloads `yolo11n-seg.pt` (~6 MB) on first run. All detections are labelled **onion** until you load a trained model.

---

## Adding Your Trained Model

1. Train YOLO11s-seg on your onion dataset (class names must match those in `config.py`)
2. Drop the `.pt` file anywhere on the machine
3. Go to **⚙️ Settings** → paste the path → **💾 Save Settings**
4. Click **🔄 Restart** in Live Demo — grading activates automatically

No code changes needed. Class names load from the model weights.

---

## GPU Acceleration

The system runs on CPU by default. For GPU (NVIDIA):

```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch (replace cu128 with your version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

YOLO auto-detects and uses the GPU — no code changes needed.

---

## Two-Camera Setup

| Mode | How |
|---|---|
| Single camera (default) | Leave **Enable Camera 2** unchecked |
| Two cameras, one laptop | Enable Camera 2, set index 1 |
| Two cameras, two laptops | Use `main.py` — Camera 2 laptop runs `RemoteCameraProxy` (sends events over WiFi) |

Camera 2 is completely optional. The pipeline degrades gracefully to single-camera mode.

---

## Demo Scenarios for Judges

**Offline sync demo:**
1. Start pipeline → let it grade a batch
2. Disconnect WiFi → observe "Pending Sync" counter in sidebar increase
3. Reconnect WiFi → `SyncWorker` daemon auto-posts pending rows and clears counter

**Source switch (no downtime):**
1. Start on Camera → while running, switch radio to "Video File" → paste path
2. Click **🔄 Restart** — feed switches in ~1 second, no manual Stop needed

**Settings live-tuning:**
1. Go to ⚙️ Settings → adjust Confidence slider → Save
2. Pipeline picks up new threshold within 2 seconds (no restart)

---

## Database Schema

```sql
onion_grades (
    global_id          INTEGER PRIMARY KEY,
    cam1_grade         TEXT,
    cam2_grade         TEXT,
    final_grade        TEXT,    -- worst-case of cam1 + cam2
    defect_type        TEXT,    -- specific defect class if present
    estimated_diameter_mm REAL,
    cam1_confidence    REAL,
    cam2_confidence    REAL,
    mask_area_cam1     INTEGER,
    mask_area_cam2     INTEGER,
    ejected            INTEGER DEFAULT 0,
    timestamp          REAL,
    batch_id           TEXT,
    farmer_name        TEXT,
    centre_id          TEXT,
    sync_status        TEXT DEFAULT 'pending'  -- 'pending' | 'synced'
)
```

---

## Grade Classes

| Grade | Severity | Description |
|---|---|---|
| `grade_a` | 0 | Premium — no defects |
| `grade_b` | 1 | Minor surface issues |
| `grade_c` | 2 | Visible defects, still marketable |
| `sprouting` | 2 | Visible sprout growth |
| `sunscald` | 2 | Sun-damaged outer skin |
| `bruising` | 2 | Physical damage |
| `thrips_damage` | 2 | Insect damage |
| `neck_rot` | 3 | Fungal infection at neck |
| `rot` | 3 | Active rot |
| `reject` | 3 | Unmarketable — triggers ejector |

Worst-case rule: if cam1 says `grade_a` and cam2 says `rot`, final grade is `rot`.

---

## Requirements

```
Python >= 3.10
ultralytics >= 8.3.0   # YOLO11n-seg
torch >= 2.3.0
opencv-python
streamlit >= 1.36.0
streamlit-elements == 0.1.*   # Material UI + Nivo charts
pandas, plotly, requests
reportlab                      # PDF certificates
pytorch-grad-cam               # EigenCAM explainability
pyserial                       # Real ejector (optional)
```

---

## Team

Built for Smart India Hackathon 2026 — Problem SIH26031  
Ministry of Consumer Affairs, Food & Public Distribution

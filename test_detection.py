"""
OnionIQ — standalone detection test (no Streamlit, no UI).
- Multithreaded: reader + inference threads
- Frames resized to 640×640 before inference
- Size + aspect-ratio filter to reject hands/people/huge blobs
- Stops when video ends (no loop)
- Saves count summary to test_result.txt
Press Q to quit early, SPACE to pause.
"""
import csv
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
_PT    = r"C:\Users\sakee\Desktop\projects\onion\models\testdataset\runs\runs\onion_v1-2\weights\best.pt"
VIDEO  = r"C:\Users\sakee\Desktop\projects\onion\videos\onion4.mp4"
CONF   = 0.30   # lower = catches more onions; raise if too many false positives
IOU    = 0.45
IMGSZ  = 640
RESULT = Path(__file__).parent / "test_result.txt"

# Auto-prefer TRT engine (best.engine) for 60+ FPS on CUDA
_engine = Path(_PT).with_suffix(".engine")
MODEL   = str(_engine) if _engine.exists() else _PT

# Post-filter: reject detections outside these bounds
MIN_SIDE_PX   = 10      # ignore boxes smaller than 10 px
MAX_SIDE_FRAC = 0.60    # ignore if side > 60% of frame — whole-body reject
MIN_ASPECT    = 0.25    # w/h ratio — reject very elongated shapes (arms, hands)
MAX_ASPECT    = 4.0

def _keep(x1, y1, x2, y2):
    """Return True if bbox looks like an onion (size + shape filter)."""
    w = x2 - x1
    h = y2 - y1
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return False
    if w > MAX_SIDE_FRAC * IMGSZ or h > MAX_SIDE_FRAC * IMGSZ:
        return False
    ar = w / max(h, 1)
    return MIN_ASPECT <= ar <= MAX_ASPECT

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Model  : {Path(MODEL).name}")
print(f"Video  : {Path(VIDEO).name}")
print(f"Conf   : {CONF}   IOU: {IOU}   imgsz: {IMGSZ}")
print(f"Filter : size {MIN_SIDE_PX}–{int(MAX_SIDE_FRAC*IMGSZ)} px, "
      f"aspect {MIN_ASPECT}–{MAX_ASPECT}")
print()

if _engine.exists():
    print(f"[TRT] Engine found — using {_engine.name}\n")
else:
    print(f"[PyTorch] No TRT engine — using {Path(_PT).name}\n")

print("Loading model…")
model = YOLO(MODEL)
print(f"Classes: {list(model.names.values())}\n")

# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO, cv2.CAP_FFMPEG)
if not cap.isOpened():
    cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"ERROR: cannot open {VIDEO}")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video  : {orig_w}×{orig_h} @ {src_fps:.1f} fps  ({total_frames} frames)")
print("Press Q to quit early, SPACE to pause\n")

# ── Shared queues, flags and loop-reset signal ────────────────────────────────
frame_q    = queue.Queue(maxsize=4)   # small — forces reader to wait for inference
result_q   = queue.Queue(maxsize=8)
stop_flag  = threading.Event()
reset_flag = threading.Event()    # set by reader on each loop; inference resets counts

# ── Reader thread — loops video, signals reset on each restart ────────────────
def reader():
    loop = 0
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            # Wait for inference to drain every queued frame before resetting
            while not frame_q.empty() and not stop_flag.is_set():
                time.sleep(0.05)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loop += 1
            reset_flag.set()
            print(f"  [reader] loop {loop} — counts reset")
            continue
        resized = cv2.resize(frame, (IMGSZ, IMGSZ))
        # Block until inference picks it up — ensures every frame is processed
        while not stop_flag.is_set():
            try:
                frame_q.put(resized, timeout=0.2)
                break
            except queue.Full:
                pass  # inference busy, retry

threading.Thread(target=reader, daemon=True, name="Reader").start()

# ── Inference thread ──────────────────────────────────────────────────────────
det_total   = 0
frame_num   = 0
errors      = 0
inf_fps_val = 0.0
unique_ids  = set()

def inference():
    global det_total, frame_num, errors, inf_fps_val, unique_ids
    t0 = time.time()
    while not stop_flag.is_set():
        # Reset counts when reader signals a new loop
        if reset_flag.is_set():
            det_total = 0; frame_num = 0; unique_ids = set()
            t0 = time.time(); errors = 0
            reset_flag.clear()

        try:
            frame = frame_q.get(timeout=0.3)
        except queue.Empty:
            continue

        frame_num += 1
        try:
            results = model.track(
                frame, conf=CONF, iou=IOU, imgsz=IMGSZ,
                persist=True, tracker="bytetrack.yaml",
                verbose=False,
            )
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [inference] frame {frame_num}: {e}")
            try:
                results = model.predict(frame, conf=CONF, iou=IOU,
                                        imgsz=IMGSZ, verbose=False)
            except Exception:
                continue

        boxes  = results[0].boxes
        passed = []   # boxes that pass the filter
        if boxes is not None and len(boxes.xyxy) > 0:
            track_ids = None
            if hasattr(boxes, "id") and boxes.id is not None:
                try:
                    track_ids = [int(boxes.id[i].item())
                                 for i in range(len(boxes.id))]
                except Exception:
                    pass

            for i, xyxy in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                if not _keep(x1, y1, x2, y2):
                    continue
                tid  = track_ids[i] if track_ids and i < len(track_ids) else -1
                conf = float(boxes.conf[i])
                cls  = int(boxes.cls[i])
                passed.append((x1, y1, x2, y2, tid, conf, cls))
                if tid > 0:
                    unique_ids.add(tid)

        det_total += len(passed)
        elapsed   = max(time.time() - t0, 1e-6)
        inf_fps_val = frame_num / elapsed

        try:
            result_q.put_nowait((frame.copy(), passed, frame_num,
                                 det_total, inf_fps_val, model.names))
        except queue.Full:
            pass

inf_thread = threading.Thread(target=inference, daemon=True, name="Inference")
inf_thread.start()

# ── Display (main thread) ─────────────────────────────────────────────────────
paused  = False
slomo   = False   # press S to toggle slow-motion (200 ms per frame)
last    = None
start   = time.time()

while True:
    if not paused:
        try:
            last = result_q.get(timeout=0.05)
        except queue.Empty:
            pass

    if last is None:
        cv2.waitKey(30)
        continue

    frame, passed, f_num, d_total, fps_inf, names = last
    out = frame.copy()

    for (x1, y1, x2, y2, tid, conf, cls) in passed:
        name  = names[cls]
        color = (0, 200, 60)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    progress = f_num / max(total_frames, 1) * 100
    hud = (f"Frame {f_num}/{total_frames} ({progress:.0f}%)  "
           f"Det: {d_total}  Tracks: {len(unique_ids)}  "
           f"Inf {fps_inf:.1f} FPS  Err:{errors}")
    cv2.putText(out, hud, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 2)

    status = []
    if paused: status.append("PAUSED")
    if slomo:  status.append("SLO-MO")
    if status:
        cv2.putText(out, "  ".join(status), (6, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

    controls = "Q=quit  SPACE=pause  S=slo-mo"
    cv2.putText(out, controls, (6, IMGSZ - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.imshow("OnionIQ detection test", out)
    wait_ms = 200 if slomo else 1
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        print("Quit by user.")
        break
    if key == ord(" "):
        paused = not paused
    if key == ord("s"):
        slomo = not slomo
        print(f"  Slo-mo: {'ON' if slomo else 'OFF'}")

stop_flag.set()
inf_thread.join(timeout=3)
cap.release()
cv2.destroyAllWindows()

elapsed_total = time.time() - start

# ── Save results ──────────────────────────────────────────────────────────────
summary = f"""OnionIQ Detection Test
======================
Date/time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video        : {VIDEO}
Model        : {MODEL}
Confidence   : {CONF}   IOU: {IOU}   imgsz: {IMGSZ}
Filter       : side {MIN_SIDE_PX}–{int(MAX_SIDE_FRAC*IMGSZ)} px, aspect {MIN_ASPECT}–{MAX_ASPECT}

Results
-------
Total frames processed : {frame_num} / {total_frames}
Total detections       : {det_total}
Unique track IDs       : {len(unique_ids)}
Inference errors       : {errors}
Avg inference FPS      : {frame_num / max(elapsed_total, 1):.1f}
Wall-clock time        : {elapsed_total:.1f} s
"""
print(summary)
RESULT.write_text(summary, encoding="utf-8")
print(f"Results saved to: {RESULT}")

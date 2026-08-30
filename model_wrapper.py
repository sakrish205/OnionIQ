"""
OnionModel: loads YOLO weights (user-supplied or auto-downloaded yolo11n-seg.pt).
Falls back to MockYOLO only if YOLO itself fails to load.
In detection-only mode all outputs are labelled "onion" regardless of COCO class.
Includes an HSV color+shape heuristic to detect onions when no custom model exists.
Video mode: TensorRT (fp16, batch=10) + multi-threaded reader for maximum throughput.
"""
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config import GRADE_CLASSES, DEFAULT_CLASS_NAMES
from utils import DetectionEvent, is_in_overlap, setup_logging

logger = setup_logging()

_DEFAULT_MODEL = "yolo11n-seg.pt"

# Use GPU automatically if available
try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


_MIN_SIDE_PX   = 10
_MAX_SIDE_FRAC = 0.60
_MIN_ASPECT    = 0.25
_MAX_ASPECT    = 4.0

def _keep(x1, y1, x2, y2, frame_size=640):
    w = x2 - x1; h = y2 - y1
    if w < _MIN_SIDE_PX or h < _MIN_SIDE_PX:
        return False
    if w > _MAX_SIDE_FRAC * frame_size or h > _MAX_SIDE_FRAC * frame_size:
        return False
    ar = w / max(h, 1)
    return _MIN_ASPECT <= ar <= _MAX_ASPECT


class OnionModel:
    def __init__(self, settings: dict):
        self._settings = settings
        self._model = None
        self._class_names: list = DEFAULT_CLASS_NAMES
        self.using_mock: bool = False
        self.trt_ready: bool = False
        self.has_custom_model: bool = False   # True = user-supplied .pt; skip HSV heuristic
        self._load(settings.get("model_path", ""))

    # ── TensorRT ──────────────────────────────────────────────────────────────
    def _try_trt_load(self, pt_path: str) -> bool:
        """Export .pt → TensorRT .engine (fp16, batch=10) then reload from it.
        Skips silently if not on CUDA or export fails."""
        if _DEVICE != "cuda":
            return False
        p = Path(pt_path)
        if not p.exists() or p.suffix != ".pt":
            return False
        engine_path = p.with_suffix(".engine")
        try:
            if not engine_path.exists():
                logger.info(
                    f"[TRT] Exporting {p.name} → TensorRT (fp16, batch=10). "
                    "First-run only — takes 2-5 min. Subsequent starts are instant."
                )
                from ultralytics import YOLO as _YOLO
                _tmp = _YOLO(str(p))
                _tmp.export(format="engine", batch=10, device=0, half=True, imgsz=640)
                logger.info(f"[TRT] Engine saved: {engine_path}")
            from ultralytics import YOLO as _YOLO
            self._model = _YOLO(str(engine_path))
            self.trt_ready = True
            logger.info(f"[TRT] Loaded engine: {engine_path}")
            return True
        except Exception as e:
            logger.warning(f"[TRT] Unavailable ({e}), using PyTorch.")
            return False

    def _read_yaml_classes(self) -> None:
        """Override class names from data.yaml if the setting is provided."""
        data_yaml = self._settings.get("data_yaml", "")
        if not data_yaml:
            return
        p = Path(data_yaml)
        if not p.exists():
            logger.warning(f"data.yaml not found: {p}")
            return
        try:
            import yaml
            with open(p) as f:
                dy = yaml.safe_load(f)
            names = dy.get("names", None)
            if isinstance(names, dict):
                self._class_names = [names[i] for i in sorted(names)]
            elif isinstance(names, list):
                self._class_names = names
            else:
                return
            logger.info(f"Class names from {p.name}: {self._class_names}")
        except Exception as e:
            logger.warning(f"data.yaml parse error ({e}), keeping model class names.")

    def _load(self, model_path: str) -> None:
        from ultralytics import YOLO
        self.trt_ready = False

        # 1. Try user-supplied path first
        path = Path(model_path) if model_path else None
        if path and path.exists():
            try:
                self._model = YOLO(str(path))
                self._class_names = list(self._model.names.values())
                self.using_mock = False
                self.has_custom_model = True        # trust YOLO; disable HSV heuristic
                logger.info(f"Loaded custom model: {path} | classes: {self._class_names}")
                self._read_yaml_classes()           # override with data.yaml names if set
                self._try_trt_load(str(path))       # upgrade to TRT if CUDA available
                return
            except Exception as e:
                logger.warning(f"Custom model failed ({e}), falling back to default.")

        # 2. Auto-download and use yolo11n-seg.pt (real inference, detection-only)
        try:
            self._model = YOLO(_DEFAULT_MODEL)
            self._model.to(_DEVICE)
            self._class_names = DEFAULT_CLASS_NAMES
            self.using_mock = False
            self._read_yaml_classes()               # still apply YAML class names if given
            logger.info(f"Using {_DEFAULT_MODEL} on {_DEVICE.upper()} (detection-only mode).")
            return
        except Exception as e:
            logger.warning(f"YOLO default model failed ({e}), falling back to MockYOLO.")

        # 3. Absolute last resort — no network, no YOLO
        from mock_model import MockYOLO
        self._model = MockYOLO(class_names=DEFAULT_CLASS_NAMES)
        self._class_names = DEFAULT_CLASS_NAMES
        self.using_mock = True
        logger.info("Using MockYOLO (no YOLO available).")

    def reload(self, new_settings: dict) -> None:
        self._settings = new_settings
        self._load(new_settings.get("model_path", ""))

    @property
    def class_names(self) -> list:
        return self._class_names

    @property
    def grading_active(self) -> bool:
        """True when loaded class names include at least some grade classes."""
        return bool(set(self._class_names) & GRADE_CLASSES)

    def _detect_onions_cv(self, frame: np.ndarray) -> List[list]:
        """
        HSV mask + distance-transform local maxima to detect INDIVIDUAL onions.
        Each round peak in the distance transform = one onion, even when touching.
        Returns list of [x1, y1, x2, y2] bboxes.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        min_dim = min(h, w)

        # Onion skin colour ranges
        mask = cv2.inRange(hsv, (8,  25,  50), (35, 230, 240))  # brown / tan / yellow
        mask |= cv2.inRange(hsv, (0,   0, 150), (40,  55, 255)) # cream / white
        mask |= cv2.inRange(hsv, (150, 25, 50), (180, 210, 210))# red-purple (H wrap)
        mask |= cv2.inRange(hsv, (0,   25, 50), (10,  210, 210))# red (low H)

        # Close small holes within an onion; open to remove thin noise
        k = max(3, min_dim // 100)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=2)

        # Distance transform: value at each pixel = distance to nearest mask edge
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Typical single onion radius in this footage (pixels):
        # adjust via calibration_factor; default assumes ~40-80px diameter
        min_r = max(12, min_dim // 35)   # ~3% of shorter dimension
        max_r = min(min_dim // 5, 90)    # cap so whole-pile blobs are rejected

        # Local maxima in distance transform: each peak is one individual onion centre.
        # Peak kernel: slightly smaller than min expected onion so touching onions
        # each keep their own distinct peak.
        pk = max(9, min_r) | 1          # odd number
        pk_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
        dilated = cv2.dilate(dist, pk_kern)
        peaks = ((dist == dilated) & (dist > min_r * 0.7)).astype(np.uint8)

        # Each connected component of the peak map = one onion centre
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)

        bboxes = []
        for i in range(1, n_labels):
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            r = int(dist[cy, cx])
            if r < min_r or r > max_r:
                continue
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)
            bboxes.append([x1, y1, x2, y2])

        return bboxes

    def predict_batch(self, frames: List[np.ndarray], cam_id: int) -> List[List[DetectionEvent]]:
        """Batched inference — runs all frames in one GPU call (TRT batch=10).
        Returns one DetectionEvent list per input frame."""
        if not frames:
            return []
        s = self._settings
        conf  = s.get("confidence_threshold", 0.35)
        iou   = s.get("iou_threshold", 0.45)
        imgsz = s.get("image_size", 640)

        if self.using_mock:
            return [self._parse(self._model(f), f, cam_id, s) for f in frames]

        try:
            results = self._model.predict(
                frames, conf=conf, iou=iou, imgsz=imgsz,
                device=_DEVICE, verbose=False,
            )
        except Exception as e:
            logger.warning(f"Batch predict failed ({e}); falling back to per-frame")
            return [self.predict(f, cam_id) for f in frames]

        if cam_id == 1:
            ov_start = s.get("overlap_start_cam1_x", 900)
            ov_end   = s.get("overlap_end_cam1_x", 1280)
        else:
            ov_start = s.get("overlap_start_cam2_x", 0)
            ov_end   = s.get("overlap_end_cam2_x", 380)

        all_events: List[List[DetectionEvent]] = []
        for frame, result in zip(frames, results):
            events = self._parse([result], frame, cam_id, s)
            if not self.has_custom_model:   # HSV only when no custom model is loaded
                cv_bboxes = self._detect_onions_cv(frame)
                existing  = [e.bbox for e in events]
                for bbox in cv_bboxes:
                    if any(_iou(bbox, eb) > 0.3 for eb in existing):
                        continue
                    events.append(DetectionEvent(
                        cam_id=cam_id, track_id=-1, class_name="onion",
                        confidence=0.72, bbox=bbox, mask_area_px=0,
                        in_overlap=is_in_overlap(bbox, ov_start, ov_end),
                        timestamp=time.time(), global_id=None, frame=None,
                    ))
                    existing.append(bbox)
            all_events.append(events)
        return all_events

    def predict(self, frame: np.ndarray, cam_id: int) -> List[DetectionEvent]:
        s = self._settings
        conf  = s.get("confidence_threshold", 0.35)
        iou   = s.get("iou_threshold", 0.45)
        imgsz = s.get("image_size", 640)

        if self.using_mock:
            results = self._model(frame)
            return self._parse(results, frame, cam_id, s)

        # Use ultralytics built-in tracking (ByteTrack) so detections carry
        # persistent track IDs across frames — model.track(persist=True)
        try:
            results = self._model.track(
                frame, conf=conf, iou=iou, imgsz=imgsz,
                persist=True, tracker="bytetrack.yaml",
                device=_DEVICE, verbose=False,
            )
        except Exception:
            results = self._model(
                frame, conf=conf, iou=iou, imgsz=imgsz,
                device=_DEVICE, verbose=False,
            )
        yolo_events = self._parse(results, frame, cam_id, s)

        # If a custom model is loaded (any class), trust YOLO — no HSV needed
        if self.has_custom_model:
            return yolo_events

        # Default COCO model: supplement sparse output with HSV heuristic
        cv_bboxes = self._detect_onions_cv(frame)
        if cam_id == 1:
            ov_start = s.get("overlap_start_cam1_x", 900)
            ov_end   = s.get("overlap_end_cam1_x",   1280)
        else:
            ov_start = s.get("overlap_start_cam2_x", 0)
            ov_end   = s.get("overlap_end_cam2_x",   380)

        # Filter cv_bboxes not already covered by a YOLO box (simple IoU check)
        existing_boxes = [e.bbox for e in yolo_events]

        for bbox in cv_bboxes:
            if any(_iou(bbox, eb) > 0.3 for eb in existing_boxes):
                continue
            yolo_events.append(DetectionEvent(
                cam_id=cam_id, track_id=-1, class_name="onion",
                confidence=0.72, bbox=bbox, mask_area_px=0,
                in_overlap=is_in_overlap(bbox, ov_start, ov_end),
                timestamp=time.time(), global_id=None, frame=None,
            ))
            existing_boxes.append(bbox)
        return yolo_events

    def _parse(self, results, frame: np.ndarray, cam_id: int, s: dict) -> List[DetectionEvent]:
        events: List[DetectionEvent] = []
        if not results:
            return events

        result = results[0]
        names = result.names if hasattr(result, "names") else {}

        # Overlap zone for this camera
        if cam_id == 1:
            ov_start = s.get("overlap_start_cam1_x", 900)
            ov_end   = s.get("overlap_end_cam1_x", 1280)
        else:
            ov_start = s.get("overlap_start_cam2_x", 0)
            ov_end   = s.get("overlap_end_cam2_x", 380)

        boxes = result.boxes
        masks = result.masks if hasattr(result, "masks") and result.masks is not None else None

        if boxes is None or len(boxes.xyxy) == 0:
            return events

        # track IDs from model.track(); None when model.predict() was used
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            try:
                track_ids = [int(np.asarray(tid).flat[0]) for tid in boxes.id]
            except Exception:
                pass

        imgsz = max(frame.shape[:2]) if frame is not None else 640
        for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            try:
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                if not _keep(x1, y1, x2, y2, imgsz):
                    continue
                confidence = float(np.asarray(conf).flat[0])
                cls_idx    = int(np.asarray(cls).flat[0])

                # Normalise class name to lowercase for consistent colour lookup
                raw_name = names.get(cls_idx, "onion")
                class_name = raw_name.lower() if self.grading_active else "onion"

                # Persistent track ID from ultralytics tracker (or -1 as fallback)
                tid = track_ids[i] if (track_ids and i < len(track_ids)) else -1

                # Mask area in pixels
                mask_area = 0
                if masks is not None:
                    try:
                        if hasattr(result, "mask_pixel_area"):
                            mask_area = result.mask_pixel_area(i)
                        elif hasattr(masks, "data") and i < len(masks.data):
                            mask_area = int(masks.data[i].sum())
                    except Exception:
                        pass

                bbox = [x1, y1, x2, y2]
                in_overlap = is_in_overlap(bbox, ov_start, ov_end)

                events.append(DetectionEvent(
                    cam_id=cam_id,
                    track_id=tid,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    mask_area_px=mask_area,
                    in_overlap=in_overlap,
                    timestamp=time.time(),
                    global_id=None,
                    frame=frame.copy() if frame is not None else None,
                ))
            except Exception as e:
                logger.debug(f"Parse error on detection {i}: {e}")

        return events

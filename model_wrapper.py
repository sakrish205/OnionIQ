"""
OnionModel: loads YOLO weights (user-supplied or auto-downloaded yolo11n-seg.pt).
Falls back to MockYOLO only if YOLO itself fails to load.
In detection-only mode all outputs are labelled "onion" regardless of COCO class.
"""
import time
from pathlib import Path
from typing import List

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


class OnionModel:
    def __init__(self, settings: dict):
        self._settings = settings
        self._model = None
        self._class_names: list = DEFAULT_CLASS_NAMES
        self.using_mock: bool = False
        self._load(settings.get("model_path", ""))

    def _load(self, model_path: str) -> None:
        from ultralytics import YOLO

        # 1. Try user-supplied path first
        path = Path(model_path) if model_path else None
        if path and path.exists():
            try:
                self._model = YOLO(str(path))
                self._class_names = list(self._model.names.values())
                self.using_mock = False
                logger.info(f"Loaded custom model: {path} | classes: {self._class_names}")
                return
            except Exception as e:
                logger.warning(f"Custom model failed ({e}), falling back to default.")

        # 2. Auto-download and use yolo11n-seg.pt (real inference, detection-only)
        try:
            self._model = YOLO(_DEFAULT_MODEL)
            self._model.to(_DEVICE)
            self._class_names = DEFAULT_CLASS_NAMES
            self.using_mock = False
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

    def predict(self, frame: np.ndarray, cam_id: int) -> List[DetectionEvent]:
        s = self._settings
        conf = s.get("confidence_threshold", 0.35)
        iou  = s.get("iou_threshold", 0.45)
        imgsz = s.get("image_size", 640)

        if self.using_mock:
            results = self._model(frame)
        else:
            results = self._model(
                frame, conf=conf, iou=iou, imgsz=imgsz,
                device=_DEVICE, verbose=False,
            )

        return self._parse(results, frame, cam_id, s)

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

        for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            try:
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                confidence = float(np.asarray(conf).flat[0])
                cls_idx    = int(np.asarray(cls).flat[0])

                # Detection-only mode: label everything as "onion" until a grading model is loaded
                if self.grading_active:
                    class_name = names.get(cls_idx, "onion")
                else:
                    class_name = "onion"

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
                    track_id=-1,
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

import logging
import math
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DetectionEvent:
    cam_id: int                        # 1 or 2
    track_id: int                      # ByteTrack local ID
    class_name: str                    # class label from model
    confidence: float
    bbox: list                         # [x1, y1, x2, y2] in pixels
    mask_area_px: int                  # segmentation mask pixel count (0 if no mask)
    in_overlap: bool                   # bbox centre is inside the overlap zone
    timestamp: float = field(default_factory=time.time)
    global_id: Optional[int] = None   # assigned by CrossCameraMatcher
    frame: Optional[np.ndarray] = None  # raw frame snapshot, cleared after DB write


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("onion")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = RotatingFileHandler(
        Path(__file__).parent / "onion_system.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
    )
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def bbox_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def mask_area_to_mm2(pixel_area: int, cal_factor: float) -> float:
    return pixel_area * cal_factor


def mm2_to_diameter(area_mm2: float) -> float:
    """Estimate diameter (mm) from area assuming circular cross-section."""
    if area_mm2 <= 0:
        return 0.0
    return 2.0 * math.sqrt(area_mm2 / math.pi)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def bbox_centre(bbox: list) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def is_in_overlap(bbox: list, start_x: int, end_x: int) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    return start_x <= cx <= end_x

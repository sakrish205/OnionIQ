"""
MockYOLO: produces synthetic detections that mirror the ultralytics Results interface.
Used when no real model weights are present.
"""
import random
import time

import numpy as np


class _MockBoxes:
    def __init__(self, data: list):
        # data: list of [x1, y1, x2, y2, conf, cls_idx]
        import numpy as np
        arr = np.array(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32)
        self.xyxy = arr[:, :4]   # shape (n, 4)
        self.conf = arr[:, 4]    # shape (n,) — matches real ultralytics interface
        self.cls = arr[:, 5]     # shape (n,)


class _MockMasks:
    def __init__(self, areas: list, shape: tuple):
        import numpy as np
        # Each mask is a random blob; we only need the pixel count (area)
        self._areas = areas
        self._shape = shape
        h, w = shape[:2]
        masks = []
        for _ in areas:
            m = np.zeros((h, w), dtype=np.uint8)
            masks.append(m)
        self.data = np.array(masks) if masks else np.zeros((0, h, w), dtype=np.uint8)
        self._pixel_areas = areas

    def pixel_areas(self):
        return self._pixel_areas


class MockResult:
    def __init__(self, boxes_data: list, mask_areas: list,
                 names: dict, frame_shape: tuple):
        self.boxes = _MockBoxes(boxes_data)
        self.masks = _MockMasks(mask_areas, frame_shape)
        self.names = names
        self._mask_areas = mask_areas

    def mask_pixel_area(self, idx: int) -> int:
        if idx < len(self._mask_areas):
            return self._mask_areas[idx]
        return 0


# Class distribution for realistic mock output
_GRADE_DIST = [
    ("grade_a", 0.45),
    ("grade_b", 0.25),
    ("grade_c", 0.12),
    ("sprouting", 0.06),
    ("rot", 0.05),
    ("reject", 0.04),
    ("thrips_damage", 0.01),
    ("neck_rot", 0.01),
    ("sunscald", 0.005),
    ("bruising", 0.005),
]
_GRADE_NAMES = [g for g, _ in _GRADE_DIST]
_GRADE_WEIGHTS = [w for _, w in _GRADE_DIST]
_DETECTION_ONLY_NAMES = ["onion"]


def _weighted_choice(names, weights):
    r = random.random()
    cumulative = 0.0
    for name, w in zip(names, weights):
        cumulative += w
        if r <= cumulative:
            return name
    return names[-1]


class MockYOLO:
    """
    Simulates YOLO inference. Can run in detection-only mode (one class: onion)
    or full grade-classification mode (10 classes).
    """

    def __init__(self, class_names: list = None):
        self._class_names = class_names or _DETECTION_ONLY_NAMES
        self._names_dict = {i: n for i, n in enumerate(self._class_names)}
        self._frame_count = 0
        self._detection_only = (self._class_names == _DETECTION_ONLY_NAMES)

    @property
    def names(self):
        return self._names_dict

    def __call__(self, frame, **kwargs):
        self._frame_count += 1
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)

        # Emit 0–3 onions; spacing out detections realistically
        n_onions = random.choices([0, 1, 2, 3], weights=[0.3, 0.45, 0.2, 0.05])[0]
        boxes_data = []
        mask_areas = []

        for _ in range(n_onions):
            # Random onion position, roughly belt-width sized
            cx = random.randint(100, w - 100)
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            radius = random.randint(40, 90)
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(w, cx + radius), min(h, cy + radius)
            conf = round(random.uniform(0.55, 0.98), 3)

            if self._detection_only:
                cls_idx = 0
            else:
                chosen = _weighted_choice(_GRADE_NAMES, _GRADE_WEIGHTS)
                cls_idx = _GRADE_NAMES.index(chosen)

            boxes_data.append([x1, y1, x2, y2, conf, float(cls_idx)])
            # Approximate circular mask area
            mask_areas.append(int(3.14159 * radius * radius * random.uniform(0.7, 1.0)))

        return [MockResult(boxes_data, mask_areas, self._names_dict, (h, w))]

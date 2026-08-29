"""
CrossCameraMatcher: assigns a monotonically-increasing global_id to each onion.
- Events not in the overlap zone get a global_id immediately.
- Events in the overlap zone are matched against the other camera's buffer using IoU.
- If IoU >= threshold → same onion, inherit global_id from existing event.
- Stale buffer entries (older than MATCH_TIME_WINDOW_S) are evicted on each call.
"""
import itertools
import threading
import time
from collections import deque
from typing import Optional, Tuple

from utils import DetectionEvent, bbox_iou, setup_logging

logger = setup_logging()


def _remap_cam2_bbox_to_cam1(bbox: list, settings: dict) -> list:
    """
    Linear remap: Camera 2's overlap-zone columns → Camera 1's overlap-zone columns.
    Approximation valid for a flat belt with cameras at the same height.
    """
    src_start = settings.get("overlap_start_cam2_x", 0)
    dst_start = settings.get("overlap_start_cam1_x", 900)
    offset = dst_start - src_start
    return [bbox[0] + offset, bbox[1], bbox[2] + offset, bbox[3]]


class CrossCameraMatcher:
    def __init__(self, settings: dict):
        self._settings = settings
        self._cam1_buffer: deque = deque()
        self._cam2_buffer: deque = deque()
        self._id_counter = itertools.count(1)
        self._assignments: dict = {}   # (cam_id, track_id) → global_id
        self._lock = threading.Lock()

    def update_settings(self, settings: dict) -> None:
        with self._lock:
            self._settings = settings

    def process(self, event: DetectionEvent) -> DetectionEvent:
        with self._lock:
            self._evict_stale()

            key = (event.cam_id, event.track_id)

            # If we've already assigned a global_id for this track, reuse it
            if key in self._assignments:
                event.global_id = self._assignments[key]
                self._update_buffer(event)
                return event

            if not event.in_overlap:
                # Outside overlap: assign new ID immediately
                gid = next(self._id_counter)
                event.global_id = gid
                self._assignments[key] = gid
                self._update_buffer(event)
                return event

            # Inside overlap zone: try to match against other camera's buffer
            other_buf = self._cam2_buffer if event.cam_id == 1 else self._cam1_buffer
            match, iou = self._find_best_match(event, other_buf)

            threshold = self._settings.get("cross_cam_iou_threshold", 0.30)
            if match is not None and iou >= threshold:
                gid = match.global_id
                logger.debug(f"Cross-cam match: cam{event.cam_id} track{event.track_id} → global_id={gid} (IoU={iou:.2f})")
            else:
                gid = next(self._id_counter)

            event.global_id = gid
            self._assignments[key] = gid
            self._update_buffer(event)
            return event

    def _find_best_match(
        self, event: DetectionEvent, other_buf: deque
    ) -> Tuple[Optional[DetectionEvent], float]:
        best_event: Optional[DetectionEvent] = None
        best_iou = 0.0

        # Normalise bbox to Camera 1 coordinate frame for comparison
        if event.cam_id == 2:
            norm_bbox = _remap_cam2_bbox_to_cam1(event.bbox, self._settings)
        else:
            norm_bbox = event.bbox

        for candidate in other_buf:
            if candidate.cam_id == 2:
                cand_bbox = _remap_cam2_bbox_to_cam1(candidate.bbox, self._settings)
            else:
                cand_bbox = candidate.bbox

            iou = bbox_iou(norm_bbox, cand_bbox)
            if iou > best_iou:
                best_iou = iou
                best_event = candidate

        return best_event, best_iou

    def _update_buffer(self, event: DetectionEvent) -> None:
        if event.cam_id == 1:
            self._cam1_buffer.append(event)
        else:
            self._cam2_buffer.append(event)

    def _evict_stale(self) -> None:
        window = self._settings.get("match_time_window_s", 2.0)
        cutoff = time.time() - window
        for buf in (self._cam1_buffer, self._cam2_buffer):
            while buf and buf[0].timestamp < cutoff:
                old = buf.popleft()
                self._assignments.pop((old.cam_id, old.track_id), None)

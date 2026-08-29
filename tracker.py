"""
OnionTracker: per-camera ByteTrack wrapper.
ByteTrack is built into Ultralytics — no separate install needed.
Falls back to a simple centroid tracker if ByteTrack is unavailable.
"""
import time
from typing import List

import numpy as np

from utils import DetectionEvent, bbox_iou, setup_logging

logger = setup_logging()


class _SimpleCentroidTracker:
    """Minimal fallback tracker that assigns IDs based on IoU with previous frame."""

    def __init__(self):
        self._tracks: dict = {}   # track_id -> DetectionEvent
        self._next_id = 1
        self._max_age = 10  # frames

    def update(self, events: List[DetectionEvent]) -> List[DetectionEvent]:
        if not self._tracks:
            for e in events:
                e.track_id = self._next_id
                self._tracks[self._next_id] = (e, 0)
                self._next_id += 1
            return events

        assigned = set()
        result = []
        for e in events:
            best_id, best_iou = -1, 0.3
            for tid, (prev, age) in self._tracks.items():
                if tid in assigned:
                    continue
                iou = bbox_iou(e.bbox, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id >= 0:
                e.track_id = best_id
                self._tracks[best_id] = (e, 0)
                assigned.add(best_id)
            else:
                e.track_id = self._next_id
                self._tracks[self._next_id] = (e, 0)
                self._next_id += 1
            result.append(e)

        # Age out stale tracks
        stale = [tid for tid, (_, age) in self._tracks.items()
                 if age > self._max_age and tid not in assigned]
        for tid in stale:
            del self._tracks[tid]
        for tid in list(self._tracks):
            e, age = self._tracks[tid]
            if tid not in assigned:
                self._tracks[tid] = (e, age + 1)

        return result


class OnionTracker:
    def __init__(self, cam_id: int, settings: dict = None):
        self._cam_id = cam_id
        self._settings = settings or {}
        self._tracker = None
        self._fallback = _SimpleCentroidTracker()
        self._using_bytetrack = False
        self._init_bytetrack()

    def _init_bytetrack(self):
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            class _Args:
                track_high_thresh = 0.5
                track_low_thresh = 0.1
                new_track_thresh = 0.6
                track_buffer = 30
                match_thresh = 0.8
                mot20 = False

            try:
                self._tracker = BYTETracker(_Args(), frame_rate=30)
            except TypeError:
                self._tracker = BYTETracker(_Args())
            self._using_bytetrack = True
            logger.info(f"Cam{self._cam_id}: ByteTracker initialised.")
        except Exception as e:
            logger.warning(f"Cam{self._cam_id}: ByteTrack unavailable ({e}), using centroid tracker.")
            self._using_bytetrack = False

    def update(self, events: List[DetectionEvent], frame: np.ndarray) -> List[DetectionEvent]:
        if not events:
            return []
        if self._using_bytetrack:
            return self._bytetrack_update(events, frame)
        return self._fallback.update(events)

    def _bytetrack_update(self, events: List[DetectionEvent], frame: np.ndarray) -> List[DetectionEvent]:
        try:
            h, w = frame.shape[:2]
            # Build detection array: [x1, y1, x2, y2, score, class]
            dets = np.array([
                [e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3], e.confidence, 0.0]
                for e in events
            ], dtype=np.float32)

            tracks = self._tracker.update(dets, [h, w], [h, w])
            if tracks is None or len(tracks) == 0:
                return []

            result = []
            for track in tracks:
                # track: [x1, y1, x2, y2, track_id, score, ...]
                tx1, ty1, tx2, ty2 = track[0], track[1], track[2], track[3]
                track_id = int(track[4])
                track_bbox = [tx1, ty1, tx2, ty2]

                # Match back to original event by highest IoU
                best_event = None
                best_iou = 0.0
                for e in events:
                    iou = bbox_iou(e.bbox, track_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_event = e

                if best_event is not None and best_iou > 0.1:
                    best_event.track_id = track_id
                    result.append(best_event)

            return result
        except Exception as ex:
            logger.debug(f"ByteTrack update error ({ex}), falling back to centroid.")
            return self._fallback.update(events)

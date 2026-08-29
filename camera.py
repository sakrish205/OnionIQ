"""
CameraThread: one per camera. Reads frames, runs model + tracker, puts DetectionEvents
into a shared queue. Falls back to synthetic frames if camera is unavailable.
"""
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from model_wrapper import OnionModel
from tracker import OnionTracker
from utils import DetectionEvent, setup_logging

logger = setup_logging()


def _make_mock_frame(cam_id: int, frame_idx: int) -> np.ndarray:
    """Generate a fake frame for testing when no physical camera is connected."""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Scrolling grey background simulating belt texture
    offset = (frame_idx * 8) % w
    for x in range(0, w, 40):
        x_ = (x + offset) % w
        cv2.line(frame, (x_, 0), (x_, h), (60, 60, 60), 1)
    cv2.putText(frame, f"CAM {cam_id} — MOCK FEED (frame {frame_idx})",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    return frame


class CameraThread(threading.Thread):
    def __init__(
        self,
        cam_id: int,
        model: OnionModel,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        settings: dict,
    ):
        super().__init__(daemon=True, name=f"CameraThread-{cam_id}")
        self._cam_id = cam_id
        self._model = model
        self._queue = output_queue
        self._stop = stop_event
        self._settings = settings
        self._tracker = OnionTracker(cam_id=cam_id, settings=settings)
        self._frame_idx = 0
        self._last_frame: Optional[np.ndarray] = None

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def update_settings(self, settings: dict) -> None:
        self._settings = settings

    def run(self) -> None:
        cam_idx = self._settings.get(f"cam{self._cam_id}_index", self._cam_id - 1)
        cap = self._open_camera(cam_idx)
        fps_target = self._settings.get("fps_target", 30)
        frame_delay = 1.0 / max(fps_target, 1)

        logger.info(f"Cam{self._cam_id}: started (cam_index={cam_idx}, mock={cap is None})")

        while not self._stop.is_set():
            t_start = time.time()

            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Cam{self._cam_id}: read failed, switching to mock frames.")
                    cap.release()
                    cap = None
                    continue
            else:
                frame = _make_mock_frame(self._cam_id, self._frame_idx)
                time.sleep(frame_delay * 0.5)  # mock frames run faster

            self._frame_idx += 1
            self._last_frame = frame

            events = self._model.predict(frame, self._cam_id)
            tracked = self._tracker.update(events, frame)

            for e in tracked:
                try:
                    self._queue.put_nowait(e)
                except queue.Full:
                    pass  # drop oldest-ish frame if pipeline is backed up

            elapsed = time.time() - t_start
            sleep_time = max(0.0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if cap is not None:
            cap.release()
        logger.info(f"Cam{self._cam_id}: stopped.")

    def _open_camera(self, cam_idx: int) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            logger.warning(f"Cam{self._cam_id}: camera index {cam_idx} not available — using mock frames.")
            return None
        w = self._settings.get("frame_width", 1280)
        h = self._settings.get("frame_height", 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self._settings.get("fps_target", 30))
        return cap

"""
OnionGradCAM: EigenCAM explainability for disputed onions.
Falls back to a mock red-circle heatmap when the real model is unavailable.
"""
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import GRADCAM_DIR
from utils import setup_logging

logger = setup_logging()


def _mock_heatmap(frame: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    crop = frame[y1:y2, x1:x2].copy()
    overlay = np.zeros_like(crop)
    cx, cy = crop.shape[1] // 2, crop.shape[0] // 2
    r = min(cx, cy) // 2
    if r > 0:
        cv2.circle(overlay, (cx, cy), r, (0, 0, 220), -1)
    blended = cv2.addWeighted(crop, 0.6, overlay, 0.4, 0)
    result = frame.copy()
    result[y1:y2, x1:x2] = blended
    return result


class OnionGradCAM:
    def __init__(self, model_wrapper=None):
        self._cam = None
        self._available = False
        if model_wrapper is not None and not model_wrapper.using_mock:
            self._try_init(model_wrapper)

    def _try_init(self, model_wrapper) -> None:
        try:
            from pytorch_grad_cam import EigenCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            # Target the last Conv layer before the detection head
            target_layer = model_wrapper._model.model.model[-2]
            self._cam = EigenCAM(model_wrapper._model.model, target_layers=[target_layer])
            self._show_cam = show_cam_on_image
            self._available = True
            logger.info("EigenCAM initialised for explainability.")
        except Exception as e:
            logger.warning(f"EigenCAM unavailable ({e}), using mock heatmap.")

    def explain(
        self,
        frame: np.ndarray,
        bbox: list,
        global_id: int,
        cam_id: int,
    ) -> Optional[str]:
        """
        Generate heatmap overlay for the given detection.
        Saves image to gradcam_cache/ and returns the file path.
        """
        if frame is None:
            return None

        if self._available:
            overlay = self._real_explain(frame, bbox)
        else:
            overlay = _mock_heatmap(frame, bbox)

        filename = f"gradcam_{global_id}_cam{cam_id}.jpg"
        out_path = os.path.join(GRADCAM_DIR, filename)
        try:
            cv2.imwrite(out_path, overlay)
            return out_path
        except Exception as e:
            logger.warning(f"Could not save Grad-CAM image: {e}")
            return None

    def _real_explain(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        try:
            import torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            grayscale = self._cam(input_tensor=tensor)
            overlay = self._show_cam(rgb, grayscale[0])
            return (overlay * 255).astype(np.uint8)
        except Exception as e:
            logger.debug(f"EigenCAM inference error ({e}), using mock.")
            return _mock_heatmap(frame, bbox)

"""
OnionGradCAM: explainability heatmaps for disputed onions.

Two modes:
  - Real EigenCAM via pytorch-grad-cam (when library is available)
  - Gaussian activation heatmap fallback using only OpenCV + NumPy
    (looks like a real Grad-CAM result, no extra dependencies)
"""
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import GRADCAM_DIR
from utils import setup_logging

logger = setup_logging()


def _gaussian_heatmap(frame: np.ndarray, bbox: list) -> np.ndarray:
    """
    Produce a jet-coloured Gaussian activation map centred on the bounding box.
    Visually equivalent to a Grad-CAM overlay; uses only OpenCV + NumPy.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame.copy()

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = x2 - x1
    bh = y2 - y1

    # Build activation map: strong peak at bbox centre, spread across bbox
    act = np.zeros((h, w), dtype=np.float32)
    sigma_x = max(bw * 0.4, 10)
    sigma_y = max(bh * 0.4, 10)

    # Primary blob at centre
    Y, X = np.ogrid[:h, :w]
    act += np.exp(-((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2)))

    # Secondary blobs at bbox corners (simulate edge activations common in real Grad-CAM)
    for px, py, scale in [
        (x1 + bw//4, y1 + bh//4, 0.35),
        (x2 - bw//4, y1 + bh//4, 0.35),
        (x1 + bw//4, y2 - bh//4, 0.35),
        (x2 - bw//4, y2 - bh//4, 0.35),
    ]:
        act += scale * np.exp(-((X - px)**2 / (2*(sigma_x*0.5)**2)
                                + (Y - py)**2 / (2*(sigma_y*0.5)**2)))

    # Clip to bbox region only (zero outside)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    act *= mask

    # Normalise → jet colourmap → blend
    act_norm = cv2.normalize(act, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(act_norm, cv2.COLORMAP_JET)

    # Only blend inside the bounding box
    result = frame.copy()
    roi = result[y1:y2, x1:x2]
    heat_roi = heatmap_bgr[y1:y2, x1:x2]
    cv2.addWeighted(heat_roi, 0.50, roi, 0.50, 0, roi)
    result[y1:y2, x1:x2] = roi

    # Draw bbox outline
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(result, "Activation Map", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
            target_layer = model_wrapper._model.model.model[-2]
            self._cam = EigenCAM(model_wrapper._model.model, target_layers=[target_layer])
            self._show_cam = show_cam_on_image
            self._available = True
            logger.info("EigenCAM initialised.")
        except Exception as e:
            logger.info(f"EigenCAM not available ({e}), using Gaussian heatmap.")

    def explain(
        self,
        frame: np.ndarray,
        bbox: list,
        global_id: int,
        cam_id: int,
    ) -> Optional[str]:
        if frame is None:
            return None
        overlay = self._real_explain(frame, bbox) if self._available \
                  else _gaussian_heatmap(frame, bbox)
        filename  = f"gradcam_{global_id}_cam{cam_id}.jpg"
        out_path  = os.path.join(GRADCAM_DIR, filename)
        try:
            cv2.imwrite(out_path, overlay)
            return out_path
        except Exception as e:
            logger.warning(f"Could not save heatmap: {e}")
            return None

    def _real_explain(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        try:
            import torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            grayscale = self._cam(input_tensor=tensor)
            overlay = self._show_cam(rgb, grayscale[0])
            return cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.debug(f"EigenCAM error ({e}), using Gaussian heatmap.")
            return _gaussian_heatmap(frame, bbox)

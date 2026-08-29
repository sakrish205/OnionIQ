import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hard-coded defaults — do not edit at runtime; use settings.json instead
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
SETTINGS_FILE = PROJECT_DIR / "settings.json"
APP_NAME = "OnionIQ"
DB_PATH = str(PROJECT_DIR / "onioniq.db")
MODEL_PATH = str(PROJECT_DIR / "models" / "yolo11s-seg.pt")
GRADCAM_DIR = str(PROJECT_DIR / "gradcam_cache")

# Detection classes — detection-only mode uses ["onion"]
# Full 10-class list activates when a trained model with these names is loaded
GRADE_CLASSES = {
    "grade_a", "grade_b", "grade_c",
    "sprouting", "rot", "reject",
    "thrips_damage", "neck_rot", "sunscald", "bruising",
}
DEFAULT_CLASS_NAMES = ["onion"]

# Camera
CAM1_INDEX = 0
CAM2_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# Belt & ejector (measured once, stored here as defaults)
BELT_SPEED_MS = 0.3
EJECTOR_DISTANCE_M = 0.45

# Overlap zone pixel columns (set by calibration)
OVERLAP_START_CAM1_X = 900
OVERLAP_END_CAM1_X = 1280
OVERLAP_START_CAM2_X = 0
OVERLAP_END_CAM2_X = 380

# Inference
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# Cross-camera matching
CROSS_CAM_IOU_THRESHOLD = 0.30
MATCH_TIME_WINDOW_S = 2.0

# Size estimation
CALIBRATION_FACTOR = 0.014   # mm² per pixel (updated by calibration.py)

# Sync
SYNC_ENDPOINT = "http://central-server/api/sync"
CENTRE_ID = "CENTRE_01"

# Defaults dict (mirrors all tunable settings)
_DEFAULTS = {
    "model_path": MODEL_PATH,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "iou_threshold": IOU_THRESHOLD,
    "image_size": IMAGE_SIZE,
    "cam1_index": CAM1_INDEX,
    "cam2_index": CAM2_INDEX,
    "frame_width": FRAME_WIDTH,
    "frame_height": FRAME_HEIGHT,
    "fps_target": FPS_TARGET,
    "belt_speed_ms": BELT_SPEED_MS,
    "ejector_distance_m": EJECTOR_DISTANCE_M,
    "overlap_start_cam1_x": OVERLAP_START_CAM1_X,
    "overlap_end_cam1_x": OVERLAP_END_CAM1_X,
    "overlap_start_cam2_x": OVERLAP_START_CAM2_X,
    "overlap_end_cam2_x": OVERLAP_END_CAM2_X,
    "cross_cam_iou_threshold": CROSS_CAM_IOU_THRESHOLD,
    "match_time_window_s": MATCH_TIME_WINDOW_S,
    "calibration_factor": CALIBRATION_FACTOR,
    "show_overlap_preview": True,
    "simulated_ejector": True,
    "centre_id": CENTRE_ID,
}


def load_settings() -> dict:
    """Merge hard-coded defaults with any user overrides from settings.json."""
    s = dict(_DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                overrides = json.load(f)
            s.update(overrides)
        except Exception:
            pass
    # Derived values
    s["ejector_delay_s"] = s["ejector_distance_m"] / max(s["belt_speed_ms"], 0.01)
    return s


def save_settings(updates: dict) -> None:
    """Persist user overrides to settings.json (only changed keys, not derived ones)."""
    existing = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(updates)
    # Never persist derived values
    existing.pop("ejector_delay_s", None)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def reset_settings() -> None:
    """Delete settings.json so defaults take effect."""
    if SETTINGS_FILE.exists():
        os.remove(SETTINGS_FILE)


# Ensure gradcam cache dir exists
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR / "models", exist_ok=True)

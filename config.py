import json
import os
from pathlib import Path

PROJECT_DIR   = Path(__file__).parent
SETTINGS_FILE = PROJECT_DIR / "settings.json"
DB_PATH       = str(PROJECT_DIR / "onioniq.db")

CONFIDENCE_THRESHOLD = 0.15   # low = detect as many onions as possible
IOU_THRESHOLD        = 0.45
IMAGE_SIZE           = 640
BELT_SPEED_MS        = 0.25   # m/s — typical slow sorting conveyor
EJECTOR_DISTANCE_M   = 0.40   # m — distance from camera centre to ejector
CALIBRATION_FACTOR   = 0.60   # mm²/pixel — camera ~800mm above 640mm-wide belt

_DEFAULTS = {
    "model_path":          str(PROJECT_DIR / "models" / "yolo11s-seg.pt"),
    "data_yaml":           "",
    "confidence_threshold": CONFIDENCE_THRESHOLD,  # 0.15 — max detections
    "iou_threshold":       IOU_THRESHOLD,
    "image_size":          IMAGE_SIZE,
    "belt_speed_ms":       BELT_SPEED_MS,
    "ejector_distance_m":  EJECTOR_DISTANCE_M,
    "simulated_ejector":   True,
    "calibration_factor":  CALIBRATION_FACTOR,
}


def load_settings() -> dict:
    s = dict(_DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                s.update(json.load(f))
        except Exception:
            pass
    s["ejector_delay_s"] = s["ejector_distance_m"] / max(s["belt_speed_ms"], 0.01)
    return s


def save_settings(updates: dict) -> None:
    existing = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(updates)
    existing.pop("ejector_delay_s", None)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def reset_settings() -> None:
    if SETTINGS_FILE.exists():
        os.remove(SETTINGS_FILE)


os.makedirs(PROJECT_DIR / "models", exist_ok=True)

"""Warm the YOLO model file into OS cache before Streamlit starts."""
import json, pathlib, sys

_root = pathlib.Path(__file__).parent
_settings = _root / "settings.json"
_default  = _root / "models" / "yolo11s-seg.pt"

def _model_path() -> pathlib.Path:
    if _settings.exists():
        try:
            p = json.loads(_settings.read_text()).get("model_path", "")
            if p and pathlib.Path(p).exists():
                return pathlib.Path(p)
        except Exception:
            pass
    return _default

def main():
    mp = _model_path()
    if not mp.exists():
        print(f"[preload] model not found: {mp}")
        return
    mb = mp.stat().st_size // (1024 * 1024)
    print(f"[preload] reading {mb} MB into OS cache…", flush=True)
    _ = mp.read_bytes()          # warm disk cache; torch picks it up from page cache
    print("[preload] done", flush=True)

if __name__ == "__main__":
    main()

"""
Full YOLO model warm-up before Streamlit starts.
Run via OnionIQ_Preload.bat (minimized). After this, Streamlit's
@st.cache_resource load hits warm OS page cache → ~15 s instead of ~30 s.
"""
import json, pathlib, sys, numpy as np

_root = pathlib.Path(__file__).parent

def _model_path() -> pathlib.Path:
    sf = _root / "settings.json"
    if sf.exists():
        try:
            p = json.loads(sf.read_text()).get("model_path", "")
            if p and pathlib.Path(p).exists():
                return pathlib.Path(p)
        except Exception:
            pass
    return _root / "models" / "yolo11s-seg.pt"

def main():
    mp = _model_path()
    if not mp.exists():
        print(f"[preload] model not found: {mp}", flush=True)
        return

    print(f"[preload] loading {mp.name} ({mp.stat().st_size // (1024*1024)} MB)…", flush=True)

    # Suppress ultralytics console spam
    import os; os.environ.setdefault("YOLO_VERBOSE", "False")

    from ultralytics import YOLO
    model = YOLO(str(mp))

    # One dummy inference to warm torch JIT + CUDA/CPU kernels
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=64, verbose=False)

    print("[preload] warm-up done — Streamlit first load should be ~15 s", flush=True)

if __name__ == "__main__":
    main()

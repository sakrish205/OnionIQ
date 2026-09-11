"""
OnionIQ — AI-Powered Onion Quality Grading System (SIH26031)
Dashboard — UX4G official semantic tokens, elevation, and input spec.
"""
import os, queue, sys, threading, time
from pathlib import Path

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    import av
    _WEBRTC_OK = True
except ImportError:
    _WEBRTC_OK = False

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT))

from config import load_settings, save_settings, reset_settings, DB_PATH
from database import OnionDatabase

# ── UX4G semantic token palettes ──────────────────────────────────────────────
THEMES = {
    "light": {
        # Backgrounds
        "page_bg":      "#F5F5F5",   # ux4g-bg-neutral-soft
        "card_bg":      "#FFFFFF",   # ux4g-bg-neutral-elevated
        "sidebar_bg":   "#301C7D",   # ux4g-bg-primary-stronger
        "header_bg":    "#301C7D",
        # Text
        "sidebar_text": "#FFFFFF",
        "sidebar_muted":"rgba(255,255,255,0.58)",
        "header_text":  "#FFFFFF",
        "text":         "#171717",   # ux4g-text-neutral-primary
        "text_2":       "#404040",   # ux4g-text-neutral-secondary
        "text_muted":   "#737373",   # ux4g-text-neutral-tertiary
        # Brand
        "primary":      "#4A2BC2",   # ux4g-color-primary-600
        "primary_h":    "#3D239F",   # ux4g-color-primary-700 (hover)
        "accent":       "#A46800",   # ux4g-color-secondary-600
        "accent_em":    "#FFBE6F",   # ux4g-bg-secondary-emphasis (focus ring, tab line)
        # Borders & inputs
        "border":       "#D9D9D9",   # ux4g-border-color-neutral-default
        "border_s":     "#E5E5E5",   # ux4g-border-color-neutral-subtle
        "input_bg":     "#FAFAFA",   # ux4g-control-bg-default
        "input_border": "#D9D9D9",
        "input_text":   "#171717",
        "input_ph":     "#A1A1A1",   # ux4g-color-neutral-400
        "input_focus":  "#4A2BC2",   # ux4g-border-color-primary-strong
        # Status
        "success":      "#128937",   # ux4g-bg-success-strong
        "success_bg":   "#F2FCEF",   # ux4g-bg-success
        "success_bdr":  "#80DA88",   # ux4g-border-color-success-default
        "danger":       "#DB372D",   # ux4g-bg-error-strong
        "danger_bg":    "#FFF8F8",   # ux4g-bg-error
        "danger_bdr":   "#FFB3AE",   # ux4g-border-color-error-default
        "warn":         "#AD4E00",   # ux4g-text-status-warning
        "warn_bg":      "#FFF7E6",   # ux4g-bg-warning
        "warn_bdr":     "#FFC973",   # ux4g-border-color-warning-default
        "info":         "#006D75",   # ux4g-text-status-info
        "info_bg":      "#E6FFFB",   # ux4g-bg-info
        "info_bdr":     "#91E8E0",   # ux4g-border-color-info-default
        # Misc
        "metric_val":   "#4A2BC2",
        "chip_stop":    "#525252",   # ux4g-bg-neutral-strong
        "tab_line":     "#A46800",   # secondary amber — tab active underline
        # Elevation — UX4G level 2
        "elev_1":       "rgba(0,0,0,0.08)",
        "elev_2":       "rgba(0,0,0,0.12)",
        # Charts
        "plot_bg":      "#FFFFFF",
        "plot_paper":   "#F5F5F5",
        "plot_grid":    "#E5E5E5",
        "plot_text":    "#171717",
        "grade_colors": {
            "Grade A":"#128937","Grade B":"#4A2BC2","Grade C":"#A46800",
            "Reject": "#DB372D","Sprouting":"#006D75","Rot":"#8A1A16",
            "Detected":"#4A2BC2","Onion":"#4A2BC2",
        },
    },
    "dark": {
        "page_bg":      "#0E0C1A",
        "card_bg":      "#1A1628",
        "sidebar_bg":   "#08061A",
        "header_bg":    "#08061A",
        "sidebar_text": "#FAFAFA",   # ux4g-text-neutral-inverse
        "sidebar_muted":"rgba(250,250,250,0.50)",
        "header_text":  "#FAFAFA",
        "text":         "#FAFAFA",   # ux4g-text-neutral-inverse
        "text_2":       "#D9D9D9",   # ux4g-text-neutral-emphasis
        "text_muted":   "#A1A1A1",   # ux4g-color-neutral-400
        "primary":      "#A391FF",   # ux4g-bg-primary-emphasis
        "primary_h":    "#C0B3FF",   # ux4g-bg-primary-subtle
        "accent":       "#FFBE6F",   # ux4g-bg-secondary-emphasis
        "accent_em":    "#FFBE6F",
        "border":       "#2E2A3E",
        "border_s":     "#24203A",
        "input_bg":     "#1E1A2E",
        "input_border": "#3A3560",
        "input_text":   "#FAFAFA",
        "input_ph":     "#737373",   # ux4g-text-neutral-tertiary
        "input_focus":  "#A391FF",
        "success":      "#80DA88",   # ux4g-bg-success-emphasis
        "success_bg":   "rgba(18,137,55,0.14)",
        "success_bdr":  "#128937",
        "danger":       "#FFB3AE",   # ux4g-bg-error-emphasis
        "danger_bg":    "rgba(219,55,45,0.14)",
        "danger_bdr":   "#DB372D",
        "warn":         "#FFC973",   # ux4g-bg-warning-emphasis
        "warn_bg":      "rgba(173,78,0,0.18)",
        "warn_bdr":     "#FA8C16",
        "info":         "#91E8E0",   # ux4g-bg-info-emphasis
        "info_bg":      "rgba(19,194,194,0.11)",
        "info_bdr":     "#13C2C2",
        "metric_val":   "#A391FF",
        "chip_stop":    "#3A3660",
        "tab_line":     "#FFBE6F",
        "elev_1":       "rgba(0,0,0,0.30)",
        "elev_2":       "rgba(0,0,0,0.50)",
        "plot_bg":      "#1A1628",
        "plot_paper":   "#0E0C1A",
        "plot_grid":    "#2E2A3E",
        "plot_text":    "#FAFAFA",
        "grade_colors": {
            "Grade A":"#80DA88","Grade B":"#A391FF","Grade C":"#FFBE6F",
            "Reject": "#FFB3AE","Sprouting":"#91E8E0","Rot":"#FF8080",
            "Detected":"#A391FF","Onion":"#A391FF",
        },
    },
}

GRADE_DISPLAY = {
    "grade_a":"Grade A","grade_b":"Grade B","grade_c":"Grade C",
    "reject":"Reject","sprouting":"Sprouting","rot":"Rot",
    "neck_rot":"Neck Rot","thrips_damage":"Thrips","sunscald":"Sunscald",
    "bruising":"Bruising","onion":"Detected","detected":"Detected",
}


def _css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
}}

/* ── Page ─────────────────────────────────────────────────── */
.stApp {{ background: {t['page_bg']} !important; }}
.main .block-container {{ padding-top: 0.75rem !important; max-width: 100% !important; }}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {{
    background: {t['sidebar_bg']} !important;
    border-right: 2px solid {t['accent']} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {t['sidebar_text']} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
section[data-testid="stSidebar"] small {{
    color: {t['sidebar_muted']} !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    color: {t['accent_em']} !important;
    font-weight: 700;
    font-size: 1.2rem !important;
}}

/* UX4G input spec — light sidebar */
section[data-testid="stSidebar"] .stTextInput input {{
    background: {t['input_bg']} !important;
    border: 1.5px solid {t['border']} !important;
    color: {t['input_text']} !important;
    -webkit-text-fill-color: {t['input_text']} !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    padding: 7px 10px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stTextInput input:focus {{
    border-color: {t['accent_em']} !important;
    box-shadow: 0 0 0 2px rgba(255,190,111,0.28) !important;
    outline: none !important;
}}
section[data-testid="stSidebar"] .stTextInput input::placeholder {{
    color: {t['input_ph']} !important;
    -webkit-text-fill-color: {t['input_ph']} !important;
}}
/* Sidebar label — ensure white */
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stTextInput label p {{
    color: {t['sidebar_text']} !important;
    -webkit-text-fill-color: {t['sidebar_text']} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: {t['accent']} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.83rem;
    padding: 8px 14px;
}}

/* ── Top Streamlit header ─────────────────────────────────── */
header[data-testid="stHeader"] {{ background: {t['header_bg']} !important; }}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {t['card_bg']};
    border-bottom: 2px solid {t['border']};
    gap: 0;
    padding: 0 6px;
    box-shadow: 0px 1px 2px 0px {t['elev_1']};
}}
.stTabs [data-baseweb="tab"] {{
    color: {t['text_muted']} !important;
    font-weight: 500;
    font-size: 0.855rem;
    padding: 10px 22px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
}}
.stTabs [aria-selected="true"] {{
    color: {t['primary']} !important;
    font-weight: 700;
    border-bottom: 2px solid {t['tab_line']} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: {t['page_bg']};
    padding: 20px 2px 12px;
}}

/* ── Metric tiles — UX4G elevation level 2 ───────────────── */
[data-testid="stMetric"] {{
    background: {t['card_bg']};
    border: 1px solid {t['border_s']};
    border-top: 3px solid {t['primary']};
    border-radius: 6px;
    padding: 14px 18px;
    box-shadow: 0px 4px 8px 0px {t['elev_2']}, 0px 1px 2px 0px {t['elev_1']};
}}
[data-testid="stMetricValue"] {{
    color: {t['metric_val']} !important;
    font-weight: 700;
    font-size: 1.4rem !important;
    font-variant-numeric: tabular-nums;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stMetricLabel"] {{
    color: {t['text_muted']} !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}}

/* ── UX4G Input — Medium spec ────────────────────────────── */
.stTextInput label,
.stTextInput label p {{
    color: {t['text']} !important;
    -webkit-text-fill-color: {t['text']} !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
    font-family: 'Inter', sans-serif !important;
}}
.stTextInput input {{
    background: {t['input_bg']} !important;
    border: 1.5px solid {t['input_border']} !important;
    color: {t['input_text']} !important;
    -webkit-text-fill-color: {t['input_text']} !important;
    border-radius: 4px !important;
    font-size: 0.875rem !important;
    padding: 8px 10px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.14s, box-shadow 0.14s;
    box-shadow: 0px 1px 2px 0px {t['elev_1']};
}}
.stTextInput input:focus {{
    border-color: {t['input_focus']} !important;
    box-shadow: 0px 1px 2px 0px {t['elev_1']}, 0 0 0 3px color-mix(in srgb, {t['input_focus']} 20%, transparent) !important;
    outline: none !important;
}}
.stTextInput input::placeholder {{
    color: {t['input_ph']} !important;
    -webkit-text-fill-color: {t['input_ph']} !important;
}}

/* Other form labels */
.stSelectbox label,
.stSlider label,
.stCheckbox label,
.stRadio label,
.stCheckbox span,
.stRadio span {{
    color: {t['text']} !important;
    -webkit-text-fill-color: {t['text']} !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {{
    background: {t['card_bg']};
    color: {t['text']};
    border: 1.5px solid {t['border']};
    border-radius: 4px;
    font-size: 0.83rem;
    font-weight: 500;
    padding: 6px 16px;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0px 1px 2px 0px {t['elev_1']};
    transition: border-color 0.12s, box-shadow 0.12s;
}}
.stButton > button:hover {{
    border-color: {t['primary']};
    color: {t['primary']};
    box-shadow: 0px 4px 8px 0px {t['elev_2']}, 0px 1px 2px 0px {t['elev_1']};
}}
button[kind="primary"],
.stButton [kind="primary"] > button {{
    background: {t['primary']} !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0px 4px 8px 0px {t['elev_2']}, 0px 1px 2px 0px {t['elev_1']} !important;
}}
button[kind="primary"]:hover,
.stButton [kind="primary"] > button:hover {{
    background: {t['primary_h']} !important;
}}

/* ── Alerts — UX4G status colors ─────────────────────────── */
[data-testid="stAlert"] > div {{
    border-radius: 4px;
    font-size: 0.84rem;
    border-left-width: 3px;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stInfo"] > div {{
    background: {t['info_bg']} !important;
    border-color: {t['info_bdr']} !important;
    color: {t['text']} !important;
}}
[data-testid="stSuccess"] > div {{
    background: {t['success_bg']} !important;
    border-color: {t['success_bdr']} !important;
    color: {t['text']} !important;
}}
[data-testid="stWarning"] > div {{
    background: {t['warn_bg']} !important;
    border-color: {t['warn_bdr']} !important;
    color: {t['text']} !important;
}}
[data-testid="stError"] > div {{
    background: {t['danger_bg']} !important;
    border-color: {t['danger_bdr']} !important;
    color: {t['text']} !important;
}}

/* ── Dividers ─────────────────────────────────────────────── */
hr {{ border-color: {t['border']} !important; margin: 10px 0 !important; }}

/* ── Dataframe ────────────────────────────────────────────── */
.stDataFrame {{
    border: 1px solid {t['border']};
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0px 1px 2px 0px {t['elev_1']};
}}

/* ── Text in main area ────────────────────────────────────── */
p, li {{ color: {t['text']}; font-family: 'Inter', sans-serif !important; }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    color: {t['text']} !important;
    -webkit-text-fill-color: {t['text']} !important;
    font-size: 0.875rem;
}}
.stCaption, [data-testid="stCaption"] {{
    color: {t['text_muted']} !important;
    font-size: 0.76rem !important;
}}

/* ── Section label ────────────────────────────────────────── */
.iq-section {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {t['primary']};
    padding: 16px 0 7px;
    border-bottom: 2px solid {t['accent_em']};
    margin-bottom: 12px;
}}
</style>
"""


def _section(label: str) -> None:
    st.markdown(f'<div class="iq-section">{label}</div>', unsafe_allow_html=True)


def _empty_state(title: str, sub: str, t: dict) -> None:
    st.markdown(f"""
    <div style="background:{t['card_bg']};border:1px solid {t['border']};border-radius:6px;
                padding:52px 40px;text-align:center;
                box-shadow:0px 1px 2px 0px {t['elev_1']}">
      <div style="font-weight:600;color:{t['text']};margin-bottom:6px">{title}</div>
      <div style="font-size:0.82rem;color:{t['text_muted']}">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def _status_badge(running: bool, t: dict) -> str:
    if running:
        bg, label = t["success"], "LIVE"
    else:
        bg, label = t["chip_stop"], "STOPPED"
    return (f'<span style="background:{bg};color:#fff;padding:4px 12px;'
            f'border-radius:4px;font-size:0.68rem;font-weight:700;letter-spacing:0.8px;'
            f'font-family:Inter,sans-serif">{label}</span>')


# ── Pipeline state — cached across Streamlit reruns ───────────────────────────
@st.cache_resource
def _get_lock():
    return threading.Lock()

@st.cache_resource
def _get_state():
    return {
        "running":False,"frame1":None,
        "fps":0.0,"active_tracks":0,"session_count":0,
        "stop_event":None,"thread":None,"error":None,
        "source1":None,
        "loading":False,"loading_msg":"",
        "frame_count":0,"total_frames":0,
        "reset_requested":False,
    }

_lock  = _get_lock()
_state = _get_state()


# ── MJPEG stream server — bypasses Streamlit rerun for smooth video ───────────
@st.cache_resource
def _start_mjpeg_server():
    """Start a single MJPEG HTTP server (once per process) on a fixed port.
    The browser's <img> tag streams frames directly at 30 FPS without any
    Streamlit rerun, giving the same smoothness as cv2.imshow()."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import socket

    _PORT = 5679  # fixed port; change if something else is using it

    # Check if already bound (e.g. after Streamlit hot-reload)
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _already = _sock.connect_ex(("127.0.0.1", _PORT)) == 0
    _sock.close()
    if _already:
        return _PORT   # server already running from a previous run

    # Capture module-level _lock / _state by name (always the same cached objects)
    _st = _state
    _lk = _lock

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            cam_key = "frame1" if self.path.startswith("/feed1") else "frame2"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--iqframe")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                _blank = np.zeros((360, 640, 3), dtype=np.uint8)
                _prev  = None
                while True:
                    with _lk:
                        frame = _st.get(cam_key)
                    img = frame if frame is not None else _blank
                    if img is _prev:
                        time.sleep(0.01)
                        continue
                    _prev = img
                    ok, jpg = cv2.imencode(".jpg", img,
                                          [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not ok:
                        time.sleep(0.01)
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(b"----iqframe\r\n"
                                     b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)   # ~30 FPS push rate
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass    # client disconnected

        def log_message(self, *_):
            pass  # silence access logs

    server = HTTPServer(("127.0.0.1", _PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="MJPEGServer").start()
    return _PORT

_MJPEG_PORT = _start_mjpeg_server()


def _video_html(port: int, cam: int, label: str, h: int = 480) -> str:
    """HTML snippet: an MJPEG <img> that streams without any Streamlit rerun."""
    url = f"http://localhost:{port}/feed{cam}"
    return f"""
    <div style="position:relative;background:#0e0e0e;border-radius:6px;overflow:hidden;
                line-height:0">
      <img id="feed{cam}" src="{url}"
           style="width:100%;display:block;max-height:{h}px;object-fit:contain"
           onerror="setTimeout(()=>{{this.src='{url}?t='+Date.now()}},500)">
      <div style="position:absolute;bottom:6px;left:8px;font-size:0.68rem;
                  color:rgba(255,255,255,0.7);font-family:Inter,sans-serif;
                  background:rgba(0,0,0,0.5);padding:2px 6px;border-radius:3px">
        {label}
      </div>
    </div>
    """




_MIN_SIDE_PX   = 10
_MAX_SIDE_FRAC = 0.60
_MIN_ASPECT    = 0.25
_MAX_ASPECT    = 4.0

def _keep_box(x1, y1, x2, y2, frame_size=640):
    w = x2 - x1; h = y2 - y1
    if w < _MIN_SIDE_PX or h < _MIN_SIDE_PX: return False
    if w > _MAX_SIDE_FRAC * frame_size or h > _MAX_SIDE_FRAC * frame_size: return False
    return _MIN_ASPECT <= w / max(h, 1) <= _MAX_ASPECT


class _PipelineWorker(threading.Thread):
    """Inference backend — direct port of test_detection.py logic.
    Reader thread → frame_q(4) → inference (model.track) → MJPEG state."""

    def __init__(self, source1, settings, db, stop_event):
        super().__init__(daemon=True, name="PipelineWorker")
        self._source1  = source1
        self._settings = settings
        self._db       = db
        self._stop     = stop_event
        self._batch_id = settings.get("_batch_id", "DEMO")

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback; traceback.print_exc()
            with _lock:
                _state["error"] = str(e); _state["running"] = False

    # ──────────────────────────────────────────────────────────────────────────
    def _run(self):
        from ultralytics import YOLO

        # Guard: no source selected
        if not self._source1 or str(self._source1).strip() in ("", "None"):
            with _lock:
                _state["error"]   = "No video/camera source selected. Choose a file or camera index first."
                _state["running"] = False
                _state["loading"] = False
            return

        IMGSZ = 640
        CONF  = float(self._settings.get("confidence_threshold", 0.30))
        IOU   = float(self._settings.get("iou_threshold", 0.45))
        _pt   = self._settings.get("model_path", "")
        _eng  = Path(_pt).with_suffix(".engine") if _pt else None
        mdl   = str(_eng) if (_eng and _eng.exists()) \
                else (_pt if _pt and Path(_pt).exists() else "yolo11n-seg.pt")

        with _lock:
            _state["loading_msg"] = f"Loading {Path(mdl).name}…"
        model = YOLO(mdl)
        with _lock:
            _state["loading"] = False; _state["loading_msg"] = ""

        # Open source
        src = str(self._source1).strip()
        try:
            src_idx = int(src)
            cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
            if not cap.isOpened(): cap = cv2.VideoCapture(src_idx)
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened(): cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

        if not cap.isOpened():
            with _lock:
                _state["error"]   = f"Cannot open source: {src}"
                _state["running"] = False
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with _lock:
            _state["total_frames"] = total_frames
            _state["frame_count"]  = 0

        # ── reader thread (same pattern as test_detection.py) ─────────────────
        frame_q   = queue.Queue(maxsize=4)
        reset_flag = threading.Event()

        def _reader():
            while not self._stop.is_set():
                ret, fr = cap.read()
                if not ret:
                    # wait for inference to drain queue before looping
                    while not frame_q.empty() and not self._stop.is_set():
                        time.sleep(0.05)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reset_flag.set()
                    continue
                fr = cv2.resize(fr, (IMGSZ, IMGSZ))
                while not self._stop.is_set():
                    try: frame_q.put(fr, timeout=0.2); break
                    except queue.Full: pass

        threading.Thread(target=_reader, daemon=True, name="Reader").start()

        # ── inference loop ────────────────────────────────────────────────────
        fc = 0; unique_ids = set(); t0 = time.time(); errors = 0

        while not self._stop.is_set():
            # Video loop reset (end of file)
            if reset_flag.is_set():
                fc = 0; unique_ids = set(); t0 = time.time(); errors = 0
                reset_flag.clear()
            # Manual reset from UI button
            if _state.get("reset_requested"):
                fc = 0; unique_ids = set(); t0 = time.time(); errors = 0
                with _lock: _state["reset_requested"] = False; _state["session_count"] = 0

            try:
                frame = frame_q.get(timeout=0.3)
            except queue.Empty:
                continue

            fc += 1
            try:
                results = model.track(
                    frame, conf=CONF, iou=IOU, imgsz=IMGSZ,
                    persist=True, tracker="bytetrack.yaml", verbose=False,
                )
            except Exception as ex:
                errors += 1
                if errors <= 3: print(f"[track] {ex}")
                try:
                    results = model.predict(frame, conf=CONF, iou=IOU,
                                            imgsz=IMGSZ, verbose=False)
                except Exception:
                    continue

            boxes = results[0].boxes
            out   = frame.copy()
            det_n = 0

            if boxes is not None and len(boxes.xyxy) > 0:
                track_ids = None
                if hasattr(boxes, "id") and boxes.id is not None:
                    try: track_ids = [int(boxes.id[i].item()) for i in range(len(boxes.id))]
                    except Exception: pass

                for i, xyxy in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    if not _keep_box(x1, y1, x2, y2, IMGSZ): continue
                    tid      = track_ids[i] if track_ids and i < len(track_ids) else -1
                    conf_val = float(boxes.conf[i])
                    if tid > 0: unique_ids.add(tid)
                    # Green box — same as test_detection.py
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 60), 2)
                    lbl = f"#{tid} {conf_val:.0%}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 60), -1)
                    cv2.putText(out, lbl, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    det_n += 1

            elapsed = max(time.time() - t0, 1e-6)
            fps_val = fc / elapsed
            hud = (f"Frame {fc}/{total_frames}  Det: {det_n}  "
                   f"Tracks: {len(unique_ids)}  {fps_val:.1f} FPS")
            cv2.putText(out, hud, (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 2)

            with _lock:
                _state["frame1"]        = out
                _state["fps"]           = round(fps_val, 1)
                _state["active_tracks"] = len(unique_ids)
                _state["session_count"] = len(unique_ids)
                _state["frame_count"]   = fc

        cap.release()
        with _lock:
            _state["running"] = False
            _state["frame1"]  = None


def _start_pipeline(s1, settings, db):
    with _lock:
        if _state.get("running"): return
        stop = threading.Event()
        _state.update({
            "stop_event": stop, "running": True, "error": None,
            "loading": True, "loading_msg": "Loading model…",
            "source1": str(s1) if s1 else None,
        })
        t = _PipelineWorker(s1, settings, db, stop)
        _state["thread"] = t; t.start()


def _stop_pipeline():
    with _lock:
        s = _state.get("stop_event")
        if s: s.set()
        _state["error"] = None


def _restart_pipeline(s1, settings, db):
    with _lock:
        s = _state.get("stop_event")
        if s: s.set()
        old_thread = _state.get("thread")
        _state["running"] = False
        _state["frame1"]  = None
        _state["error"]   = None
    if old_thread and old_thread.is_alive():
        old_thread.join(timeout=2.0)
    _start_pipeline(s1, settings, db)


def _bar_chart(data: dict, title: str, t: dict) -> go.Figure:
    labels = [GRADE_DISPLAY.get(k, k.replace("_", " ").title()) for k in data]
    values = list(data.values())
    colors = [t["grade_colors"].get(
        GRADE_DISPLAY.get(k, k.replace("_", " ").title()), t["primary"]
    ) for k in data]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, marker_line_width=0,
        text=values, textposition="outside",
        textfont=dict(size=11, color=t["plot_text"], family="Inter"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=t["plot_text"], family="Inter"), x=0),
        plot_bgcolor=t["plot_bg"], paper_bgcolor=t["plot_paper"],
        font=dict(color=t["plot_text"], family="Inter", size=11),
        xaxis=dict(gridcolor=t["plot_grid"], linecolor=t["border"],
                   tickfont=dict(size=10, color=t["plot_text"])),
        yaxis=dict(gridcolor=t["plot_grid"], linecolor=t["border"],
                   tickfont=dict(size=10, color=t["plot_text"])),
        margin=dict(t=36, b=10, l=10, r=10), height=270, showlegend=False,
    )
    return fig


def _pie_chart(data: dict, title: str, t: dict) -> go.Figure:
    labels = [GRADE_DISPLAY.get(k, k.replace("_", " ").title()) for k in data]
    values = list(data.values())
    colors = [t["grade_colors"].get(
        GRADE_DISPLAY.get(k, k.replace("_", " ").title()), t["primary"]
    ) for k in data]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, marker_colors=colors,
        hole=0.42, textinfo="percent+label",
        textfont=dict(size=10, color=t["plot_text"]),
        insidetextorientation="radial",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=t["plot_text"], family="Inter"), x=0),
        plot_bgcolor=t["plot_bg"], paper_bgcolor=t["plot_paper"],
        font=dict(color=t["plot_text"], family="Inter", size=10),
        legend=dict(font=dict(size=9, color=t["plot_text"]),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(t=36, b=10, l=10, r=10), height=270,
    )
    return fig


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OnionIQ — SIH26031",
    page_icon=":material/agriculture:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_defaults = {
    "theme": "light",
    "video_path": "", "source_type": "Video File",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

T = THEMES[st.session_state.theme]
st.markdown(_css(T), unsafe_allow_html=True)


@st.cache_resource
def get_db(): return OnionDatabase(DB_PATH)

db          = get_db()
s_global    = load_settings()
running     = _state.get("running", False)
_cp         = s_global.get("model_path", "")
_has_custom = bool(_cp) and Path(_cp).exists()

# Cache expensive DB queries — only refresh every 2 s, not on every 0.05 s rerun
_now = time.time()
_cache_stale = (
    "_db_cache_ts" not in st.session_state
    or _now - st.session_state._db_cache_ts > 2.0
)
if _cache_stale:
    st.session_state._db_cache_ts = _now
    st.session_state._db_summary  = db.get_batch_summary("DEMO")
    st.session_state._db_recent   = db.get_recent(n=200, batch_id="DEMO")




# ── Page header ────────────────────────────────────────────────────────────────
_model_label = Path(_cp).stem if _has_custom else "YOLO11n-seg"
st.markdown(f"""
<div style="background:{T['header_bg']};color:{T['header_text']};padding:12px 20px;
            border-radius:6px;margin-bottom:20px;
            display:flex;align-items:center;justify-content:space-between;
            border-left:4px solid {T['accent_em']};
            box-shadow:0px 4px 8px 0px {T['elev_2']},0px 1px 2px 0px {T['elev_1']}">
  <div>
    <div style="font-size:0.98rem;font-weight:700;font-family:Inter,sans-serif;
                letter-spacing:0.2px">
      OnionIQ — Onion Quality Grading System
    </div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.55);margin-top:3px;
                font-family:Inter,sans-serif">
      Ministry of Consumer Affairs, Food &amp; Public Distribution
      &nbsp;&middot;&nbsp; SIH26031
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px">
    <span style="font-size:0.68rem;color:rgba(255,255,255,0.42);font-family:Inter,sans-serif">
      {_model_label}
    </span>
    {_status_badge(running, T)}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Top strip: model info + theme toggle ──────────────────────────────────────
_strip_l, _strip_r = st.columns([8, 1])
_strip_l.markdown(
    f'<span style="font-size:0.72rem;color:{T["text_muted"]};font-family:Inter,sans-serif">'
    f'Model: <b>{Path(_cp).stem if _has_custom else "YOLO11n-seg (default)"}</b>'
    f' &nbsp;·&nbsp; FPS: {_state.get("fps", 0):.1f}'
    f' &nbsp;·&nbsp; v1.0.0-beta</span>',
    unsafe_allow_html=True,
)
_lbl = "Dark" if st.session_state.theme == "light" else "Light"
if _strip_r.button(_lbl, key="theme_btn"):
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_live, tab_analytics, tab_settings = st.tabs([
    "Live Detection", "Analytics & Reports", "Configuration",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    _fc = _state.get("frame_count", 0)
    _tf = _state.get("total_frames", 0)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Onions Counted", _state.get("active_tracks", 0))
    k2.metric("Frame Rate",     f"{_state.get('fps', 0):.1f} FPS")
    k3.metric("Frames",         f"{_fc}/{_tf}" if _tf > 0 else str(_fc))
    k4.metric("Status",         "Live" if running else "Stopped")

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    feed_col, ctrl_col = st.columns([2.4, 1], gap="large")

    with ctrl_col:
        _section("Input Source")
        st.radio("Type", ["Camera", "Video File"],
                 horizontal=True, label_visibility="collapsed", key="source_type")
        source_type = st.session_state.source_type
        if source_type == "Camera":
            _scan_col, _btn_col = st.columns([3, 1])
            if _btn_col.button("Scan", use_container_width=True, key="scan_btn"):
                _found = []
                with st.spinner("Scanning 0–4…"):
                    for _idx in range(5):
                        _c = cv2.VideoCapture(_idx, cv2.CAP_DSHOW)
                        if _c.isOpened():
                            _r, _ = _c.read()
                            if _r: _found.append(_idx)
                        _c.release()
                st.session_state["cam_options"] = _found if _found else list(range(5))
                if not _found: st.warning("No cameras detected — showing all indices.")
            _cam_opts = st.session_state.get("cam_options", list(range(5)))
            _cam_sel  = _scan_col.selectbox(
                "Select Camera",
                options=_cam_opts,
                format_func=lambda x: f"Camera {x}" + (" (built-in)" if x == 0 else ""),
                key="cam1_idx_sel",
            )
            source1 = str(_cam_sel)
        else:
            # Drag-and-drop upload (saves to project videos/ folder)
            uploaded = st.file_uploader(
                "Drop video here", type=["mp4", "avi", "mov", "mkv", "webm"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                _save_path = _PROJECT / "videos" / uploaded.name
                _PROJECT.joinpath("videos").mkdir(exist_ok=True)
                _save_path.write_bytes(uploaded.getbuffer())
                st.session_state.video_path = str(_save_path)
                st.success(f"Saved: {uploaded.name}")
            st.text_input("Or enter path", key="video_path",
                          placeholder=r"C:\Videos\belt.mp4")
            source1 = st.session_state.video_path if st.session_state.video_path else None
            if source1 and not Path(source1).exists():
                st.warning("File not found.")

        _s1 = str(source1).strip() if source1 else None
        if running and _state.get("source1") != _s1:
            st.warning("Source changed — click Restart to apply.")

        _section("Pipeline Control")
        b1, b2, b3 = st.columns(3)
        # Valid source = non-empty string that isn't literally "None"
        _source_ok = bool(_s1 and _s1 not in ("", "None"))
        with b1:
            if st.button("Start", disabled=(running or not _source_ok),
                         use_container_width=True, type="primary"):
                s = load_settings()
                s["_batch_id"] = "DEMO"
                _start_pipeline(source1, s, db); st.rerun()
        with b2:
            if st.button("Stop", disabled=not running, use_container_width=True):
                _stop_pipeline(); st.rerun()
        with b3:
            if st.button("Restart", disabled=not running, use_container_width=True,
                         help="Restart with current source settings"):
                s = load_settings()
                s["_batch_id"] = "DEMO"
                _restart_pipeline(source1, s, db); st.rerun()

        if st.button("Reset Count", use_container_width=True,
                     disabled=not running,
                     help="Zero the frame counter and unique track IDs mid-run"):
            with _lock:
                _state["reset_requested"] = True

        if _state.get("error"):
            st.error(f"Pipeline error: {_state['error']}")

        _section("Detection Parameters")
        conf    = st.slider("Confidence", 0.10, 0.95,
                            float(s_global.get("confidence_threshold", 0.50)), 0.05)
        iou_val = st.slider("NMS IoU",    0.10, 0.95,
                            float(s_global.get("iou_threshold", 0.45)), 0.05)
        _apply_lbl = "Save (restart to apply)" if running else "Apply"
        if st.button(_apply_lbl, use_container_width=True):
            save_settings({"confidence_threshold": conf, "iou_threshold": iou_val})
            st.toast("Parameters saved.")

    with feed_col:
        if _state.get("loading"):
            st.warning(f"⏳ {_state.get('loading_msg', 'Initialising pipeline…')} "
                       "This may take up to 30 s on first run.")
        elif _has_custom:
            _model_name = Path(_cp).stem
            st.success(f"Model: **{_model_name}** — ByteTrack active. HSV heuristic disabled.")
        else:
            st.info(
                "No custom model — using YOLO11n + HSV heuristic. "
                "Load a trained model in Configuration to enable grade classification."
            )

        # ── WebRTC path (smoothest — if streamlit-webrtc installed) ──────────
        if _WEBRTC_OK and source1 and not running:
            # WebRTC streams MP4 through aiortc.MediaPlayer → VideoProcessor
            # → browser at true 30 FPS without any MJPEG or Streamlit rerun.
            from ultralytics import YOLO as _YOLO

            _pt  = s_global.get("model_path", "")
            _eng = Path(_pt).with_suffix(".engine") if _pt else None
            _mdl_path = str(_eng) if (_eng and _eng.exists()) \
                        else (_pt if _pt and Path(_pt).exists() else "yolo11n-seg.pt")
            _wconf = float(s_global.get("confidence_threshold", 0.30))
            _wiou  = float(s_global.get("iou_threshold", 0.45))

            @st.cache_resource
            def _load_webrtc_model(path):
                return _YOLO(path)

            _wmodel = _load_webrtc_model(_mdl_path)

            class _OnionProcessor(VideoProcessorBase):
                def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.resize(img, (640, 640))
                    try:
                        results = _wmodel.track(
                            img, conf=_wconf, iou=_wiou, imgsz=640,
                            persist=True, tracker="bytetrack.yaml", verbose=False,
                        )
                    except Exception:
                        results = _wmodel.predict(img, conf=_wconf, iou=_wiou,
                                                  imgsz=640, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes.xyxy) > 0:
                        tids = None
                        if hasattr(boxes, "id") and boxes.id is not None:
                            try: tids = [int(boxes.id[i].item()) for i in range(len(boxes.id))]
                            except Exception: pass
                        for i, xyxy in enumerate(boxes.xyxy):
                            x1,y1,x2,y2 = [int(v) for v in xyxy]
                            if not _keep_box(x1, y1, x2, y2, 640): continue
                            tid = tids[i] if tids and i<len(tids) else -1
                            cf  = float(boxes.conf[i])
                            cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,60),2)
                            lbl = f"#{tid} {cf:.0%}"
                            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                            cv2.rectangle(img,(x1,y1-th-8),(x1+tw+4,y1),(0,200,60),-1)
                            cv2.putText(img,lbl,(x1+2,y1-4),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            try:
                from aiortc.contrib.media import MediaPlayer as _MediaPlayer
                _src1 = str(source1)

                def _player_factory():
                    return _MediaPlayer(_src1)

                webrtc_streamer(
                    key=f"onion-{Path(_src1).name}",
                    mode=WebRtcMode.RECVONLY,
                    player_factory=_player_factory,
                    video_processor_factory=_OnionProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            except Exception as _we:
                st.warning(f"WebRTC unavailable ({_we}) — using MJPEG fallback.")
                _WEBRTC_OK_LOCAL = False
            else:
                _WEBRTC_OK_LOCAL = True
        else:
            _WEBRTC_OK_LOCAL = False

        # ── MJPEG fallback (pipeline running or webrtc not available) ─────────
        if not _WEBRTC_OK or _WEBRTC_OK_LOCAL is False or running:
            if running or _state.get("frame1") is not None:
                # Label is static — dynamic FPS/frame info is drawn by cv2.putText
                # on the frame itself. Changing the HTML string would reload the
                # iframe on every Streamlit rerun, disconnecting the MJPEG stream.
                st.components.v1.html(
                    _video_html(_MJPEG_PORT, 1, "Camera Feed", h=500),
                    height=514,
                )
            else:
                st.markdown(f"""
                <div style="background:{T['card_bg']};border:1px dashed {T['border']};
                            border-radius:6px;padding:60px 40px;text-align:center;
                            box-shadow:0px 1px 2px 0px {T['elev_1']}">
                  <div style="font-weight:600;color:{T['text']};margin-bottom:6px;
                              font-family:Inter,sans-serif">Camera feed not active</div>
                  <div style="font-size:0.82rem;color:{T['text_muted']};font-family:Inter,sans-serif">
                    Select a video file or camera index, then click
                    <strong>Start</strong>.
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    summary  = st.session_state.get("_db_summary", {"total": 0, "counts": {}, "defects": {}})
    total_s  = summary["total"]
    counts   = summary.get("counts", {})
    defects  = summary.get("defects", {})
    rows     = st.session_state.get("_db_recent", [])
    dias     = [r["estimated_diameter_mm"] for r in rows if r.get("estimated_diameter_mm")]
    # Use summary counts (full DB) not rows slice for percentages
    _a_cnt  = counts.get("grade_a", 0)
    _rj_cnt = counts.get("reject", 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Graded", total_s)
    k2.metric("Grade A",      f"{_a_cnt/total_s*100:.1f}%"  if total_s else "—")
    k3.metric("Reject Rate",  f"{_rj_cnt/total_s*100:.1f}%" if total_s else "—")
    k4.metric("Avg Diameter", f"{sum(dias)/len(dias):.1f} mm" if dias else "—")

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    if total_s == 0:
        _empty_state(
            "No records yet",
            "Start the pipeline and run onions through the belt to generate data.",
            T,
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            if counts:
                st.plotly_chart(_bar_chart(counts, "Grade Distribution", T),
                                use_container_width=True, config={"displayModeBar": False})
        with c2:
            src = defects if defects else counts
            if src:
                st.plotly_chart(
                    _pie_chart(src, "Defect Breakdown" if defects else "Grade Share", T),
                    use_container_width=True, config={"displayModeBar": False},
                )

        _section("Recent Detections")
        if rows:
            df = pd.DataFrame(rows[:60])
            dcols = [c for c in [
                "global_id", "final_grade", "defect_type",
                "estimated_diameter_mm", "cam1_confidence", "timestamp",
            ] if c in df.columns]
            df_disp = df[dcols].copy()
            df_disp.columns = [c.replace("_", " ").title() for c in dcols]
            st.dataframe(df_disp, use_container_width=True, height=280)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Configuration
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    s = load_settings()

    _section("Model")
    mc1, mc2 = st.columns(2)
    model_path = mc1.text_input(
        "Custom Model Path (.pt)",
        value=s.get("model_path", ""),
        help="Absolute path to your trained YOLO11 weights (e.g. models/best.pt). "
             "Leave blank to use YOLO11n-seg (auto-downloaded, detection-only).",
    )
    data_yaml = mc2.text_input(
        "Dataset YAML (data.yaml)",
        value=s.get("data_yaml", ""),
        placeholder="dataset/data.yaml",
        help="Path to the data.yaml from your training dataset. "
             "Provides class names (Grade A / B / Reject …) to the pipeline. "
             "Export format: YOLOv11 or YOLOv8 from Roboflow.",
    )
    model_exists = Path(model_path).exists() if model_path else False
    yaml_exists  = Path(data_yaml).exists()  if data_yaml  else False
    if model_path:
        if model_exists: st.success("Model file found — grading activates on next Start.")
        else:            st.warning("File not found — YOLO11n-seg (detection-only) will be used.")
    if data_yaml:
        if yaml_exists:
            try:
                import yaml as _yaml
                _dy = _yaml.safe_load(open(data_yaml))
                _nc   = _dy.get("nc", "?")
                _names = list(_dy.get("names", {}).values()) \
                         if isinstance(_dy.get("names"), dict) \
                         else _dy.get("names", [])
                st.info(f"YAML OK — {_nc} classes: {', '.join(str(n) for n in _names[:8])}"
                        + (" …" if len(_names) > 8 else ""))
            except Exception as ex:
                st.warning(f"YAML parse error: {ex}")
        else:
            st.warning("YAML file not found.")

    if not Path("yolo11n-seg.pt").exists() and \
       not (Path(_PROJECT / "models" / "yolo11n-seg.pt")).exists():
        if st.button("Download YOLO11n-seg weights (~6 MB)", type="primary"):
            with st.spinner("Downloading…"):
                try:
                    from ultralytics import YOLO
                    YOLO("yolo11n-seg.pt")
                    st.success("Downloaded successfully.")
                except Exception as ex:
                    st.error(f"Download failed: {ex}")

    _section("Belt & Ejector")
    col_e, col_f = st.columns(2)
    belt_spd = col_e.slider("Belt Speed (m/s)",     0.10, 1.0,
                             float(s.get("belt_speed_ms",       0.3)),  0.05)
    eject_d  = col_f.slider("Ejector Distance (m)", 0.10, 2.0,
                             float(s.get("ejector_distance_m", 0.45)), 0.05)
    st.info(f"Ejector delay: {eject_d / max(belt_spd, 0.01):.2f} s  ({eject_d} m / {belt_spd} m/s)")
    simulated = st.checkbox(
        "Simulated ejector (logging only — uncheck for real serial/GPIO)",
        value=bool(s.get("simulated_ejector", True)),
    )

    _section("Size Estimation")
    cal = st.slider("Calibration Factor (mm/pixel)", 0.001, 0.10,
                    float(s.get("calibration_factor", 0.014)), 0.001, format="%.3f")
    st.caption("Run calibration.py with a reference object of known diameter to compute this value.")

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    sa, sb = st.columns(2)
    if sa.button("Save Configuration", use_container_width=True, type="primary"):
        save_settings({
            "model_path":          model_path,
            "data_yaml":           data_yaml,
            "belt_speed_ms":       belt_spd,
            "ejector_distance_m":  eject_d,
            "simulated_ejector":   simulated,
            "calibration_factor":  cal,
        })
        st.success("Configuration saved. Restart pipeline to apply changes.")
    if sb.button("Reset to Defaults", use_container_width=True):
        reset_settings()
        st.info("Settings reset to defaults.")


# ── Auto-refresh ───────────────────────────────────────────────────────────────
time.sleep(0.5 if _state.get("running") else 2.0)
st.rerun()

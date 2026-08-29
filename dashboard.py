"""
OnionIQ — AI-powered Onion Quality Grading System (SIH26031)
Dashboard: Streamlit + streamlit-elements (Material UI + Nivo charts)

Run:  streamlit run dashboard.py
"""
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_elements import dashboard, elements, mui, nivo

_PROJECT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT))

from config import load_settings, save_settings, reset_settings, DB_PATH
from database import OnionDatabase
from grader import GradeDecisionEngine
from matcher import CrossCameraMatcher
from model_wrapper import OnionModel

# ─────────────────────────────────────────────────────────────────────────────
# Brand & colour constants
# ─────────────────────────────────────────────────────────────────────────────
PRIMARY      = "#FF8C42"   # onion amber
PRIMARY_DARK = "#E65100"
CARD_BG      = "#1A1F2E"
HEADER_BG    = "#0D1117"

GRADE_BGR = {
    "grade_a": (80, 200, 80),   "grade_b": (40, 160, 255),
    "grade_c": (40, 40, 220),   "reject":  (130, 50, 200),
    "sprouting":(20,180,200),   "rot":     (30, 30, 180),
    "thrips_damage":(40,80,220),"neck_rot":(20, 20, 180),
    "sunscald":(20,200,220),    "bruising":(60,100,200),
    "onion":   (200,160,60),    "detected":(200,160,60),
}
GRADE_HEX = {
    "grade_a":"#50c850","grade_b":"#ffa028","grade_c":"#dc2828",
    "reject":"#8832c8","sprouting":"#c8b420","rot":"#b41414",
    "onion":"#3c8cd8","detected":"#3c8cd8",
}
NIVO_THEME = {
    "background": "#0E1117",
    "textColor":  "#FAFAFA",
    "fontSize":   12,
    "axis": {
        "domain": {"line": {"stroke": "#555"}},
        "ticks":  {"line": {"stroke": "#555"}, "text": {"fill": "#AAAAAA"}},
        "legend": {"text": {"fill": "#FAFAFA"}},
    },
    "grid":   {"line": {"stroke": "#333"}},
    "legends":{"text": {"fill": "#AAAAAA"}},
    "tooltip":{"container": {"background": "#1A1F2E", "color": "#FAFAFA"}},
}

def _bbox_color(cls: str) -> tuple:
    return GRADE_BGR.get(cls, (160, 160, 160))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level pipeline state (survives Streamlit reruns within same process)
# ─────────────────────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    "running": False,
    "frame1": None, "frame2": None,
    "fps": 0.0, "active_tracks": 0, "session_count": 0,
    "stop_event": None, "thread": None, "error": None,
    "source1": None, "source2": None,
    "cam1_open": False, "cam2_open": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────────────────
def annotate_frame(frame, events, settings, cam_id, show_overlap=True):
    out = frame.copy()
    h, w = out.shape[:2]
    if show_overlap:
        x1 = int(settings.get(f"overlap_start_cam{cam_id}_x", 900 if cam_id==1 else 0))
        x2 = int(settings.get(f"overlap_end_cam{cam_id}_x",  1280 if cam_id==1 else 380))
        ov = out.copy()
        cv2.rectangle(ov, (x1, 0), (x2, h), (0, 200, 200), -1)
        cv2.addWeighted(ov, 0.10, out, 0.90, 0, out)
        cv2.line(out, (x1, 0), (x1, h), (0, 200, 200), 2)
        cv2.line(out, (x2, 0), (x2, h), (0, 200, 200), 2)
        cv2.putText(out, "OVERLAP", (x1+4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,200), 1)
    for e in events:
        if e.cam_id != cam_id:
            continue
        x1, y1, x2, y2 = [int(v) for v in e.bbox]
        col = _bbox_color(e.class_name)
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        tid = f"#{e.track_id}"
        (tw,th),_ = cv2.getTextSize(tid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+6, y1), col, -1)
        cv2.putText(out, tid, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(out, f"{e.class_name} {e.confidence:.0%}", (x1, y2+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if e.global_id:
            cv2.putText(out, f"G{e.global_id}", (x1, y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)
    cv2.putText(out, f"CAM {cam_id}", (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker
# ─────────────────────────────────────────────────────────────────────────────
class _PipelineWorker(threading.Thread):
    def __init__(self, source1, source2, settings, db, stop_event):
        super().__init__(daemon=True, name="PipelineWorker")
        self._source1  = source1
        self._source2  = source2
        self._settings = settings
        self._db       = db
        self._stop     = stop_event
        self._batch_id = settings.get("_batch_id", "DEMO")
        self._farmer   = settings.get("_farmer", "")

    def run(self):
        try:
            self._run()
        except Exception as e:
            with _lock:
                _state["error"]   = str(e)
                _state["running"] = False

    def _open(self, source):
        if source is None:
            return None
        try:
            idx = int(source)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(str(source))
        return cap if cap.isOpened() else None

    def _run(self):
        from tracker import OnionTracker

        model    = OnionModel(self._settings)
        matcher  = CrossCameraMatcher(self._settings)
        engine   = GradeDecisionEngine(self._settings)
        tracker1 = OnionTracker(cam_id=1, settings=self._settings)
        tracker2 = OnionTracker(cam_id=2, settings=self._settings)

        cap1 = self._open(self._source1)
        cap2 = self._open(self._source2) if self._source2 is not None else None
        with _lock:
            _state["cam1_open"] = cap1 is not None
            _state["cam2_open"] = cap2 is not None

        show_overlap = self._settings.get("show_overlap_preview", True)
        WINDOW       = self._settings.get("match_time_window_s", 2.0)
        frame_count  = 0
        t_start      = time.time()
        pending: dict = {}

        def _blank():
            f = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(f, "No source", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,80,80), 2)
            return f

        while not self._stop.is_set():
            # Read frames
            if cap1 and cap1.isOpened():
                ret1, frame1 = cap1.read()
                if not ret1:
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret1, frame1 = cap1.read()
                if not ret1:
                    frame1 = _blank()
            else:
                frame1 = _blank()

            if cap2 and cap2.isOpened():
                ret2, frame2 = cap2.read()
                if not ret2:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret2, frame2 = cap2.read()
                if not ret2:
                    frame2 = _blank()
            else:
                frame2 = None

            # Inference
            all_events = []
            evts1 = model.predict(frame1, cam_id=1)
            evts1 = tracker1.update(evts1, frame1)
            all_events.extend(evts1)
            if frame2 is not None:
                evts2 = model.predict(frame2, cam_id=2)
                evts2 = tracker2.update(evts2, frame2)
                all_events.extend(evts2)

            # Cross-camera matching + grading
            for e in all_events:
                e = matcher.process(e)
                if e.global_id is None:
                    continue
                pending.setdefault(e.global_id, []).append(e)

            now = time.time()
            for gid in list(pending):
                grp    = pending[gid]
                cam_ids = {ev.cam_id for ev in grp}
                oldest  = min(ev.timestamp for ev in grp)
                if len(cam_ids) >= 2 or (now - oldest) > WINDOW:
                    g  = pending.pop(gid)
                    c1 = next((ev for ev in g if ev.cam_id==1), None)
                    c2 = next((ev for ev in g if ev.cam_id==2), None)
                    dec = engine.decide(gid, c1, c2, self._batch_id,
                                        self._farmer, model.grading_active)
                    self._db.insert_grade(dec.to_db_row())

            # Annotate
            ann1 = annotate_frame(frame1, all_events, self._settings, 1, show_overlap)
            ann2 = annotate_frame(frame2, all_events, self._settings, 2, show_overlap) \
                   if frame2 is not None else None

            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            session_count = self._db.get_batch_summary(self._batch_id).get("total", 0)

            with _lock:
                _state["frame1"]         = ann1
                _state["frame2"]         = ann2
                _state["fps"]            = round(fps, 1)
                _state["active_tracks"]  = len(pending)
                _state["session_count"]  = session_count

            time.sleep(0.033)

        if cap1: cap1.release()
        if cap2: cap2.release()
        with _lock:
            _state["running"] = False
            _state["frame1"]  = None
            _state["frame2"]  = None


def _start_pipeline(source1, source2, settings, db):
    with _lock:
        if _state.get("running"):
            return
        stop = threading.Event()
        _state.update({
            "stop_event": stop, "running": True, "error": None,
            "source1": str(source1) if source1 is not None else None,
            "source2": str(source2) if source2 is not None else None,
        })
        t = _PipelineWorker(source1, source2, settings, db, stop)
        _state["thread"] = t
        t.start()


def _stop_pipeline():
    with _lock:
        s = _state.get("stop_event")
        if s:
            s.set()


def _restart_pipeline(source1, source2, settings, db):
    with _lock:
        s = _state.get("stop_event")
        if s:
            s.set()
        _state["running"] = False
        _state["frame1"]  = None
        _state["frame2"]  = None
    _start_pipeline(source1, source2, settings, db)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OnionIQ",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session-state defaults
for k, v in [
    ("batch_id", "DEMO001"), ("farmer_name", ""),
    ("source_type", "Camera"), ("cam1_idx", "0"),
    ("cam2_enabled", False), ("cam2_source", "1"),
    ("video_path", ""), ("refresh_ms", 200),
]:
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource
def get_db():
    return OnionDatabase(DB_PATH)

db      = get_db()
s_global = load_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:8px 0'>
        <span style='font-size:2rem'>🧅</span><br>
        <span style='font-size:1.4rem;font-weight:700;color:{PRIMARY}'>OnionIQ</span><br>
        <span style='font-size:0.7rem;color:#888'>SIH26031 · Ministry of Consumer Affairs</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.session_state.batch_id    = st.text_input("Batch ID",         value=st.session_state.batch_id)
    st.session_state.farmer_name = st.text_input("Farmer / Supplier", value=st.session_state.farmer_name)
    st.divider()
    pending_count = db.count_pending_sync()
    col_a, col_b = st.columns(2)
    col_a.metric("Pending Sync", pending_count)
    col_b.metric("Detected", _state.get("session_count", 0))
    st.divider()
    if st.button("📄 Generate Certificate", use_container_width=True):
        import certificate
        out = certificate.generate(st.session_state.batch_id, db, str(_PROJECT))
        if out and os.path.exists(out):
            with open(out, "rb") as f:
                st.download_button("⬇ Download PDF", f.read(),
                                   file_name=Path(out).name, mime="application/pdf")
        else:
            st.error("No data yet or ReportLab not installed.")
    st.divider()
    _cp = s_global.get("model_path", "")
    _has_custom = bool(_cp) and Path(_cp).exists()
    st.caption(f"Model: {'🟢 Custom' if _has_custom else '🔵 YOLO11n-seg (pre-trained)'}")
    st.caption(f"Mode:  {'Grading Active' if _has_custom else 'Detection Only'}")
    st.caption(f"Inference: CPU  ·  FPS {_state.get('fps', 0):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Header bar (Material UI)
# ─────────────────────────────────────────────────────────────────────────────
running = _state.get("running", False)
with elements("header"):
    with mui.Paper(elevation=2, sx={
        "p": "10px 20px", "mb": 1, "display": "flex",
        "alignItems": "center", "justifyContent": "space-between",
        "bgcolor": CARD_BG, "borderRadius": 2,
    }):
        with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 2}):
            mui.Typography("🧅 OnionIQ", variant="h6",
                           sx={"fontWeight": 700, "color": PRIMARY})
            mui.Typography("AI Onion Quality Grading · SIH26031",
                           variant="caption", sx={"color": "#888"})
        with mui.Box(sx={"display": "flex", "gap": 1}):
            mui.Chip(
                label="● RUNNING" if running else "○ STOPPED",
                color="success" if running else "default",
                size="small",
                sx={"fontWeight": 600},
            )
            mui.Chip(
                label=f"🟢 {Path(_cp).stem}" if _has_custom else "🔵 YOLO11n-seg",
                color="warning" if _has_custom else "info",
                size="small",
            )
            mui.Chip(
                label=f"Batch: {st.session_state.batch_id}",
                size="small", variant="outlined",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_demo, tab_analytics, tab_disputes, tab_settings = st.tabs([
    "📹 Live Demo", "📊 Analytics", "🔍 Disputes", "⚙️ Settings",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Demo
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:

    # ── Live metric cards (Material UI) ─────────────────────────────────────
    with elements("live_metrics"):
        with mui.Grid(container=True, spacing=2, sx={"mb": 2}):
            for label, value, icon_name, color in [
                ("Detected",      _state.get("session_count", 0), "Radar",         PRIMARY),
                ("FPS",           f"{_state.get('fps', 0):.1f}",  "Speed",         "#4CAF50"),
                ("Active Tracks", _state.get("active_tracks", 0), "TrackChanges",  "#2196F3"),
                ("Status",        "Running" if running else "Stopped",
                                                                    "FiberManualRecord",
                                  "#4CAF50" if running else "#888"),
            ]:
                with mui.Grid(item=True, xs=3):
                    with mui.Card(elevation=3, sx={
                        "bgcolor": CARD_BG, "borderRadius": 2,
                        "borderLeft": f"4px solid {color}",
                    }):
                        with mui.CardContent(sx={"pb": "12px !important", "pt": 1.5}):
                            with mui.Box(sx={"display":"flex","alignItems":"center","gap":1}):
                                getattr(mui.icon, icon_name)(sx={"color": color, "fontSize": 20})
                                mui.Typography(label, variant="caption",
                                               sx={"color": "#888", "textTransform": "uppercase",
                                                   "letterSpacing": 1})
                            mui.Typography(str(value), variant="h4",
                                           sx={"fontWeight": 700, "color": "#fff", "mt": 0.5})

    # ── Model mode banner ────────────────────────────────────────────────────
    if _has_custom:
        st.success(f"🟢 Custom model loaded — grading active ({Path(_cp).name})")
    else:
        st.info("🔵 **YOLO11n-seg** running in detection-only mode — "
                "all detections labelled **onion**. "
                "Add your trained model in ⚙️ Settings to enable grading.")

    # ── Feed (left) | Controls (right) ──────────────────────────────────────
    feed_col, ctrl_col = st.columns([2.2, 1], gap="medium")

    # RIGHT — Controls
    with ctrl_col:
        st.subheader("Source")
        source_type = st.radio("Input type", ["Camera", "Video File"],
                               horizontal=True, label_visibility="collapsed")

        if source_type == "Camera":
            if st.button("📷 Scan cameras", key="scan_btn",
                         help="Tests indices 0–4"):
                found = []
                for idx in range(5):
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            found.append(idx)
                    cap.release()
                st.session_state["cam_scan"] = found
            scan = st.session_state.get("cam_scan")
            if scan is not None:
                if scan:
                    st.success(f"Available: {scan}")
                else:
                    st.warning("No cameras found. Try a video file.")
            cam1_val = st.text_input("Camera 1 index", value=st.session_state.cam1_idx,
                                     help="0 = built-in webcam, 1 = first USB cam")
            st.session_state.cam1_idx = cam1_val
            source1 = cam1_val
        else:
            vpath = st.text_input("Video file path", value=st.session_state.video_path,
                                  placeholder=r"C:\Videos\onion_demo.mp4")
            st.session_state.video_path = vpath
            source1 = vpath if vpath else None
            if source1 and not Path(source1).exists():
                st.warning("File not found — check the path.")

        cam2_enabled = st.checkbox("Enable Camera 2", value=st.session_state.cam2_enabled)
        st.session_state.cam2_enabled = cam2_enabled
        if cam2_enabled:
            cam2_val = st.text_input("Camera 2 index" if source_type == "Camera" else "Video 2 path",
                                     value=st.session_state.cam2_source)
            st.session_state.cam2_source = cam2_val
            source2 = cam2_val if cam2_val else None
        else:
            source2 = None

        # Source-changed warning
        _s1 = str(source1) if source1 is not None else None
        _s2 = str(source2) if source2 is not None else None
        source_changed = running and (
            _state.get("source1") != _s1 or _state.get("source2") != _s2
        )
        if source_changed:
            st.warning("Source changed — click **Restart** to switch.")

        st.divider()

        # Start / Stop / Restart
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶ Start", disabled=running, use_container_width=True, type="primary"):
                s = load_settings()
                s["_batch_id"] = st.session_state.batch_id
                s["_farmer"]   = st.session_state.farmer_name
                _start_pipeline(source1, source2, s, db)
                st.rerun()
        with b2:
            if st.button("⏹ Stop", disabled=not running, use_container_width=True):
                _stop_pipeline()
                st.rerun()
        with b3:
            if st.button("🔄", disabled=not running, use_container_width=True,
                         help="Restart with current source"):
                s = load_settings()
                s["_batch_id"] = st.session_state.batch_id
                s["_farmer"]   = st.session_state.farmer_name
                _restart_pipeline(source1, source2, s, db)
                st.rerun()

        if _state.get("error"):
            st.error(f"Error: {_state['error']}")

        st.divider()
        st.subheader("Detection Controls")
        s_live = load_settings()
        conf    = st.slider("Confidence",    0.10, 0.95,
                            float(s_live.get("confidence_threshold", 0.35)), 0.05)
        iou_val = st.slider("IoU threshold", 0.10, 0.95,
                            float(s_live.get("iou_threshold", 0.45)), 0.05)
        show_ov = st.checkbox("Show overlap zone",
                              value=bool(s_live.get("show_overlap_preview", True)))
        if st.button("Apply", use_container_width=True):
            save_settings({
                "confidence_threshold": conf,
                "iou_threshold": iou_val,
                "show_overlap_preview": show_ov,
            })
            st.toast("Applied — takes effect on next frame.")

    # LEFT — Video Feed
    with feed_col:
        frame1_ph = st.empty()
        frame2_ph = st.empty()
        with _lock:
            f1 = _state.get("frame1")
            f2 = _state.get("frame2")

        if f1 is not None:
            frame1_ph.image(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB),
                            caption="Camera 1 ✅ Live" if _state.get("cam1_open")
                            else "Camera 1 ⚠️ No signal",
                            use_container_width=True)
        else:
            frame1_ph.info("📷 Camera 1 — press **▶ Start** to begin.\n\n"
                           "Not sure which index? Click **📷 Scan cameras** first.")

        if f2 is not None:
            frame2_ph.image(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB),
                            caption="Camera 2 ✅ Live" if _state.get("cam2_open")
                            else "Camera 2 ⚠️ No signal",
                            use_container_width=True)
        elif cam2_enabled:
            frame2_ph.info("Camera 2 — waiting for pipeline to start.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    batch_id = st.session_state.batch_id
    summary  = db.get_batch_summary(batch_id)
    total_s  = summary["total"]
    counts   = summary.get("counts", {})
    defects  = summary.get("defects", {})
    rows     = db.get_recent(n=200, batch_id=batch_id)

    reject_n = sum(1 for r in rows if r.get("final_grade") == "reject")
    a_n      = sum(1 for r in rows if r.get("final_grade") == "grade_a")
    dias     = [r["estimated_diameter_mm"] for r in rows if r.get("estimated_diameter_mm")]

    # Summary metric cards (Material UI)
    with elements("analytics_metrics"):
        with mui.Grid(container=True, spacing=2, sx={"mb": 2}):
            for label, value, color in [
                ("Total Graded",  total_s, PRIMARY),
                ("Grade A",
                 f"{a_n/total_s*100:.1f}%" if total_s else "—", "#4CAF50"),
                ("Reject Rate",
                 f"{reject_n/total_s*100:.1f}%" if total_s else "—", "#F44336"),
                ("Avg Diameter",
                 f"{sum(dias)/len(dias):.1f} mm" if dias else "—", "#2196F3"),
            ]:
                with mui.Grid(item=True, xs=3):
                    with mui.Card(elevation=3, sx={
                        "bgcolor": CARD_BG, "borderRadius": 2,
                        "borderTop": f"3px solid {color}",
                    }):
                        with mui.CardContent(sx={"pb": "12px !important"}):
                            mui.Typography(label, variant="caption",
                                           sx={"color": "#888", "textTransform": "uppercase"})
                            mui.Typography(str(value), variant="h5",
                                           sx={"fontWeight": 700, "color": "#fff"})

    if total_s == 0:
        st.info("No data yet for this batch. Start the pipeline and run onions through.")
    else:
        # Nivo charts inside a draggable dashboard
        nivo_layout = [
            dashboard.Item("grade_chart",  0, 0, 6, 5),
            dashboard.Item("defect_chart", 6, 0, 6, 5),
        ]
        with elements("analytics_charts"):
            with dashboard.Grid(nivo_layout, draggableHandle=".drag-handle"):

                # Grade distribution bar chart
                with mui.Card(key="grade_chart", elevation=3,
                              sx={"bgcolor": CARD_BG, "borderRadius": 2, "height": "100%"}):
                    with mui.CardContent(sx={"height": "100%"}):
                        with mui.Box(className="drag-handle", sx={
                            "display": "flex", "alignItems": "center",
                            "cursor": "move", "mb": 1,
                        }):
                            mui.icon.DragIndicator(sx={"color": "#555", "mr": 1})
                            mui.Typography("Grade Distribution", variant="subtitle1",
                                           sx={"fontWeight": 600, "color": "#fff"})

                        bar_data = [
                            {
                                "Grade": k.replace("_", " ").title(),
                                "Count": v,
                                "color": GRADE_HEX.get(k, "#888"),
                            }
                            for k, v in counts.items() if v > 0
                        ]
                        if bar_data:
                            nivo.Bar(
                                data=bar_data,
                                keys=["Count"],
                                indexBy="Grade",
                                margin={"top": 10, "right": 10, "bottom": 70, "left": 50},
                                padding=0.35,
                                colors={"datum": "data.color"},
                                axisBottom={
                                    "tickRotation": -35,
                                    "legend": "Grade",
                                    "legendOffset": 60,
                                    "legendPosition": "middle",
                                },
                                axisLeft={
                                    "legend": "Count",
                                    "legendOffset": -40,
                                    "legendPosition": "middle",
                                },
                                enableLabel=True,
                                labelSkipHeight=8,
                                theme=NIVO_THEME,
                                animate=True,
                            )

                # Defect breakdown pie chart
                with mui.Card(key="defect_chart", elevation=3,
                              sx={"bgcolor": CARD_BG, "borderRadius": 2, "height": "100%"}):
                    with mui.CardContent(sx={"height": "100%"}):
                        with mui.Box(className="drag-handle", sx={
                            "display": "flex", "alignItems": "center",
                            "cursor": "move", "mb": 1,
                        }):
                            mui.icon.DragIndicator(sx={"color": "#555", "mr": 1})
                            mui.Typography(
                                "Defect Breakdown" if defects else "Grade Breakdown",
                                variant="subtitle1", sx={"fontWeight": 600, "color": "#fff"},
                            )

                        pie_data_src = defects if defects else counts
                        pie_data = [
                            {"id": k.replace("_", " ").title(),
                             "label": k.replace("_", " ").title(),
                             "value": v,
                             "color": GRADE_HEX.get(k, "#888")}
                            for k, v in pie_data_src.items() if v > 0
                        ]
                        if pie_data:
                            nivo.Pie(
                                data=pie_data,
                                margin={"top": 10, "right": 80, "bottom": 50, "left": 80},
                                innerRadius=0.5,
                                padAngle=1.5,
                                cornerRadius=4,
                                colors={"datum": "data.color"},
                                enableArcLinkLabels=True,
                                arcLinkLabelsColor={"from": "color"},
                                arcLabelsSkipAngle=10,
                                theme=NIVO_THEME,
                                legends=[{
                                    "anchor": "bottom",
                                    "direction": "row",
                                    "translateY": 45,
                                    "itemWidth": 90,
                                    "itemHeight": 18,
                                    "symbolSize": 12,
                                    "symbolShape": "circle",
                                }],
                            )

        st.divider()
        st.subheader("Recent Detections")
        if rows:
            df = pd.DataFrame(rows[:50])
            dcols = [c for c in [
                "global_id", "final_grade", "defect_type",
                "estimated_diameter_mm", "cam1_confidence",
                "cam2_confidence", "timestamp",
            ] if c in df.columns]
            df_disp = df[dcols].copy()
            df_disp.columns = [c.replace("_", " ").title() for c in dcols]
            st.dataframe(df_disp, use_container_width=True, height=300)

        if st.session_state.farmer_name:
            st.divider()
            st.subheader(f"Farmer Profile — {st.session_state.farmer_name}")
            fp = db.get_farmer_fingerprint(st.session_state.farmer_name)
            if fp["total"] > 0:
                with elements("farmer_cards"):
                    with mui.Grid(container=True, spacing=2):
                        for lbl, val, color in [
                            ("All-time Onions", fp["total"],            PRIMARY),
                            ("Sessions",        fp["sessions"],         "#2196F3"),
                            ("Reject Rate",     f"{fp['reject_rate_pct']}%", "#F44336"),
                            ("Grade A Rate",    f"{fp['grade_a_rate_pct']}%","#4CAF50"),
                        ]:
                            with mui.Grid(item=True, xs=3):
                                with mui.Card(elevation=2, sx={
                                    "bgcolor": CARD_BG, "borderRadius": 2,
                                    "borderLeft": f"4px solid {color}",
                                }):
                                    with mui.CardContent(sx={"pb": "12px !important"}):
                                        mui.Typography(lbl, variant="caption", sx={"color":"#888"})
                                        mui.Typography(str(val), variant="h5",
                                                       sx={"fontWeight":700,"color":"#fff"})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Disputes
# ═══════════════════════════════════════════════════════════════════════════════
with tab_disputes:
    st.subheader("Camera Disagreements")
    disputed = db.get_disputed(batch_id=st.session_state.batch_id)
    if not disputed:
        st.info("No disputed onions (cam1 ≠ cam2 grade) in this batch.")
    else:
        st.caption(f"{len(disputed)} disputed onion(s).")
        ids = [r["global_id"] for r in disputed]
        sel = st.selectbox("Select global_id", ids)
        row = next((r for r in disputed if r["global_id"] == sel), None)
        if row:
            with elements("dispute_cards"):
                with mui.Grid(container=True, spacing=2, sx={"mb": 2}):
                    for lbl, val, color in [
                        ("Camera 1",  (row.get("cam1_grade") or "—").upper(), "#2196F3"),
                        ("Camera 2",  (row.get("cam2_grade") or "—").upper(), "#FF9800"),
                        ("Final",     (row.get("final_grade") or "—").upper(), "#4CAF50"),
                    ]:
                        with mui.Grid(item=True, xs=4):
                            with mui.Card(elevation=3, sx={
                                "bgcolor": CARD_BG, "borderRadius": 2,
                                "borderTop": f"3px solid {color}",
                            }):
                                with mui.CardContent():
                                    mui.Typography(lbl, variant="caption", sx={"color":"#888"})
                                    mui.Typography(val, variant="h5",
                                                   sx={"fontWeight":700,"color":"#fff"})
            img_cols = st.columns(3)
            for col, key, label in zip(
                img_cols,
                ["cam1_path", "cam2_path", "overlay_path"],
                ["Camera 1 Frame", "Camera 2 Frame", "Grad-CAM Overlay"],
            ):
                p = row.get(key)
                if p and os.path.exists(p):
                    img = cv2.imread(p)
                    col.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                              caption=label, use_container_width=True)
                else:
                    col.caption(f"{label}: not yet available")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Settings
# ═══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    s = load_settings()

    st.subheader("Model")
    model_path  = st.text_input("Model path (.pt)", value=s.get("model_path", ""),
                                help="Path to your trained weights. Classes auto-detected.")
    model_exists = Path(model_path).exists() if model_path else False
    st.caption("✅ File found — custom model will load on next Start"
               if model_exists else "⚠️ Not found — YOLO11n-seg (detection-only) active")

    _default_pt = str(_PROJECT / "models" / "yolo11n-seg.pt")
    yolo_exists = Path(_default_pt).exists() or Path("yolo11n-seg.pt").exists()
    if not yolo_exists:
        if st.button("⬇ Download yolo11n-seg.pt (~6 MB)"):
            with st.spinner("Downloading…"):
                try:
                    from ultralytics import YOLO
                    (_PROJECT / "models").mkdir(exist_ok=True)
                    YOLO("yolo11n-seg.pt")
                    st.success("Downloaded. It will be used automatically on next Start.")
                except Exception as ex:
                    st.error(f"Download failed: {ex}")

    st.divider()
    st.subheader("Overlap Zone — Camera 1")
    ov1s = st.slider("Cam1 Start X", 0, 1280, int(s.get("overlap_start_cam1_x", 900)), 10)
    ov1e = st.slider("Cam1 End X",   0, 1280, int(s.get("overlap_end_cam1_x",   1280)), 10)
    st.subheader("Overlap Zone — Camera 2")
    ov2s = st.slider("Cam2 Start X", 0, 1280, int(s.get("overlap_start_cam2_x", 0)),   10)
    ov2e = st.slider("Cam2 End X",   0, 1280, int(s.get("overlap_end_cam2_x",   380)), 10)

    st.divider()
    st.subheader("Belt & Ejector")
    belt_spd = st.slider("Belt speed (m/s)",     0.10, 1.0,
                         float(s.get("belt_speed_ms", 0.3)), 0.05)
    eject_d  = st.slider("Ejector distance (m)", 0.10, 2.0,
                         float(s.get("ejector_distance_m", 0.45)), 0.05)
    st.caption(f"Ejector delay: **{eject_d/max(belt_spd,0.01):.2f} s**")
    simulated = st.checkbox("Simulated ejector",
                            value=bool(s.get("simulated_ejector", True)))

    st.divider()
    st.subheader("Cross-Camera Matching")
    match_iou = st.slider("IoU threshold",  0.10, 0.80,
                          float(s.get("cross_cam_iou_threshold", 0.30)), 0.05)
    match_win = st.slider("Time window (s)", 0.5, 10.0,
                          float(s.get("match_time_window_s", 2.0)), 0.5)

    st.divider()
    st.subheader("Size Estimation")
    cal = st.slider("Calibration factor (mm²/px)", 0.001, 0.10,
                    float(s.get("calibration_factor", 0.014)), 0.001, format="%.3f")

    sc1, sc2 = st.columns(2)
    if sc1.button("💾 Save Settings", use_container_width=True, type="primary"):
        save_settings({
            "model_path": model_path,
            "overlap_start_cam1_x": ov1s, "overlap_end_cam1_x": ov1e,
            "overlap_start_cam2_x": ov2s, "overlap_end_cam2_x": ov2e,
            "belt_speed_ms": belt_spd, "ejector_distance_m": eject_d,
            "simulated_ejector": simulated,
            "cross_cam_iou_threshold": match_iou,
            "match_time_window_s": match_win,
            "calibration_factor": cal,
        })
        st.success("Settings saved. Restart pipeline to apply.")
    if sc2.button("↺ Reset Defaults", use_container_width=True):
        reset_settings()
        st.info("Defaults restored.")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
refresh_ms = 150 if _state.get("running") else 2000
time.sleep(refresh_ms / 1000)
st.rerun()

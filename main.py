"""
main.py — Onion Quality Grading System entrypoint.
Starts camera threads, cross-camera matcher, grading engine, ejector, sync worker,
and launches the Streamlit dashboard in a subprocess.

Usage:
  python main.py [--batch-id BATCH001] [--farmer "Ramesh Patil"]
                 [--no-cam2] [--no-dashboard] [--simulated]
"""
import argparse
import collections
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from camera import CameraThread
from config import load_settings, save_settings
from database import OnionDatabase
from ejector import EjectorController
from grad_cam import OnionGradCAM
from grader import GradeDecisionEngine
from matcher import CrossCameraMatcher
from model_wrapper import OnionModel
from sync import SyncWorker
from utils import DetectionEvent, setup_logging

logger = setup_logging()

PROJECT_DIR = Path(__file__).parent
SETTINGS_POLL_INTERVAL = 2.0  # seconds between settings reload


def parse_args():
    p = argparse.ArgumentParser(description="Onion Quality Grading System")
    p.add_argument("--batch-id", default="BATCH001", help="Batch identifier")
    p.add_argument("--farmer", default="", help="Farmer/supplier name")
    p.add_argument("--no-cam2", action="store_true", help="Single-camera mode")
    p.add_argument("--no-dashboard", action="store_true", help="Skip Streamlit launch")
    p.add_argument("--simulated", action="store_true", help="Simulated ejector (default)")
    return p.parse_args()


def _is_complete(events: list, window_s: float) -> bool:
    """
    A global_id group is ready for grading when:
    - Both cameras have reported, OR
    - The oldest event is older than the match time window (other camera missed it).
    """
    cam_ids = {e.cam_id for e in events}
    if len(cam_ids) >= 2:
        return True
    oldest = min(e.timestamp for e in events)
    return (time.time() - oldest) > window_s


def main():
    args = parse_args()
    settings = load_settings()

    if args.simulated:
        settings["simulated_ejector"] = True
        save_settings({"simulated_ejector": True})

    logger.info("=== Onion Grading System Starting ===")
    logger.info(f"Batch: {args.batch_id}  Farmer: {args.farmer}")

    db = OnionDatabase()
    model = OnionModel(settings)
    ejector = EjectorController(settings)
    matcher = CrossCameraMatcher(settings)
    engine = GradeDecisionEngine(settings)
    grad_cam = OnionGradCAM(model)
    sync_worker = SyncWorker(db, settings)

    event_queue: queue.Queue = queue.Queue(maxsize=500)
    stop_event = threading.Event()

    cam1 = CameraThread(1, model, event_queue, stop_event, settings)
    cam2 = CameraThread(2, model, event_queue, stop_event, settings) if not args.no_cam2 else None

    # Launch Streamlit dashboard
    dash_proc = None
    if not args.no_dashboard:
        dash_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             str(PROJECT_DIR / "dashboard.py"),
             "--server.headless", "true",
             "--server.port", "8501"],
            cwd=str(PROJECT_DIR),
        )
        logger.info("Dashboard: http://localhost:8501")

    sync_worker.start()
    cam1.start()
    if cam2:
        cam2.start()

    # Pending groups: global_id → list of DetectionEvents
    pending: dict = collections.defaultdict(list)
    last_settings_reload = time.time()

    logger.info("Processing loop running. Press Ctrl+C to stop.")
    try:
        while True:
            # Poll for settings changes every 2 seconds
            if time.time() - last_settings_reload > SETTINGS_POLL_INTERVAL:
                new_settings = load_settings()
                if new_settings != settings:
                    settings = new_settings
                    model.reload(settings)
                    ejector.update_settings(settings)
                    matcher.update_settings(settings)
                    engine.update_settings(settings)
                    sync_worker.update_settings(settings)
                    cam1.update_settings(settings)
                    if cam2:
                        cam2.update_settings(settings)
                    logger.info("Settings reloaded.")
                last_settings_reload = time.time()

            # Drain event queue
            try:
                event: DetectionEvent = event_queue.get(timeout=0.1)
            except queue.Empty:
                # Flush any groups that have timed out
                window = settings.get("match_time_window_s", 2.0)
                ready_ids = [
                    gid for gid, evts in list(pending.items())
                    if _is_complete(evts, window)
                ]
                for gid in ready_ids:
                    _process_group(gid, pending.pop(gid), engine, db, ejector,
                                   grad_cam, settings, args.batch_id, args.farmer,
                                   model.grading_active)
                continue

            event = matcher.process(event)
            if event.global_id is None:
                continue

            pending[event.global_id].append(event)

            window = settings.get("match_time_window_s", 2.0)
            if _is_complete(pending[event.global_id], window):
                gid = event.global_id
                _process_group(gid, pending.pop(gid), engine, db, ejector,
                               grad_cam, settings, args.batch_id, args.farmer,
                               model.grading_active)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        stop_event.set()
        cam1.join(timeout=3)
        if cam2:
            cam2.join(timeout=3)
        sync_worker.stop()
        if dash_proc:
            dash_proc.terminate()
        logger.info("Shutdown complete.")


def _process_group(
    global_id: int,
    events: list,
    engine: GradeDecisionEngine,
    db: OnionDatabase,
    ejector: EjectorController,
    grad_cam: OnionGradCAM,
    settings: dict,
    batch_id: str,
    farmer: str,
    grading_active: bool,
) -> None:
    cam1_ev = next((e for e in events if e.cam_id == 1), None)
    cam2_ev = next((e for e in events if e.cam_id == 2), None)

    decision = engine.decide(
        global_id=global_id,
        cam1_event=cam1_ev,
        cam2_event=cam2_ev,
        batch_id=batch_id,
        farmer_name=farmer,
        grading_active=grading_active,
    )

    db.insert_grade(decision.to_db_row())
    logger.info(
        f"Onion #{global_id}: cam1={decision.cam1_grade} cam2={decision.cam2_grade} "
        f"→ {decision.final_grade} | {decision.estimated_diameter_mm}mm"
    )

    if decision.ejected:
        ejector.schedule_ejection(global_id)

    # Grad-CAM for disputed onions
    if engine.is_disputed(cam1_ev, cam2_ev):
        for ev in [cam1_ev, cam2_ev]:
            if ev and ev.frame is not None:
                path = grad_cam.explain(ev.frame, ev.bbox, global_id, ev.cam_id)
                if path:
                    db.insert_gradcam(global_id,
                                      path if ev.cam_id == 1 else None,
                                      path if ev.cam_id == 2 else None,
                                      None)
        ev.frame = None  # release memory


if __name__ == "__main__":
    main()

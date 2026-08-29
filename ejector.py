"""
EjectorController: fires ejection signal after EJECTOR_DELAY_S for rejected onions.
Real mode: writes to a serial port (solenoid valve relay).
Simulated mode: logs the event.
"""
import threading
from typing import List

from utils import now_iso, setup_logging

logger = setup_logging()


class EjectorController:
    def __init__(self, settings: dict):
        self._settings = settings
        self._recent: List[dict] = []
        self._lock = threading.Lock()
        self._ser = None
        if not settings.get("simulated_ejector", True):
            self._init_serial()

    def update_settings(self, settings: dict) -> None:
        self._settings = settings

    def _init_serial(self) -> None:
        try:
            import serial
            port = self._settings.get("ejector_port", "COM3")
            self._ser = serial.Serial(port, baudrate=9600, timeout=1)
            logger.info(f"Ejector serial port opened: {port}")
        except Exception as e:
            logger.warning(f"Serial ejector unavailable ({e}), falling back to simulated.")
            self._ser = None

    def schedule_ejection(self, global_id: int) -> None:
        delay = self._settings.get("ejector_delay_s", 1.5)
        timer = threading.Timer(delay, self._fire, args=[global_id])
        timer.daemon = True
        timer.start()

    def _fire(self, global_id: int) -> None:
        ts = now_iso()
        if self._ser is not None:
            try:
                self._ser.write(b"EJECT\n")
            except Exception as e:
                logger.error(f"Serial write failed: {e}")
        logger.info(f"[EJECTOR] global_id={global_id} at {ts}")
        with self._lock:
            self._recent.append({"global_id": global_id, "time": ts})
            if len(self._recent) > 50:
                self._recent.pop(0)

    def recent(self, n: int = 10) -> List[dict]:
        with self._lock:
            return list(self._recent[-n:])

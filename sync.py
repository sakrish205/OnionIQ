"""
SyncWorker: daemon thread that periodically POSTs pending SQLite rows to a
central REST endpoint when internet is available. Rows stay 'pending' offline.
"""
import threading
import time
from typing import TYPE_CHECKING

from utils import setup_logging

if TYPE_CHECKING:
    from database import OnionDatabase

logger = setup_logging()
_CHECK_INTERVAL = 30  # seconds


class SyncWorker(threading.Thread):
    def __init__(self, db: "OnionDatabase", settings: dict):
        super().__init__(daemon=True, name="SyncWorker")
        self._db = db
        self._settings = settings
        self._stop = threading.Event()

    def update_settings(self, settings: dict) -> None:
        self._settings = settings

    def run(self) -> None:
        logger.info("SyncWorker started.")
        while not self._stop.wait(timeout=_CHECK_INTERVAL):
            if self._has_internet():
                self._push_pending()

    def stop(self) -> None:
        self._stop.set()

    def _has_internet(self) -> bool:
        try:
            import requests
            requests.get("http://8.8.8.8", timeout=2)
            return True
        except Exception:
            return False

    def _push_pending(self) -> None:
        rows = self._db.get_pending_sync()
        if not rows:
            return
        endpoint = self._settings.get("sync_endpoint", "")
        if not endpoint or "central-server" in endpoint:
            # Placeholder endpoint — skip silently
            return
        try:
            import requests
            resp = requests.post(endpoint, json=rows, timeout=10)
            if resp.status_code == 200:
                ids = [r["global_id"] for r in rows]
                self._db.mark_synced(ids)
                logger.info(f"Synced {len(ids)} records to {endpoint}.")
            else:
                logger.warning(f"Sync endpoint returned {resp.status_code}.")
        except Exception as e:
            logger.warning(f"Sync failed: {e}")

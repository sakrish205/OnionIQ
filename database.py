import sqlite3
import threading
from typing import List, Optional

from config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS onion_grades (
    global_id             INTEGER PRIMARY KEY,
    final_grade           TEXT NOT NULL,
    defect_type           TEXT,
    estimated_diameter_mm REAL,
    cam1_confidence       REAL,
    ejected               INTEGER DEFAULT 0,
    timestamp             TEXT NOT NULL,
    batch_id              TEXT
);

CREATE INDEX IF NOT EXISTS idx_batch ON onion_grades(batch_id);
CREATE INDEX IF NOT EXISTS idx_ts    ON onion_grades(timestamp);
"""


class OnionDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        conn = self._connect()
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_grade(self, row: dict) -> None:
        sql = """
            INSERT OR REPLACE INTO onion_grades
            (global_id, final_grade, defect_type, estimated_diameter_mm,
             cam1_confidence, ejected, timestamp, batch_id)
            VALUES
            (:global_id, :final_grade, :defect_type, :estimated_diameter_mm,
             :cam1_confidence, :ejected, :timestamp, :batch_id)
        """
        with self._lock:
            conn = self._connect()
            conn.execute(sql, row)
            conn.commit()
            conn.close()

    def get_recent(self, n: int = 50, batch_id: Optional[str] = None) -> List[dict]:
        conn = self._connect()
        if batch_id:
            rows = conn.execute(
                "SELECT * FROM onion_grades WHERE batch_id=? ORDER BY timestamp DESC LIMIT ?",
                (batch_id, n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM onion_grades ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_batch_summary(self, batch_id: str) -> dict:
        conn = self._connect()
        rows = conn.execute(
            "SELECT final_grade, defect_type, estimated_diameter_mm FROM onion_grades WHERE batch_id=?",
            (batch_id,),
        ).fetchall()
        conn.close()
        total = len(rows)
        counts: dict = {}
        defects: dict = {}
        diameters = []
        for r in rows:
            g = r["final_grade"]
            counts[g] = counts.get(g, 0) + 1
            d = r["defect_type"]
            if d:
                defects[d] = defects.get(d, 0) + 1
            if r["estimated_diameter_mm"]:
                diameters.append(r["estimated_diameter_mm"])
        return {
            "total": total,
            "counts": counts,
            "defects": defects,
            "avg_diameter_mm": sum(diameters) / len(diameters) if diameters else 0.0,
        }

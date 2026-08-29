import sqlite3
import threading
from dataclasses import asdict
from typing import List, Optional

from config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS onion_grades (
    global_id             INTEGER PRIMARY KEY,
    cam1_grade            TEXT,
    cam2_grade            TEXT,
    final_grade           TEXT NOT NULL,
    defect_type           TEXT,
    estimated_diameter_mm REAL,
    cam1_confidence       REAL,
    cam2_confidence       REAL,
    mask_area_cam1        INTEGER,
    mask_area_cam2        INTEGER,
    ejected               INTEGER DEFAULT 0,
    timestamp             TEXT NOT NULL,
    batch_id              TEXT,
    farmer_name           TEXT,
    centre_id             TEXT,
    sync_status           TEXT DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_batch ON onion_grades(batch_id);
CREATE INDEX IF NOT EXISTS idx_sync  ON onion_grades(sync_status);
CREATE INDEX IF NOT EXISTS idx_ts    ON onion_grades(timestamp);

CREATE TABLE IF NOT EXISTS gradcam_evidence (
    global_id   INTEGER PRIMARY KEY,
    cam1_path   TEXT,
    cam2_path   TEXT,
    overlay_path TEXT
);
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
            (global_id, cam1_grade, cam2_grade, final_grade, defect_type,
             estimated_diameter_mm, cam1_confidence, cam2_confidence,
             mask_area_cam1, mask_area_cam2, ejected, timestamp,
             batch_id, farmer_name, centre_id, sync_status)
            VALUES
            (:global_id, :cam1_grade, :cam2_grade, :final_grade, :defect_type,
             :estimated_diameter_mm, :cam1_confidence, :cam2_confidence,
             :mask_area_cam1, :mask_area_cam2, :ejected, :timestamp,
             :batch_id, :farmer_name, :centre_id, 'pending')
        """
        with self._lock:
            conn = self._connect()
            conn.execute(sql, row)
            conn.commit()
            conn.close()

    def insert_gradcam(self, global_id: int, cam1_path: Optional[str],
                       cam2_path: Optional[str], overlay_path: Optional[str]) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO gradcam_evidence VALUES (?,?,?,?)",
                (global_id, cam1_path, cam2_path, overlay_path),
            )
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

    def get_disputed(self, batch_id: Optional[str] = None) -> List[dict]:
        conn = self._connect()
        base = """
            SELECT og.*, ge.cam1_path, ge.cam2_path, ge.overlay_path
            FROM onion_grades og
            LEFT JOIN gradcam_evidence ge ON og.global_id = ge.global_id
            WHERE og.cam1_grade IS NOT NULL AND og.cam2_grade IS NOT NULL
              AND og.cam1_grade != og.cam2_grade
        """
        if batch_id:
            rows = conn.execute(base + " AND og.batch_id=?", (batch_id,)).fetchall()
        else:
            rows = conn.execute(base).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_pending_sync(self) -> List[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM onion_grades WHERE sync_status='pending'"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def mark_synced(self, global_ids: List[int]) -> None:
        if not global_ids:
            return
        placeholders = ",".join("?" * len(global_ids))
        with self._lock:
            conn = self._connect()
            conn.execute(
                f"UPDATE onion_grades SET sync_status='synced' WHERE global_id IN ({placeholders})",
                global_ids,
            )
            conn.commit()
            conn.close()

    def get_farmer_fingerprint(self, farmer_name: str) -> dict:
        conn = self._connect()
        rows = conn.execute(
            "SELECT final_grade, defect_type, estimated_diameter_mm, batch_id "
            "FROM onion_grades WHERE farmer_name=?",
            (farmer_name,),
        ).fetchall()
        conn.close()
        total = len(rows)
        if total == 0:
            return {"total": 0, "farmer_name": farmer_name}
        counts: dict = {}
        defects: dict = {}
        diameters = []
        batches = set()
        for r in rows:
            counts[r["final_grade"]] = counts.get(r["final_grade"], 0) + 1
            if r["defect_type"]:
                defects[r["defect_type"]] = defects.get(r["defect_type"], 0) + 1
            if r["estimated_diameter_mm"]:
                diameters.append(r["estimated_diameter_mm"])
            batches.add(r["batch_id"])
        reject_count = counts.get("reject", 0)
        return {
            "farmer_name": farmer_name,
            "total": total,
            "sessions": len(batches),
            "counts": counts,
            "defects": defects,
            "reject_rate_pct": round(reject_count / total * 100, 1),
            "grade_a_rate_pct": round(counts.get("grade_a", 0) / total * 100, 1),
            "avg_diameter_mm": round(sum(diameters) / len(diameters), 1) if diameters else 0.0,
        }

    def count_pending_sync(self) -> int:
        conn = self._connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM onion_grades WHERE sync_status='pending'"
        ).fetchone()[0]
        conn.close()
        return n

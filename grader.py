"""
GradeDecisionEngine: worst-case grading logic.
In detection-only mode (no grade classes) → final_grade = "detected".
"""
from dataclasses import dataclass
from typing import Optional

from utils import DetectionEvent, mm2_to_diameter, mask_area_to_mm2, now_iso
from config import GRADE_CLASSES

SEVERITY: dict = {
    "grade_a": 0,
    "grade_b": 1,
    "grade_c": 2,
    "thrips_damage": 2,
    "sunscald": 2,
    "bruising": 2,
    "sprouting": 2,
    "neck_rot": 3,
    "rot": 3,
    "reject": 3,
    # Detection-only class
    "onion": 0,
    "detected": 0,
}

SEVERITY_TO_GRADE: dict = {
    0: "grade_a",
    1: "grade_b",
    2: "grade_c",
    3: "reject",
}

EXTENDED_DEFECTS = {"thrips_damage", "neck_rot", "sunscald", "bruising"}


@dataclass
class GradeDecision:
    global_id: int
    cam1_grade: Optional[str]
    cam2_grade: Optional[str]
    final_grade: str
    defect_type: Optional[str]
    estimated_diameter_mm: float
    cam1_confidence: Optional[float]
    cam2_confidence: Optional[float]
    mask_area_cam1: int
    mask_area_cam2: int
    ejected: bool
    timestamp: str
    batch_id: str
    farmer_name: str
    centre_id: str

    def to_db_row(self) -> dict:
        return {
            "global_id": self.global_id,
            "cam1_grade": self.cam1_grade,
            "cam2_grade": self.cam2_grade,
            "final_grade": self.final_grade,
            "defect_type": self.defect_type,
            "estimated_diameter_mm": self.estimated_diameter_mm,
            "cam1_confidence": self.cam1_confidence,
            "cam2_confidence": self.cam2_confidence,
            "mask_area_cam1": self.mask_area_cam1,
            "mask_area_cam2": self.mask_area_cam2,
            "ejected": int(self.ejected),
            "timestamp": self.timestamp,
            "batch_id": self.batch_id,
            "farmer_name": self.farmer_name,
            "centre_id": self.centre_id,
        }


class GradeDecisionEngine:
    def __init__(self, settings: dict):
        self._settings = settings

    def update_settings(self, settings: dict) -> None:
        self._settings = settings

    def decide(
        self,
        global_id: int,
        cam1_event: Optional[DetectionEvent],
        cam2_event: Optional[DetectionEvent],
        batch_id: str,
        farmer_name: str,
        grading_active: bool = False,
    ) -> GradeDecision:
        cam1_grade = cam1_event.class_name if cam1_event else None
        cam2_grade = cam2_event.class_name if cam2_event else None

        if not grading_active:
            final_grade = "detected"
        else:
            grades = [g for g in [cam1_grade, cam2_grade] if g is not None]
            if not grades:
                final_grade = "detected"
            else:
                worst_sev = max(SEVERITY.get(g, 0) for g in grades)
                final_grade = SEVERITY_TO_GRADE.get(worst_sev, "grade_a")

        defect_type = self._extract_defect([cam1_grade, cam2_grade])

        # Size estimation
        cal = self._settings.get("calibration_factor", 0.014)
        area1 = cam1_event.mask_area_px if cam1_event else 0
        area2 = cam2_event.mask_area_px if cam2_event else 0
        area = area1 or area2
        diameter = round(mm2_to_diameter(mask_area_to_mm2(area, cal)), 1)

        ejected = grading_active and final_grade == "reject"

        return GradeDecision(
            global_id=global_id,
            cam1_grade=cam1_grade,
            cam2_grade=cam2_grade,
            final_grade=final_grade,
            defect_type=defect_type,
            estimated_diameter_mm=diameter,
            cam1_confidence=cam1_event.confidence if cam1_event else None,
            cam2_confidence=cam2_event.confidence if cam2_event else None,
            mask_area_cam1=area1,
            mask_area_cam2=area2,
            ejected=ejected,
            timestamp=now_iso(),
            batch_id=batch_id,
            farmer_name=farmer_name,
            centre_id=self._settings.get("centre_id", "CENTRE_01"),
        )

    def _extract_defect(self, grades: list) -> Optional[str]:
        for g in grades:
            if g and g in EXTENDED_DEFECTS:
                return g
        return None

    def is_disputed(
        self, cam1_event: Optional[DetectionEvent], cam2_event: Optional[DetectionEvent]
    ) -> bool:
        if not cam1_event or not cam2_event:
            return False
        sev1 = SEVERITY.get(cam1_event.class_name, 0)
        sev2 = SEVERITY.get(cam2_event.class_name, 0)
        return abs(sev1 - sev2) >= 2

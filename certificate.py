"""
CertificateGenerator: produces a batch grading certificate PDF using ReportLab.
"""
import os
import time
from typing import TYPE_CHECKING

from utils import setup_logging

if TYPE_CHECKING:
    from database import OnionDatabase

logger = setup_logging()


def generate(batch_id: str, db: "OnionDatabase", output_dir: str = ".") -> str:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        logger.error("reportlab not installed. Run: pip install reportlab")
        return ""

    summary = db.get_batch_summary(batch_id)
    total = summary["total"]
    counts = summary.get("counts", {})
    defects = summary.get("defects", {})
    avg_dia = summary.get("avg_diameter_mm", 0.0)

    filename = f"certificate_{batch_id}_{int(time.time())}.pdf"
    out_path = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    elements = []

    # Header
    elements.append(Paragraph(
        "Ministry of Consumer Affairs, Food & Public Distribution",
        styles["Heading2"]
    ))
    elements.append(Paragraph(
        "Onion Quality Grading Certificate — SIH26031",
        styles["Heading1"]
    ))
    elements.append(Spacer(1, 6*mm))

    # Batch info
    info_data = [
        ["Batch ID", batch_id],
        ["Generated", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())],
        ["Total Onions", str(total)],
        ["Avg Diameter", f"{avg_dia:.1f} mm"],
    ]
    info_table = Table(info_data, colWidths=[60*mm, 100*mm])
    info_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 6*mm))

    # Grade summary
    elements.append(Paragraph("Grade Distribution", styles["Heading3"]))
    grade_data = [["Grade", "Count", "Percentage"]]
    grade_order = ["grade_a", "grade_b", "grade_c", "reject",
                   "sprouting", "rot", "detected"]
    for g in grade_order:
        c = counts.get(g, 0)
        if c > 0 or g in ("grade_a", "grade_b", "grade_c", "reject"):
            pct = f"{c / total * 100:.1f}%" if total > 0 else "—"
            grade_data.append([g.replace("_", " ").title(), str(c), pct])

    grade_table = Table(grade_data, colWidths=[60*mm, 40*mm, 50*mm])
    grade_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(grade_table)
    elements.append(Spacer(1, 6*mm))

    # Defect breakdown (if any)
    if defects:
        elements.append(Paragraph("Defect Breakdown", styles["Heading3"]))
        defect_data = [["Defect Type", "Count"]]
        for d, c in defects.items():
            defect_data.append([d.replace("_", " ").title(), str(c)])
        defect_table = Table(defect_data, colWidths=[80*mm, 40*mm])
        defect_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.orange),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        elements.append(defect_table)
        elements.append(Spacer(1, 6*mm))

    # Footer
    elements.append(Paragraph(
        "This certificate is generated automatically by the AI-based onion grading system. "
        "Records are stored in an immutable local database and cannot be altered after write.",
        styles["Italic"]
    ))

    doc.build(elements)
    logger.info(f"Certificate saved: {out_path}")
    return out_path

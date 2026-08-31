from collections.abc import Mapping
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_NAVY = HexColor("#10233D")
_BLUE = HexColor("#246BCE")
_PALE_BLUE = HexColor("#EAF2FC")
_PALE_GRAY = HexColor("#F4F6F8")
_MID_GRAY = HexColor("#D8DEE6")
_TEXT = HexColor("#18212B")
_MUTED = HexColor("#52606D")


def _safe_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        text = default
    else:
        text = str(value).strip()
        if not text:
            text = default
    return escape(text, {'"': "&quot;", "'": "&#39;"}).replace("\n", "<br/>")


def _format_integer(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return str(value) if value not in (None, "") else "N/A"


def _format_decimal(value: Any, decimal_places: int) -> str:
    try:
        return f"{float(value):.{decimal_places}f}"
    except (TypeError, ValueError, OverflowError):
        return str(value) if value not in (None, "") else "N/A"


def _paragraph(value: Any, style: ParagraphStyle, default: str = "N/A") -> Paragraph:
    return Paragraph(_safe_text(value, default=default), style)


def _report_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="InspectionTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=_NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=_MUTED,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=_NAVY,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=_TEXT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionLabel",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=_NAVY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionValue",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=_TEXT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionTableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.5,
            alignment=TA_CENTER,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionTableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.2,
            alignment=TA_CENTER,
            textColor=_TEXT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InspectionNote",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.5,
            textColor=_MUTED,
            leftIndent=8,
            firstLineIndent=-8,
            spaceAfter=3,
        )
    )
    return styles


def _section_heading(label: str, styles) -> Paragraph:
    return Paragraph(_safe_text(label), styles["InspectionSection"])


def _key_value_table(rows, styles) -> Table:
    data = [
        [
            _paragraph(label, styles["InspectionLabel"]),
            _paragraph(value, styles["InspectionValue"]),
        ]
        for label, value in rows
    ]
    table = Table(data, colWidths=[48 * mm, 126 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), _PALE_BLUE),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, _MID_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, _MID_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _defect_table(report_defects, styles) -> LongTable:
    header_labels = (
        "Defect ID",
        "Area",
        "Size",
        "Severity Score",
        "Severity",
        "Priority",
        "Spatial Region",
    )
    rows = [
        [
            _paragraph(label, styles["InspectionTableHeader"])
            for label in header_labels
        ]
    ]

    for defect in list(report_defects):
        defect_id = defect.get("id", "N/A") if isinstance(defect, Mapping) else "N/A"
        area = defect.get("area", 0) if isinstance(defect, Mapping) else 0
        width = defect.get("width", 0) if isinstance(defect, Mapping) else 0
        height = defect.get("height", 0) if isinstance(defect, Mapping) else 0
        severity_score = (
            defect.get("severity_score", 0.0) if isinstance(defect, Mapping) else 0.0
        )
        severity = (
            defect.get("severity_level", "N/A")
            if isinstance(defect, Mapping)
            else "N/A"
        )
        priority = (
            defect.get("priority_rank", "N/A")
            if isinstance(defect, Mapping)
            else "N/A"
        )
        spatial_region = (
            defect.get("spatial_region", "N/A")
            if isinstance(defect, Mapping)
            else "N/A"
        )
        display_id = "N/A" if defect_id in (None, "", "N/A") else f"D{defect_id}"
        display_priority = (
            "N/A" if priority in (None, "", "N/A") else f"#{priority}"
        )
        values = (
            display_id,
            f"{_format_integer(area)} px²",
            f"{_format_integer(width)} × {_format_integer(height)} px",
            _format_decimal(severity_score, 6),
            severity,
            display_priority,
            spatial_region,
        )
        rows.append(
            [
                _paragraph(value, styles["InspectionTableCell"])
                for value in values
            ]
        )

    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("BOX", (0, 0), (-1, -1), 0.6, _NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, _MID_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if len(rows) == 1:
        rows.append(
            [
                _paragraph(
                    "No defect regions were reported.",
                    styles["InspectionTableCell"],
                ),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        table_style.extend(
            [
                ("SPAN", (0, 1), (-1, 1)),
                ("BACKGROUND", (0, 1), (-1, 1), _PALE_GRAY),
                ("ALIGN", (0, 1), (-1, 1), "CENTER"),
            ]
        )
    else:
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                table_style.append(
                    ("BACKGROUND", (0, row_index), (-1, row_index), _PALE_GRAY)
                )

    table = LongTable(
        rows,
        colWidths=[15 * mm, 22 * mm, 24 * mm, 28 * mm, 22 * mm, 18 * mm, 45 * mm],
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle(table_style))
    return table


def _draw_page_frame(canvas, document) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setTitle("PCB Defect Inspection Report")
    canvas.setAuthor("PCB Inspection System")
    canvas.setCreator("PCB Inspection System - ReportLab")
    canvas.setStrokeColor(_BLUE)
    canvas.setLineWidth(1.2)
    canvas.line(18 * mm, page_height - 13 * mm, page_width - 18 * mm, page_height - 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_MUTED)
    canvas.drawString(18 * mm, 10 * mm, "PCB Inspection System | Single Inspection Report")
    canvas.drawRightString(
        page_width - 18 * mm,
        10 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def build_inspection_pdf(
    inspection_report: Mapping[str, Any],
    inspection_conclusion: str,
) -> bytes:
    if not isinstance(inspection_report, Mapping):
        raise TypeError("inspection_report must be a mapping of completed report data.")

    styles = _report_styles()
    report_defects = inspection_report.get("defects", []) or []
    if isinstance(report_defects, (str, bytes)):
        raise TypeError("inspection_report['defects'] must be a sequence of defect rows.")

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="PCB Defect Inspection Report",
        author="PCB Inspection System",
        subject="Automated PCB defect inspection results and findings",
    )

    story = [
        Paragraph("PCB Defect Inspection Report", styles["InspectionTitle"]),
        Paragraph(
            "Automated export of completed inspection results and findings",
            styles["InspectionSubtitle"],
        ),
        HRFlowable(width="100%", thickness=1, color=_BLUE, spaceAfter=6),
        _section_heading("A. Inspection Information", styles),
        _key_value_table(
            (
                ("Test Image", inspection_report.get("test_filename", "N/A")),
                (
                    "Reference PCB",
                    inspection_report.get("template_filename", "N/A"),
                ),
                ("Generated Time", inspection_report.get("timestamp", "N/A")),
                (
                    "Processing Time",
                    f"{_format_decimal(inspection_report.get('processing_time'), 2)} seconds",
                ),
            ),
            styles,
        ),
        _section_heading("B. Inspection Summary", styles),
        _key_value_table(
            (
                (
                    "Inspection Result",
                    inspection_report.get("inspection_status", "N/A"),
                ),
                (
                    "Total Defects",
                    _format_integer(inspection_report.get("total_defects", 0)),
                ),
                (
                    "Total Defect Area",
                    f"{_format_integer(inspection_report.get('total_defect_area', 0))} px²",
                ),
                (
                    "Detected Coverage",
                    f"{_format_decimal(inspection_report.get('defect_coverage_percentage', 0.0), 4)}%",
                ),
                ("Highest Severity", inspection_report.get("severity", "N/A")),
            ),
            styles,
        ),
        _section_heading("C. Inspection Conclusion", styles),
        Paragraph(
            _safe_text(inspection_conclusion, default="No conclusion is available."),
            styles["InspectionBody"],
        ),
        CondPageBreak(48 * mm),
        _section_heading("D. Defect Assessment Summary", styles),
        _defect_table(report_defects, styles),
        CondPageBreak(48 * mm),
        _section_heading("E. Interpretation Note", styles),
        Paragraph(
            "- Severity represents relative geometric extent.",
            styles["InspectionNote"],
        ),
        Paragraph(
            "- Severity is not a validated manufacturing acceptance limit.",
            styles["InspectionNote"],
        ),
        Paragraph(
            "- Measurements remain in pixels and pixel².",
            styles["InspectionNote"],
        ),
        Paragraph(
            "- Physical pixel-to-mm scaling is unavailable because the dataset does not "
            "provide a validated physical reference.",
            styles["InspectionNote"],
        ),
        Spacer(1, 3 * mm),
    ]

    document.build(
        story,
        onFirstPage=_draw_page_frame,
        onLaterPages=_draw_page_frame,
    )
    pdf_bytes = output.getvalue()
    output.close()
    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError("ReportLab did not produce a valid PDF byte stream.")
    return pdf_bytes

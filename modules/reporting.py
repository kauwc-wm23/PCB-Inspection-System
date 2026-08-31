
from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_inspection_summary(
    test_filename: str,
    template_filename: str,
    processing_time: float,
    defects: List[Dict[str, Any]],
    analysis_metrics: Dict[str, Any],
    evaluation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if evaluation_result is not None:
        total_defects = evaluation_result.get("total_defect_count", 0)
        total_area = evaluation_result.get("total_defect_area", 0)
        avg_area = evaluation_result.get("average_defect_area", 0.0)
        largest_area = evaluation_result.get("largest_defect_area", 0)
        inspection_status = evaluation_result.get("status_label", "NORMAL / PASS")
        coverage = evaluation_result.get("defect_coverage_percentage", 0.0)
        severity = evaluation_result.get("highest_severity_level", "N/A")
        highest_priority_id = evaluation_result.get("highest_priority_defect_id", "N/A")
        highest_priority_region = evaluation_result.get(
            "highest_priority_defect_region", "N/A"
        )
        most_concentrated_region = evaluation_result.get("most_concentrated_region", "N/A")
        largest_defect_region = evaluation_result.get("largest_defect_region", "N/A")
        spatial_distribution = evaluation_result.get("spatial_distribution", {})
        report_defects = evaluation_result.get("defects", defects)
    else:
        total_defects = analysis_metrics.get("total_defects", 0)
        total_area = analysis_metrics.get("total_defect_area", 0)
        avg_area = analysis_metrics.get("average_area", 0.0)
        largest_defect = analysis_metrics.get("largest_defect")
        largest_area = largest_defect.get("area", 0) if largest_defect else 0
        inspection_status = "No Defect Detected" if total_defects == 0 else "Defects Detected"
        coverage = 0.0
        severity = "N/A" if total_defects == 0 else "Not assessed"
        highest_priority_id = "N/A"
        highest_priority_region = "N/A"
        most_concentrated_region = "N/A"
        largest_defect_region = "N/A"
        spatial_distribution = {}
        report_defects = defects

    if total_defects == 0:
        summary_text = (
            "Inspection completed successfully. No valid defect regions were detected "
            "when the test PCB was compared with the supplied defect-free template."
        )
    else:
        region_word = "region" if total_defects == 1 else "regions"

        summary_text = (
            f"Inspection completed successfully. A total of {total_defects} potential defect "
            f"{region_word} were detected. The combined defect area was {total_area} pixels, "
            f"with an average defect area of {avg_area:.1f} pixels. "
            f"The largest detected defect region measured {largest_area} pixels. "
            f"Detected coverage was {coverage:.4f}%. The highest-priority defect was "
            f"Defect #{highest_priority_id} ({severity}) in {highest_priority_region}. "
            f"The most concentrated region was {most_concentrated_region}."
        )

    report = {
        "test_filename": test_filename,
        "template_filename": template_filename,
        "processing_time": processing_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inspection_status": inspection_status,
        "total_defects": total_defects,
        "total_defect_area": total_area,
        "average_defect_area": avg_area,
        "largest_defect_area": largest_area,
        "defect_coverage_percentage": coverage,
        "severity": severity,
        "highest_priority_defect_id": highest_priority_id,
        "highest_priority_defect_region": highest_priority_region,
        "most_concentrated_region": most_concentrated_region,
        "largest_defect_region": largest_defect_region,
        "spatial_distribution": spatial_distribution,
        "summary_text": summary_text,
        "defects": report_defects,
    }

    return report


def format_defect_table_data(defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    if not defects:
        return []

    table_data = []
    for defect in defects:
        bbox = defect.get("bounding_box", {})
        loc = defect.get("location", {})

        row = {
            "Defect ID": defect.get("id", ""),
            "Area (px²)": defect.get("area", 0),
            "Width (px)": defect.get("width", 0),
            "Height (px)": defect.get("height", 0),
            "Centroid X": f"{loc.get('x', 0):.1f}",
            "Centroid Y": f"{loc.get('y', 0):.1f}",
            "BBox (x, y, w, h)": f"({bbox.get('x', 0)}, {bbox.get('y', 0)}, {bbox.get('width', 0)}, {bbox.get('height', 0)})",
            "Area Ratio (%)": f"{defect.get('area_ratio', 0.0) * 100:.4f}",
            "Severity Score": f"{defect.get('severity_score', 0.0):.6f}",
            "Severity": defect.get("severity_level", "N/A"),
            "Priority": defect.get("priority_rank", "N/A"),
            "Region": defect.get("spatial_region", "N/A"),
        }
        table_data.append(row)

    return table_data


def build_inspection_conclusion(report: Dict[str, Any]) -> str:

    total_defects = int(report.get("total_defects", 0) or 0)
    coverage = float(report.get("defect_coverage_percentage", 0.0) or 0.0)
    if total_defects == 0:
        return (
            "No valid potential defect regions were identified. "
            f"Detected candidate coverage: {coverage:.4f}%."
        )

    highest_priority_id = report.get("highest_priority_defect_id", "N/A")
    report_defects = report.get("defects", []) or []
    highest_priority_defect = next(
        (
            defect
            for defect in report_defects
            if defect.get("id") == highest_priority_id
        ),
        None,
    )
    identified_text = (
        "1 potential defect region was identified."
        if total_defects == 1
        else f"{total_defects} potential defect regions were identified."
    )
    if highest_priority_defect is None:
        return f"{identified_text} Detected candidate coverage: {coverage:.4f}%."

    return (
        f"{identified_text} D{highest_priority_defect.get('id')} received Priority "
        f"#{highest_priority_defect.get('priority_rank')} with "
        f"{highest_priority_defect.get('severity_level', 'N/A')} relative geometric "
        "severity and is located in the "
        f"{highest_priority_defect.get('spatial_region', 'N/A')} region. "
        f"Detected candidate coverage: {coverage:.4f}%."
    )

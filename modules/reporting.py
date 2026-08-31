"""
=============================================================================
Module      : reporting.py
Project     : PCB Defect Inspection System

Description :
    Reporting support for Module 4 — PCB Inspection Interface.

    Processing Pipeline:
        1. Consume measured results from Modules 3 and 5
        2. Generate structured inspection summary report
        3. Provide GUI visualisation and automated report text
        4. Export results to PDF (optional)

    This module performs ONLY presentation and reporting. Module 5 remains
    responsible for severity, priority, and spatial calculations.
    It does NOT perform preprocessing, segmentation, or feature extraction.

    Responsibilities:
        - Automated defect summary generation
        - Image visualisation
        - Inspection result display
        - PDF report export (optional)

Usage:
    from modules.reporting import generate_inspection_summary

    summary = generate_inspection_summary(
        test_filename="test_pcb.jpg",
        template_filename="template_pcb.jpg",
        processing_time=2.45,
        defects=[...],                  # From Module 3
        analysis_metrics={...}          # From Module 3
    )

Author:
    PCB Inspection Team - Reporting Module
=============================================================================
"""

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
    """
    Generate a structured inspection summary report.

    This function consumes Module 3 features and Module 5 evaluation data to generate
    a high-level summary suitable for display and PDF export.

    Parameters
    ----------
    test_filename : str
        Name of the test/defective PCB image file.

    template_filename : str
        Name of the defect-free template image file.

    processing_time : float
        Total pipeline execution time in seconds.

    defects : List[Dict[str, Any]]
        List of defect dictionaries from Module 3 analyse_features().
        May be empty if no defects detected.

    analysis_metrics : Dict[str, Any]
        Inspection-level metrics from Module 3 analyse_features().
        Should contain:
            - total_defects
            - total_defect_area
            - average_area
            - largest_defect
            - smallest_defect

    evaluation_result : Dict[str, Any], optional
        Authoritative severity, priority, and spatial metrics from Module 5.
        If omitted, the legacy Module 3-only summary remains available.

    Returns
    -------
    Dict[str, Any]
        Structured report dictionary containing:
            {
                "test_filename": str,
                "template_filename": str,
                "processing_time": float,
                "timestamp": str,
                "inspection_status": str,
                "total_defects": int,
                "total_defect_area": int,
                "average_defect_area": float,
                "largest_defect_area": int,
                "summary_text": str,
                "defects": List[Dict]
            }

    Notes
    -----
    - Module 5 results are used when ``evaluation_result`` is supplied.
    - The legacy Module 3 status is retained only for backward compatibility.
    - The summary_text is generated automatically based on inspection results.
    - No defect classification is performed; results indicate "potential defects".
    """

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
        # Backward-compatible fallback for callers that do not yet pass Module 5.
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

    # Generate automated summary text
    if total_defects == 0:
        summary_text = (
            "Inspection completed successfully. No valid defect regions were detected "
            "when the test PCB was compared with the supplied defect-free template."
        )
    else:
        # Pluralise "region" / "regions"
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

    # Build report dictionary
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
    """
    Format defects for tabular display.

    Formats Module 3 measurements enriched by Module 5 severity and spatial
    assessment for presentation in a data table.

    Parameters
    ----------
    defects : List[Dict[str, Any]]
        Module 5 defect dictionaries (or legacy Module 3 dictionaries).

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries formatted for table display.

    Notes
    -----
    - Returns empty list if defects is empty
    - Centroid coordinates are formatted to 1 decimal place
    - Bounding box coordinates are integers
    """

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

"""
=============================================================================
Module      : reporting.py
Project     : PCB Defect Inspection System

Description :
    Module 4 — PCB Inspection Interface and Reporting Module.

    Processing Pipeline:
        1. Consume processed results from Modules 1–3
        2. Generate structured inspection summary report
        3. Provide GUI visualisation and automated report text
        4. Export results to PDF (optional)

    This module performs ONLY presentation and reporting.
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

from typing import Dict, Any, List
from datetime import datetime


def generate_inspection_summary(
    test_filename: str,
    template_filename: str,
    processing_time: float,
    defects: List[Dict[str, Any]],
    analysis_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a structured inspection summary report.

    This function consumes results from Modules 1–3 and generates
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
    - If defects is empty, inspection_status = "No Defect Detected"
    - If defects is non-empty, inspection_status = "Defects Detected"
    - The summary_text is generated automatically based on inspection results.
    - No defect classification is performed; results indicate "potential defects".
    """

    total_defects = analysis_metrics.get("total_defects", 0)
    total_area = analysis_metrics.get("total_defect_area", 0)
    avg_area = analysis_metrics.get("average_area", 0.0)
    largest_defect = analysis_metrics.get("largest_defect")
    largest_area = largest_defect.get("area", 0) if largest_defect else 0

    # Determine inspection status
    if total_defects == 0:
        inspection_status = "No Defect Detected"
    else:
        inspection_status = "Defects Detected"

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
            f"{region_word} were detected. The combined defect area was {total_area} px², "
            f"with an average defect area of {avg_area:.1f} px². "
            f"The largest detected defect region measured {largest_area} px²."
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
        "summary_text": summary_text,
        "defects": defects,
    }

    return report


def format_defect_table_data(defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format defects for tabular display.

    Extracts and formats defect data from Module 3 output for
    presentation in a data table.

    Parameters
    ----------
    defects : List[Dict[str, Any]]
        List of defect dictionaries from Module 3 analyse_features().

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
        }
        table_data.append(row)

    return table_data

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_paths import find_reference_image
from modules.inspection_pipeline import run_inspection_arrays
from modules.pdf_reporting import build_inspection_pdf
from modules.reporting import build_inspection_conclusion

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


BASELINE_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "images"
    / "Missing_hole"
    / "01_missing_hole_05.jpg"
)


def _defect(defect_id: int, priority: int, severity: str = "HIGH"):
    return {
        "id": defect_id,
        "area": 1200 + defect_id,
        "width": 40 + defect_id,
        "height": 20 + defect_id,
        "bounding_box": {
            "x": 10 * defect_id,
            "y": 12 * defect_id,
            "width": 40 + defect_id,
            "height": 20 + defect_id,
        },
        "location": {"x": 30.0 + defect_id, "y": 25.0 + defect_id},
        "centroid": {"x": 30.0 + defect_id, "y": 25.0 + defect_id},
        "area_ratio": 0.0001 * defect_id,
        "width_ratio": 0.001 * defect_id,
        "height_ratio": 0.001 * defect_id,
        "severity_score": 0.005 + defect_id / 100000.0,
        "severity_level": severity,
        "priority_rank": priority,
        "spatial_region": "TOP_LEFT" if defect_id % 2 else "BOTTOM_RIGHT",
    }


def _report(defects, **overrides):
    total_area = sum(defect["area"] for defect in defects)
    report = {
        "test_filename": "01_missing_hole_05.jpg",
        "template_filename": "01.JPG",
        "processing_time": 1.25,
        "timestamp": "2026-09-01 10:30:00",
        "inspection_status": "DEFECTIVE / FAIL",
        "total_defects": len(defects),
        "total_defect_area": total_area,
        "average_defect_area": total_area / len(defects) if defects else 0.0,
        "largest_defect_area": max(
            (defect["area"] for defect in defects),
            default=0,
        ),
        "defect_coverage_percentage": 0.0536 if defects else 0.0,
        "severity": "HIGH" if defects else "N/A",
        "highest_priority_defect_id": defects[0]["id"] if defects else "N/A",
        "highest_priority_defect_region": (
            defects[0]["spatial_region"] if defects else "N/A"
        ),
        "most_concentrated_region": "TOP_LEFT" if defects else "N/A",
        "largest_defect_region": "TOP_LEFT" if defects else "N/A",
        "spatial_distribution": {
            "TOP_LEFT": len(defects),
            "TOP_RIGHT": 0,
            "BOTTOM_LEFT": 0,
            "BOTTOM_RIGHT": 0,
        },
        "summary_text": "Completed inspection summary.",
        "defects": defects,
    }
    report.update(overrides)
    return report


def _read_pdf(pdf_bytes):
    if PdfReader is None:
        return None, ""
    reader = PdfReader(BytesIO(pdf_bytes))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return reader, extracted_text


def _assert_pdf(report, conclusion, expected_text=()):
    before = deepcopy(report)
    pdf_bytes = build_inspection_pdf(report, conclusion)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1500
    assert b"%%EOF" in pdf_bytes[-32:]
    assert report == before

    reader, extracted_text = _read_pdf(pdf_bytes)
    if reader is not None:
        assert len(reader.pages) >= 1
        assert "PCB Defect Inspection Report" in extracted_text
        for expected in expected_text:
            assert expected in extracted_text, expected
    return pdf_bytes, reader, extracted_text


def _test_defective_report():
    report = _report([_defect(1, 1), _defect(2, 2, "MEDIUM")])
    conclusion = (
        "2 potential defect regions were identified. D1 received Priority #1 with "
        "HIGH relative geometric severity and is located in the TOP_LEFT region."
    )
    _assert_pdf(
        report,
        conclusion,
        (
            report["test_filename"],
            report["template_filename"],
            "DEFECTIVE / FAIL",
            "HIGH",
            "D1",
            "TOP_LEFT",
        ),
    )


def _test_zero_defect_report():
    report = _report(
        [],
        inspection_status="NORMAL / PASS",
        severity="N/A",
        highest_priority_defect_id="N/A",
        highest_priority_defect_region="N/A",
    )
    _assert_pdf(
        report,
        "No valid potential defect regions were identified.",
        (
            "NORMAL / PASS",
            "No defect regions were reported.",
            "N/A",
        ),
    )


def _test_multi_page_report():
    defects = [
        _defect(defect_id, defect_id, "MEDIUM" if defect_id % 2 else "LOW")
        for defect_id in range(1, 61)
    ]
    report = _report(defects, severity="MEDIUM")
    _, reader, extracted_text = _assert_pdf(
        report,
        "Multiple potential defect regions were identified.",
        ("D1", "D60", "BOTTOM_RIGHT"),
    )
    if reader is not None:
        assert len(reader.pages) >= 2
        assert extracted_text.count("Defect ID") >= 2


def _test_special_characters():
    report = _report(
        [_defect(1, 1)],
        test_filename="board & <prototype> > sample.jpg",
        template_filename="reference & <clean>.JPG",
    )
    conclusion = "Review A & B <without> > unsupported conclusions."
    _assert_pdf(
        report,
        conclusion,
        (
            "board & <prototype> > sample.jpg",
            "reference & <clean>.JPG",
            "Review A & B <without> > unsupported conclusions.",
        ),
    )


def _test_accepted_baseline():
    reference_path = find_reference_image(BASELINE_IMAGE)
    assert reference_path is not None
    result = run_inspection_arrays(
        cv2.imread(str(BASELINE_IMAGE), cv2.IMREAD_COLOR),
        cv2.imread(str(reference_path), cv2.IMREAD_COLOR),
        BASELINE_IMAGE.name,
        reference_path.name,
    )
    report = result["inspection_report"]
    evaluation = result["evaluation"]
    assert result["calibration_metadata"]["status"] == "ALREADY_ALIGNED"
    assert result["calibration_metadata"]["warp_applied"] is False
    assert report["total_defects"] == 5
    assert report["total_defect_area"] == 2579
    assert round(report["defect_coverage_percentage"], 4) == 0.0536
    assert report["severity"] == "HIGH"
    assert evaluation["highest_priority_defect_id"] == 1
    assert evaluation["highest_priority_defect_region"] == "TOP_LEFT"

    conclusion = build_inspection_conclusion(report)
    pdf_bytes, _, extracted_text = _assert_pdf(
        report,
        conclusion,
        (
            "01_missing_hole_05.jpg",
            "01.JPG",
            "DEFECTIVE / FAIL",
            "5",
            "2,579",
            "0.0536%",
            "HIGH",
            "D1",
            "TOP_LEFT",
        ),
    )
    return pdf_bytes, extracted_text


def main() -> int:
    _test_defective_report()
    _test_zero_defect_report()
    _test_multi_page_report()
    _test_special_characters()
    baseline_pdf, _ = _test_accepted_baseline()
    parser_status = "with pypdf validation" if PdfReader is not None else "without pypdf"
    print(
        "PDF reporting passed: defective, zero-defect, multi-page, special-text, "
        f"immutability, and accepted baseline cases ({len(baseline_pdf):,} bytes, "
        f"{parser_status})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

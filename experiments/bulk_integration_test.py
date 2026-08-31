from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_paths import find_reference_image
from modules.inspection_pipeline import run_inspection_arrays


GUI_PATH = PROJECT_ROOT / "gui" / "interface.py"
BASELINE_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "images"
    / "Missing_hole"
    / "01_missing_hole_05.jpg"
)
RECTIFIED_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "rotation"
    / "Open_circuit_rotation"
    / "04_open_circuit_01.jpg"
)
OTHER_BOARD_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "images"
    / "Missing_hole"
    / "06_missing_hole_01.jpg"
)
DUPLICATE_SOURCE = (
    PROJECT_ROOT
    / "dataset"
    / "images"
    / "Missing_hole"
    / "01_missing_hole_01.jpg"
)


def _upload(file_id: str, path: Path, name: str = None) -> UploadedFile:
    return UploadedFile(
        UploadedFileRec(
            file_id,
            name or path.name,
            "image/png" if (name or path.name).lower().endswith(".png") else "image/jpeg",
            path.read_bytes(),
        ),
        None,
    )


def _bytes_upload(file_id: str, name: str, content: bytes) -> UploadedFile:
    return UploadedFile(
        UploadedFileRec(file_id, name, "image/jpeg", content),
        None,
    )


def _encoded_blank(width: int, height: int) -> bytes:
    blank = np.full((height, width, 3), 127, dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", blank)
    assert encoded_ok
    return encoded.tobytes()


def _run_bulk(uploads, timeout: int = 180):
    app = AppTest.from_file(str(GUI_PATH), default_timeout=timeout)
    app.session_state["inspection_mode"] = "Bulk Images"
    app.session_state["bulk_uploads"] = list(uploads)
    app.run(timeout=timeout)
    assert not app.exception, [str(item.value) for item in app.exception]
    run_button = next(
        button for button in app.button if "Run Bulk Inspection" in button.label
    )
    assert not run_button.disabled
    run_button.click().run(timeout=timeout)
    assert not app.exception, [str(item.value) for item in app.exception]
    return app, app.session_state["bulk_batch"]


def _without_runtime(value):
    copied = deepcopy(value)
    copied.pop("processing_time", None)
    copied.pop("timestamp", None)
    return copied


def _contains_array(value) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_array(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_array(item) for item in value)
    return False


def _assert_single_bulk_equivalence(app) -> None:
    reference_path = find_reference_image(BASELINE_IMAGE)
    assert reference_path is not None
    single_result = run_inspection_arrays(
        cv2.imread(str(BASELINE_IMAGE), cv2.IMREAD_COLOR),
        cv2.imread(str(reference_path), cv2.IMREAD_COLOR),
        BASELINE_IMAGE.name,
        reference_path.name,
    )
    bulk_result = app.session_state["bulk_detail_result"]

    assert single_result["calibration_metadata"] == bulk_result["calibration_metadata"]
    assert all(
        np.array_equal(single_result["test_stages"][stage], bulk_result["test_stages"][stage])
        for stage in single_result["test_stages"]
    )
    assert all(
        np.array_equal(single_result["template_stages"][stage], bulk_result["template_stages"][stage])
        for stage in single_result["template_stages"]
    )
    assert np.array_equal(
        single_result["seg_stages"]["morphology"],
        bulk_result["seg_stages"]["morphology"],
    )
    assert single_result["seg_metrics"]["defect_count"] == bulk_result["seg_metrics"]["defect_count"]
    assert single_result["seg_metrics"]["defect_area_px"] == bulk_result["seg_metrics"]["defect_area_px"]
    assert single_result["seg_metrics"]["defect_area_pct"] == bulk_result["seg_metrics"]["defect_area_pct"]
    assert single_result["defects"] == bulk_result["defects"]
    assert single_result["analysis_metrics"] == bulk_result["analysis_metrics"]
    assert _without_runtime(single_result["evaluation"]) == _without_runtime(
        bulk_result["evaluation"]
    )
    assert _without_runtime(single_result["inspection_report"]) == _without_runtime(
        bulk_result["inspection_report"]
    )


def main() -> int:
    baseline_app, baseline_batch = _run_bulk(
        [_upload("bulk-baseline", BASELINE_IMAGE)]
    )
    baseline_summary = baseline_batch["results"][0]
    assert baseline_summary["status"] == "SUCCESS"
    assert baseline_summary["reference_filename"] == "01.JPG"
    assert baseline_summary["calibration_status"] == "ALREADY_ALIGNED"
    assert baseline_summary["warp_applied"] is False
    assert baseline_summary["defect_count"] == 5
    assert baseline_summary["total_defect_area"] == 2579
    assert round(baseline_summary["coverage_percentage"], 4) == 0.0536
    assert baseline_summary["inspection_result"] == "DEFECTIVE"
    assert not _contains_array(baseline_batch)
    assert "pcb_result" not in baseline_app.session_state
    assert [tab.label for tab in baseline_app.tabs] == [
        "Image Pre-processing",
        "Defect Segmentation",
        "Feature Analysis",
        "Severity & Spatial Analysis",
        "Inspection Report",
    ]
    assert any(
        button.label == "Download Bulk Inspection Results (JSON)"
        for button in baseline_app.get("download_button")
    )
    json.dumps(
        {
            "schema_version": baseline_batch["schema_version"],
            "generated_at": baseline_batch["generated_at"],
            "batch_summary": baseline_batch["batch_summary"],
            "results": baseline_batch["results"],
        }
    )
    _assert_single_bulk_equivalence(baseline_app)

    mixed_app, mixed_batch = _run_bulk([
        _upload("mixed-identity", BASELINE_IMAGE),
        _upload("mixed-rectified", RECTIFIED_IMAGE),
        _upload("mixed-board-06", OTHER_BOARD_IMAGE),
    ])
    mixed_results = mixed_batch["results"]
    assert [result["reference_filename"] for result in mixed_results] == [
        "01.JPG",
        "04.JPG",
        "06.JPG",
    ]
    assert [result["status"] for result in mixed_results] == [
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    ]
    assert mixed_results[0]["calibration_status"] == "ALREADY_ALIGNED"
    assert mixed_results[0]["warp_applied"] is False
    assert mixed_results[1]["calibration_status"] == "RECTIFIED"
    assert mixed_results[1]["warp_applied"] is True
    batch_signature_before = mixed_batch["signature"]
    result_selector = next(
        selector
        for selector in mixed_app.selectbox
        if selector.label == "Select Inspection Result"
    )
    result_selector.select(mixed_results[1]["item_id"]).run(timeout=180)
    assert not mixed_app.exception, [str(item.value) for item in mixed_app.exception]
    assert mixed_app.session_state["bulk_detail_result"]["bulk_item_id"] == mixed_results[1]["item_id"]
    assert mixed_app.session_state["bulk_detail_result"]["calibration_metadata"]["status"] == "RECTIFIED"
    assert mixed_app.session_state["bulk_batch"]["signature"] == batch_signature_before

    failure_app, failure_batch = _run_bulk([
        _upload("failure-valid-first", BASELINE_IMAGE),
        _bytes_upload("failure-corrupt", "01_corrupt.jpg", b"not an image"),
        _upload("failure-unresolved", BASELINE_IMAGE, name="99_unknown.jpg"),
        _upload("failure-unsupported", BASELINE_IMAGE, name="01_unsupported.bmp"),
        _bytes_upload(
            "failure-calibration-failed",
            "01_bad_geometry.png",
            _encoded_blank(100, 100),
        ),
        _bytes_upload(
            "failure-calibration-unverified",
            "01_blank.png",
            _encoded_blank(3034, 1586),
        ),
        _upload("failure-valid-last", OTHER_BOARD_IMAGE),
    ])
    failure_statuses = [result["status"] for result in failure_batch["results"]]
    assert failure_statuses == [
        "SUCCESS",
        "DECODE_ERROR",
        "REFERENCE_UNRESOLVED",
        "UNSUPPORTED_FORMAT",
        "CALIBRATION_FAILED",
        "CALIBRATION_UNVERIFIED",
        "SUCCESS",
    ]
    assert failure_batch["batch_summary"]["errors"] == 5
    assert failure_batch["results"][-1]["reference_filename"] == "06.JPG"
    assert not failure_app.exception

    _, duplicate_batch = _run_bulk([
        _upload("duplicate-first", BASELINE_IMAGE, name="01_duplicate.jpg"),
        _upload("duplicate-second", DUPLICATE_SOURCE, name="01_duplicate.jpg"),
    ])
    duplicate_results = duplicate_batch["results"]
    assert [result["filename"] for result in duplicate_results] == [
        "01_duplicate.jpg",
        "01_duplicate.jpg",
    ]
    assert duplicate_results[0]["item_id"] != duplicate_results[1]["item_id"]
    assert all(result["status"] == "SUCCESS" for result in duplicate_results)

    limit_app = AppTest.from_file(str(GUI_PATH), default_timeout=45)
    limit_app.session_state["inspection_mode"] = "Bulk Images"
    limit_app.session_state["bulk_uploads"] = [
        _bytes_upload(f"limit-{index}", f"01_limit_{index}.jpg", b"invalid")
        for index in range(11)
    ]
    limit_app.run(timeout=45)
    limit_button = next(
        button for button in limit_app.button if "Run Bulk Inspection" in button.label
    )
    assert limit_button.disabled
    assert "bulk_batch" not in limit_app.session_state
    assert any("maximum of 10 images" in str(error.value) for error in limit_app.error)

    print(
        "Bulk integration passed: baseline equivalence, 3-board mixed batch, "
        "failure isolation, duplicate names, array-free summaries, and 10-image limit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

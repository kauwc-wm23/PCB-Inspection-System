from time import perf_counter
from typing import Any, Dict

import numpy as np

from modules.calibration import calibrate_to_reference
from modules.feature_analysis import analyse_features
from modules.inspection_evaluation import assess_defect_severity
from modules.preprocessing import get_preprocessing_stages_from_array
from modules.reporting import generate_inspection_summary
from modules.segmentation import get_segmentation_stages


class CalibrationPipelineError(ValueError):
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = dict(metadata)
        self.calibration_status = str(metadata.get("status", "UNVERIFIED"))
        message = str(metadata.get("message", "Calibration could not be verified."))
        super().__init__(f"Calibration {self.calibration_status}: {message}")


def run_inspection_arrays(
    test_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    test_name: str,
    reference_name: str,
) -> Dict[str, Any]:
    started = perf_counter()

    calibration_result = calibrate_to_reference(test_bgr, reference_bgr)
    calibration_metadata = calibration_result.metadata
    if not calibration_result.is_verified:
        raise CalibrationPipelineError(calibration_metadata)

    test_stages, test_metrics = get_preprocessing_stages_from_array(
        calibration_result.calibrated_test
    )
    template_stages, template_metrics = get_preprocessing_stages_from_array(
        reference_bgr
    )
    processed_test = test_stages["enhanced"]
    processed_template = template_stages["enhanced"]

    seg_stages, seg_metrics = get_segmentation_stages(
        processed_test,
        processed_template,
    )
    defects, analysis_metrics = analyse_features(
        seg_stages["morphology"],
        seg_metrics["contours"],
    )
    evaluation = assess_defect_severity(
        defects,
        seg_stages["morphology"].shape,
    )

    processing_time = perf_counter() - started
    evaluation["processing_time"] = processing_time
    inspection_report = generate_inspection_summary(
        test_filename=test_name,
        template_filename=reference_name,
        processing_time=processing_time,
        defects=defects,
        analysis_metrics=analysis_metrics,
        evaluation_result=evaluation,
    )

    return {
        "test_stages": test_stages,
        "test_metrics": test_metrics,
        "template_stages": template_stages,
        "template_metrics": template_metrics,
        "calibration_metadata": calibration_metadata,
        "calibration_valid_region_mask": calibration_result.valid_region_mask,
        "processed_test": processed_test,
        "processed_template": processed_template,
        "seg_stages": seg_stages,
        "seg_metrics": seg_metrics,
        "defects": defects,
        "analysis_metrics": analysis_metrics,
        "evaluation": evaluation,
        "inspection_report": inspection_report,
        "proc_time": processing_time,
        "test_name": test_name,
        "template_name": reference_name,
    }


def create_compact_inspection_result(
    detailed_result: Dict[str, Any],
    item_id: str,
) -> Dict[str, Any]:
    evaluation = detailed_result["evaluation"]
    calibration = detailed_result["calibration_metadata"]
    quality_decision = str(evaluation.get("quality_decision", "FAIL"))

    return {
        "item_id": item_id,
        "filename": detailed_result["test_name"],
        "reference_filename": detailed_result["template_name"],
        "status": "SUCCESS",
        "error_message": "",
        "calibration_status": calibration.get("status", "N/A"),
        "warp_applied": bool(calibration.get("warp_applied", False)),
        "defect_count": int(evaluation.get("total_defect_count", 0)),
        "total_defect_area": int(evaluation.get("total_defect_area", 0)),
        "coverage_percentage": float(
            evaluation.get("defect_coverage_percentage", 0.0)
        ),
        "highest_severity": evaluation.get("highest_severity_level", "N/A"),
        "highest_priority_defect": evaluation.get(
            "highest_priority_defect_id", "N/A"
        ),
        "highest_priority_region": evaluation.get(
            "highest_priority_defect_region", "N/A"
        ),
        "dominant_spatial_region": evaluation.get(
            "most_concentrated_region", "N/A"
        ),
        "inspection_result": (
            "PASS" if quality_decision == "PASS" else "DEFECTIVE"
        ),
        "processing_time": float(detailed_result.get("proc_time", 0.0)),
        "compact_report": detailed_result["inspection_report"],
    }


def create_compact_error_result(
    item_id: str,
    filename: str,
    status: str,
    error_message: str,
    reference_filename: str = "N/A",
    calibration_status: str = "N/A",
) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "filename": filename,
        "reference_filename": reference_filename,
        "status": status,
        "error_message": error_message,
        "calibration_status": calibration_status,
        "warp_applied": False,
        "defect_count": 0,
        "total_defect_area": 0,
        "coverage_percentage": 0.0,
        "highest_severity": "N/A",
        "highest_priority_defect": "N/A",
        "highest_priority_region": "N/A",
        "dominant_spatial_region": "N/A",
        "inspection_result": "ERROR",
        "processing_time": 0.0,
        "compact_report": None,
    }

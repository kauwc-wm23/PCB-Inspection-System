import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.calibration import calibrate_to_reference
from modules.dataset_paths import discover_dataset_images, find_reference_image
from modules.feature_analysis import analyse_features
from modules.inspection_evaluation import assess_defect_severity
from modules.preprocessing import load_image, preprocess_image_array
from modules.reporting import generate_inspection_summary
from modules.segmentation import get_segmentation_stages


DEFAULT_TEST_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "images"
    / "Missing_hole"
    / "01_missing_hole_05.jpg"
)
DEFAULT_REFERENCE_IMAGE = PROJECT_ROOT / "dataset" / "PCB_USED" / "01.JPG"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "experiments" / "calibration_evaluation.json"
)


def _synthetic_transform(
    image: np.ndarray,
    rotation_deg: float = 0.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        rotation_deg,
        scale,
    )
    matrix[:, 2] += (translation_x, translation_y)
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return transformed, matrix


def _inverse_affine(matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.vstack((matrix, (0.0, 0.0, 1.0)))
    return np.linalg.inv(homogeneous)[:2]


def _properties(
    matrix: np.ndarray,
    source_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
) -> Dict[str, float]:
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    source_centre = np.float32([[[source_width / 2.0, source_height / 2.0]]])
    mapped = cv2.transform(source_centre, matrix)[0, 0]
    return {
        "scale": scale,
        "rotation_deg": rotation,
        "translation_x": float(mapped[0] - target_width / 2.0),
        "translation_y": float(mapped[1] - target_height / 2.0),
    }


def _edge_residual(test_image: np.ndarray, reference_image: np.ndarray) -> float:
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    test_edges = cv2.Canny(test_gray, 40, 120)
    reference_edges = cv2.Canny(reference_gray, 40, 120)
    return float(np.mean(cv2.absdiff(test_edges, reference_edges)))


def _segmentation_metrics(test_image: np.ndarray, reference_image: np.ndarray) -> Dict:
    processed_test = preprocess_image_array(test_image)
    processed_reference = preprocess_image_array(reference_image)
    stages, metrics = get_segmentation_stages(processed_test, processed_reference)
    return {
        "defect_count": int(metrics["defect_count"]),
        "defect_area_px": int(metrics["defect_area_px"]),
        "defect_area_pct": float(metrics["defect_area_pct"]),
        "mask": stages["morphology"],
        "contours": metrics["contours"],
    }


def _evaluate_case(
    name: str,
    transformed_test: np.ndarray,
    reference_image: np.ndarray,
    applied_matrix: Optional[np.ndarray],
) -> Dict[str, Any]:
    result = calibrate_to_reference(transformed_test, reference_image)
    metadata = result.metadata
    row: Dict[str, Any] = {
        "case": name,
        "status": metadata["status"],
        "warp_applied": metadata["warp_applied"],
        "match_count": metadata["match_count"],
        "inlier_count": metadata["inlier_count"],
        "inlier_ratio": metadata["inlier_ratio"],
        "reprojection_error_px": metadata["reprojection_error"],
        "estimated_scale": metadata["scale"],
        "estimated_rotation_deg": metadata["rotation_deg"],
        "estimated_translation_x_px": metadata["translation_x"],
        "estimated_translation_y_px": metadata["translation_y"],
        "overlap_ratio": metadata["overlap_ratio"],
        "message": metadata["message"],
        "edge_residual_before": _edge_residual(transformed_test, reference_image),
        "edge_residual_after": None,
    }
    if result.calibrated_test is not None:
        row["edge_residual_after"] = _edge_residual(
            result.calibrated_test,
            reference_image,
        )
    if applied_matrix is not None and metadata["scale"] is not None:
        expected = _properties(
            _inverse_affine(applied_matrix),
            transformed_test.shape,
            reference_image.shape,
        )
        row.update(
            {
                "expected_scale": expected["scale"],
                "expected_rotation_deg": expected["rotation_deg"],
                "expected_translation_x_px": expected["translation_x"],
                "expected_translation_y_px": expected["translation_y"],
                "scale_error": abs(metadata["scale"] - expected["scale"]),
                "rotation_error_deg": abs(
                    metadata["rotation_deg"] - expected["rotation_deg"]
                ),
                "translation_error_px": float(
                    np.hypot(
                        metadata["translation_x"] - expected["translation_x"],
                        metadata["translation_y"] - expected["translation_y"],
                    )
                ),
            }
        )
    return row


def _identity_downstream_regression(
    test_image: np.ndarray,
    reference_image: np.ndarray,
) -> Dict[str, Any]:
    calibration = calibrate_to_reference(test_image, reference_image)
    if calibration.metadata["status"] != "ALREADY_ALIGNED":
        raise AssertionError("The accepted production example did not use identity alignment.")
    if calibration.calibrated_test is not test_image:
        raise AssertionError("Identity calibration did not preserve the original array object.")

    baseline_test = preprocess_image_array(test_image)
    calibrated_test = preprocess_image_array(calibration.calibrated_test)
    processed_reference = preprocess_image_array(reference_image)
    baseline_stages, baseline_metrics = get_segmentation_stages(
        baseline_test, processed_reference
    )
    calibrated_stages, calibrated_metrics = get_segmentation_stages(
        calibrated_test, processed_reference
    )
    if not np.array_equal(
        baseline_stages["morphology"], calibrated_stages["morphology"]
    ):
        raise AssertionError("Identity calibration changed the segmentation mask.")

    baseline_defects, baseline_analysis = analyse_features(
        baseline_stages["morphology"], baseline_metrics["contours"]
    )
    calibrated_defects, calibrated_analysis = analyse_features(
        calibrated_stages["morphology"], calibrated_metrics["contours"]
    )
    baseline_evaluation = assess_defect_severity(
        baseline_defects, baseline_stages["morphology"].shape
    )
    calibrated_evaluation = assess_defect_severity(
        calibrated_defects, calibrated_stages["morphology"].shape
    )
    baseline_report = generate_inspection_summary(
        "01_missing_hole_05.jpg",
        "01.JPG",
        1.0,
        baseline_defects,
        baseline_analysis,
        baseline_evaluation,
    )
    calibrated_report = generate_inspection_summary(
        "01_missing_hole_05.jpg",
        "01.JPG",
        1.0,
        calibrated_defects,
        calibrated_analysis,
        calibrated_evaluation,
    )
    baseline_report.pop("timestamp", None)
    calibrated_report.pop("timestamp", None)
    checks = {
        "preprocessed_image_equal": np.array_equal(baseline_test, calibrated_test),
        "segmentation_mask_equal": np.array_equal(
            baseline_stages["morphology"], calibrated_stages["morphology"]
        ),
        "contour_count_equal": len(baseline_metrics["contours"])
        == len(calibrated_metrics["contours"]),
        "feature_values_equal": baseline_defects == calibrated_defects,
        "analysis_metrics_equal": baseline_analysis == calibrated_analysis,
        "severity_values_equal": baseline_evaluation == calibrated_evaluation,
        "report_equal": baseline_report == calibrated_report,
    }
    if not all(checks.values()):
        raise AssertionError(f"Identity downstream regression failed: {checks}")
    checks["defect_count"] = len(baseline_defects)
    return checks


def _production_regression() -> Dict[str, Any]:
    started = time.perf_counter()
    counts = {
        "total_images": 0,
        "identity_cases": 0,
        "warped_cases": 0,
        "rejected_cases": 0,
        "identity_pixel_changes": 0,
        "identity_segmentation_changes": 0,
        "baseline_segmentation_changes": 0,
        "warped_registration_improvements": 0,
        "warped_details": [],
        "rejected_details": [],
    }
    references: Dict[Path, np.ndarray] = {}
    for index, image_path in enumerate(discover_dataset_images(), start=1):
        reference_path = find_reference_image(image_path)
        if reference_path is None:
            counts["rejected_cases"] += 1
            continue
        test_image = load_image(image_path)
        if reference_path not in references:
            references[reference_path] = load_image(reference_path)
        reference_image = references[reference_path]
        result = calibrate_to_reference(test_image, reference_image)
        counts["total_images"] += 1
        if result.metadata["status"] == "ALREADY_ALIGNED":
            counts["identity_cases"] += 1
            if not np.array_equal(result.calibrated_test, test_image):
                counts["identity_pixel_changes"] += 1
                counts["identity_segmentation_changes"] += 1
                counts["baseline_segmentation_changes"] += 1
        elif result.metadata["status"] == "RECTIFIED":
            counts["warped_cases"] += 1
            counts["warped_details"].append(
                {
                    "image": str(image_path.relative_to(PROJECT_ROOT)),
                    "scale": result.metadata["scale"],
                    "rotation_deg": result.metadata["rotation_deg"],
                    "translation_x": result.metadata["translation_x"],
                    "translation_y": result.metadata["translation_y"],
                }
            )
            print(
                f"Unexpected production warp: {image_path.name} "
                f"{counts['warped_details'][-1]}",
                flush=True,
            )
            baseline = _segmentation_metrics(test_image, reference_image)
            calibrated = _segmentation_metrics(
                result.calibrated_test, reference_image
            )
            if (
                baseline["defect_count"] != calibrated["defect_count"]
                or not np.array_equal(baseline["mask"], calibrated["mask"])
            ):
                counts["baseline_segmentation_changes"] += 1
            if calibrated["defect_area_pct"] < baseline["defect_area_pct"]:
                counts["warped_registration_improvements"] += 1
            counts["warped_details"][-1].update(
                {
                    "baseline_defect_count": baseline["defect_count"],
                    "calibrated_defect_count": calibrated["defect_count"],
                    "baseline_defect_area_pct": baseline["defect_area_pct"],
                    "calibrated_defect_area_pct": calibrated["defect_area_pct"],
                }
            )
        else:
            counts["rejected_cases"] += 1
            counts["rejected_details"].append(
                {
                    "image": str(image_path.relative_to(PROJECT_ROOT)),
                    "status": result.metadata["status"],
                    "message": result.metadata["message"],
                }
            )
            print(
                f"Unexpected production rejection: {image_path.name} "
                f"{result.metadata['message']}",
                flush=True,
            )
        if index % 100 == 0:
            print(
                f"Production calibration regression: {index} images checked",
                flush=True,
            )
    counts["elapsed_seconds"] = time.perf_counter() - started
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded reference-relative PCB image registration."
    )
    parser.add_argument("--test-image", type=Path, default=DEFAULT_TEST_IMAGE)
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--production-regression", action="store_true")
    args = parser.parse_args(argv)

    test_image = load_image(args.test_image)
    reference_image = load_image(args.reference_image)
    cases = []
    identity_matrix = np.float64([[1, 0, 0], [0, 1, 0]])
    cases.append(
        _evaluate_case(
            "already_aligned", test_image, reference_image, identity_matrix
        )
    )
    synthetic_cases = (
        ("small_translation", 0.0, 4.0, -3.0, 1.0),
        ("small_rotation", 1.0, 0.0, 0.0, 1.0),
        ("combined_small_transform", 1.0, 5.0, -4.0, 1.01),
        ("excessive_rotation", 8.0, 0.0, 0.0, 1.0),
    )
    transformed_images = {}
    for name, rotation, translation_x, translation_y, scale in synthetic_cases:
        transformed, matrix = _synthetic_transform(
            test_image,
            rotation,
            translation_x,
            translation_y,
            scale,
        )
        transformed_images[name] = transformed
        cases.append(_evaluate_case(name, transformed, reference_image, matrix))

    blank = np.full((480, 640, 3), 127, dtype=np.uint8)
    cases.append(_evaluate_case("insufficient_features", blank, blank.copy(), None))

    expected_statuses = {
        "already_aligned": "ALREADY_ALIGNED",
        "small_translation": "RECTIFIED",
        "small_rotation": "RECTIFIED",
        "combined_small_transform": "RECTIFIED",
        "excessive_rotation": "UNVERIFIED",
        "insufficient_features": "UNVERIFIED",
    }
    for row in cases:
        if row["status"] != expected_statuses[row["case"]]:
            raise AssertionError(
                f"Unexpected status for {row['case']}: {row['status']}"
            )
        if row["status"] == "RECTIFIED":
            if row["edge_residual_after"] >= row["edge_residual_before"]:
                raise AssertionError(
                    f"Registration did not reduce residual for {row['case']}"
                )
            if row["rotation_error_deg"] > 0.10:
                raise AssertionError(f"Rotation recovery failed for {row['case']}")
            if row["scale_error"] > 0.002:
                raise AssertionError(f"Scale recovery failed for {row['case']}")
            if row["translation_error_px"] > 1.5:
                raise AssertionError(f"Translation recovery failed for {row['case']}")

    downstream = _identity_downstream_regression(test_image, reference_image)
    segmentation_recovery = {}
    for name in ("small_translation", "small_rotation", "combined_small_transform"):
        before = _segmentation_metrics(transformed_images[name], reference_image)
        calibration = calibrate_to_reference(
            transformed_images[name], reference_image
        )
        after = _segmentation_metrics(calibration.calibrated_test, reference_image)
        segmentation_recovery[name] = {
            "before_defect_count": before["defect_count"],
            "after_defect_count": after["defect_count"],
            "before_defect_area_pct": before["defect_area_pct"],
            "after_defect_area_pct": after["defect_area_pct"],
        }

    output: Dict[str, Any] = {
        "definition": (
            "Reference-relative geometric registration; no physical-unit calibration."
        ),
        "cases": cases,
        "identity_downstream_regression": downstream,
        "segmentation_recovery": segmentation_recovery,
        "production_regression": (
            _production_regression() if args.production_regression else None
        ),
    }
    if output["production_regression"] is not None:
        regression = output["production_regression"]
        print(json.dumps({"production_regression": regression}, indent=2))
        if regression["identity_pixel_changes"] != 0:
            raise AssertionError("Identity calibration changed production pixels.")
        if regression["identity_segmentation_changes"] != 0:
            raise AssertionError("Calibration changed aligned production segmentation.")
        if regression["warped_registration_improvements"] != regression["warped_cases"]:
            raise AssertionError("A production warp did not improve registration residuals.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

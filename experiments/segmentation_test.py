"""Smoke test for Modules 1, 2, 3, and 5 on one real dataset image."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_paths import discover_dataset_images, find_reference_image
from modules.feature_analysis import analyse_features
from modules.inspection_evaluation import assess_defect_severity
from modules.preprocessing import preprocess_image
from modules.segmentation import get_segmentation_stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Missing_hole")
    args = parser.parse_args()

    images = discover_dataset_images(args.category)
    if not images:
        raise FileNotFoundError(f"No dataset images found for category: {args.category}")
    image_path = images[0]
    reference_path = find_reference_image(image_path)
    if reference_path is None:
        raise FileNotFoundError(f"No PCB_USED reference found for: {image_path.name}")

    processed_test = preprocess_image(str(image_path))
    processed_reference = preprocess_image(str(reference_path))
    stages, segmentation_metrics = get_segmentation_stages(processed_test, processed_reference)
    defects, analysis_metrics = analyse_features(
        stages["morphology"], segmentation_metrics["contours"]
    )
    evaluation = assess_defect_severity(defects, stages["morphology"].shape)

    assert processed_test.ndim == 2 and processed_test.dtype == np.uint8
    assert processed_test.shape == processed_reference.shape
    assert set(np.unique(stages["morphology"])).issubset({0, 255})
    assert analysis_metrics["total_defects"] == len(defects)
    assert evaluation["total_defect_count"] == len(defects)
    assert all("bounding_box" in defect and "centroid" in defect for defect in defects)
    assert sum(evaluation["spatial_distribution"].values()) == len(defects)
    assert sorted(defect["priority_rank"] for defect in evaluation["defects"]) == list(
        range(1, len(defects) + 1)
    )
    for defect in evaluation["defects"]:
        assert defect["area_ratio"] >= 0.0
        assert defect["width_ratio"] >= 0.0
        assert defect["height_ratio"] >= 0.0
        assert defect["severity_score"] >= 0.0
        assert defect["severity_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert defect["spatial_region"] in {
            "TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"
        }

    output_dir = PROJECT_ROOT / "outputs" / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"test_{image_path.stem}.png"
    if not cv2.imwrite(str(output_path), stages["overlay"]):
        raise OSError(f"Unable to write test output: {output_path}")

    print(f"Image: {image_path}")
    print(f"Reference: {reference_path}")
    print(f"Otsu threshold: {segmentation_metrics['threshold']}")
    print(f"Defects: {evaluation['total_defect_count']}")
    print(f"Coverage: {evaluation['defect_coverage_percentage']:.4f}%")
    print(
        f"Status: {evaluation['status_label']} | "
        f"Highest severity: {evaluation['highest_severity_level']} | "
        f"Priority defect: {evaluation['highest_priority_defect_id']} | "
        f"Concentrated region: {evaluation['most_concentrated_region']}"
    )
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

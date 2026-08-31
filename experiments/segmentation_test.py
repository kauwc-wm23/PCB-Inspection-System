
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
from modules.segmentation import (
    CLOSING_ITERATIONS,
    KERNEL_CLOSE_SIZE,
    KERNEL_OPEN_SIZE,
    OPENING_ITERATIONS,
    analyse_morphology_effects,
    count_foreground_components,
    extract_change_detail_roi,
    get_segmentation_stages,
)


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
    assert all(
        key in stages
        for key in (
            "otsu_binary",
            "opening",
            "morphology",
            "opening_removed",
            "closing_added",
        )
    )
    assert np.array_equal(
        stages["opening_removed"],
        np.where(
            (stages["otsu_binary"] > 0) & ~(stages["opening"] > 0),
            255,
            0,
        ).astype(np.uint8),
    )
    assert np.array_equal(
        stages["closing_added"],
        np.where(
            (stages["morphology"] > 0) & ~(stages["opening"] > 0),
            255,
            0,
        ).astype(np.uint8),
    )

    morphology_analysis = segmentation_metrics["morphology_analysis"]
    opening_metrics = morphology_analysis["opening"]
    closing_metrics = morphology_analysis["closing"]
    assert opening_metrics["pixel_change"] == cv2.countNonZero(
        stages["opening_removed"]
    )
    assert closing_metrics["pixel_change"] == cv2.countNonZero(
        stages["closing_added"]
    )
    assert np.isclose(
        opening_metrics["change_percentage"],
        (
            opening_metrics["pixel_change"]
            / opening_metrics["foreground_before"]
            * 100.0
            if opening_metrics["foreground_before"]
            else 0.0
        ),
    )
    assert np.isclose(
        closing_metrics["change_percentage"],
        (
            closing_metrics["pixel_change"]
            / closing_metrics["foreground_before"]
            * 100.0
            if closing_metrics["foreground_before"]
            else 0.0
        ),
    )
    assert opening_metrics["components_before"] == count_foreground_components(
        stages["otsu_binary"]
    )
    assert opening_metrics["components_after"] == count_foreground_components(
        stages["opening"]
    )
    assert closing_metrics["components_before"] == opening_metrics["components_after"]
    assert closing_metrics["components_after"] == count_foreground_components(
        stages["morphology"]
    )

    for change_map in (stages["opening_removed"], stages["closing_added"]):
        detail_roi, detail_bounds = extract_change_detail_roi(change_map)
        if cv2.countNonZero(change_map) == 0:
            assert detail_roi is None and detail_bounds is None
        else:
            assert detail_roi is not None and detail_bounds is not None
            x_start, y_start, x_end, y_end = detail_bounds
            image_height, image_width = change_map.shape
            assert 0 <= x_start < x_end <= image_width
            assert 0 <= y_start < y_end <= image_height
            assert cv2.countNonZero(detail_roi) == cv2.countNonZero(change_map)

    synthetic_pre = np.zeros((9, 9), dtype=np.uint8)
    synthetic_pre[1, 1] = 255
    synthetic_pre[4:7, 4:7] = 255
    synthetic_opened = synthetic_pre.copy()
    synthetic_opened[1, 1] = 0
    synthetic_closed = synthetic_opened.copy()
    synthetic_closed[3, 4] = 255
    synthetic_maps, synthetic_metrics = analyse_morphology_effects(
        synthetic_pre,
        synthetic_opened,
        synthetic_closed,
    )
    assert cv2.countNonZero(synthetic_maps["opening_removed"]) == 1
    assert cv2.countNonZero(synthetic_maps["closing_added"]) == 1
    assert synthetic_metrics["opening"]["foreground_before"] == 10
    assert synthetic_metrics["opening"]["foreground_after"] == 9
    assert np.isclose(synthetic_metrics["opening"]["change_percentage"], 10.0)
    assert synthetic_metrics["opening"]["components_before"] == 2
    assert synthetic_metrics["opening"]["components_after"] == 1
    assert synthetic_metrics["closing"]["foreground_before"] == 9
    assert synthetic_metrics["closing"]["foreground_after"] == 10
    assert np.isclose(
        synthetic_metrics["closing"]["change_percentage"],
        100.0 / 9.0,
    )

    zero_map = np.zeros((9, 9), dtype=np.uint8)
    assert extract_change_detail_roi(zero_map) == (None, None)
    _, zero_metrics = analyse_morphology_effects(zero_map, zero_map, zero_map)
    assert zero_metrics["opening"]["change_percentage"] == 0.0
    assert zero_metrics["closing"]["change_percentage"] == 0.0
    edge_change_map = np.zeros((9, 9), dtype=np.uint8)
    edge_change_map[0, 0] = 255
    edge_roi, edge_bounds = extract_change_detail_roi(edge_change_map)
    assert edge_roi is not None and edge_bounds is not None
    assert edge_bounds[0] == 0 and edge_bounds[1] == 0
    assert edge_bounds[2] <= 9 and edge_bounds[3] <= 9

    boolean_pre = synthetic_pre.astype(bool)
    boolean_opened = synthetic_opened.astype(bool)
    boolean_closed = synthetic_closed.astype(bool)
    boolean_maps, boolean_metrics = analyse_morphology_effects(
        boolean_pre,
        boolean_opened,
        boolean_closed,
    )
    assert boolean_maps["opening_removed"].dtype == np.uint8
    assert boolean_maps["closing_added"].dtype == np.uint8
    assert boolean_metrics == synthetic_metrics
    boolean_detail, boolean_bounds = extract_change_detail_roi(
        boolean_maps["opening_removed"].astype(bool)
    )
    assert boolean_detail is not None and boolean_bounds is not None
    assert cv2.countNonZero(boolean_detail) == 1

    morphology_configuration = segmentation_metrics["morphology_configuration"]
    assert morphology_configuration["opening"]["kernel_size"] == (
        KERNEL_OPEN_SIZE,
        KERNEL_OPEN_SIZE,
    )
    assert morphology_configuration["opening"]["iterations"] == OPENING_ITERATIONS
    assert morphology_configuration["closing"]["kernel_size"] == (
        KERNEL_CLOSE_SIZE,
        KERNEL_CLOSE_SIZE,
    )
    assert morphology_configuration["closing"]["iterations"] == CLOSING_ITERATIONS
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

"""Streamlit interaction smoke test for the full PCB inspection pipeline."""

from pathlib import Path
import re
import sys

import cv2
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.preprocessing_presentation import (
    create_filtering_change_map,
    extract_matching_center_rois,
)
import gui.preprocessing_presentation as preprocessing_presentation_module
from modules.dataset_paths import discover_dataset_images, find_reference_image
from modules.preprocessing import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    apply_clahe,
    apply_median_filter,
    convert_grayscale,
    get_preprocessing_stages,
    preprocess_image,
)


GUI_PATH = PROJECT_ROOT / "gui" / "interface.py"
_MISSING_HOLE_IMAGES = discover_dataset_images("Missing_hole")
if not _MISSING_HOLE_IMAGES:
    raise FileNotFoundError("No Missing_hole images are available for GUI integration testing.")
TEST_IMAGE = _MISSING_HOLE_IMAGES[0]
TEST_REFERENCE = find_reference_image(TEST_IMAGE)
if TEST_REFERENCE is None:
    raise FileNotFoundError(f"No reference image matches {TEST_IMAGE.name}.")


def main() -> int:
    stages, metrics = get_preprocessing_stages(str(TEST_IMAGE))
    original = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_COLOR)
    expected_grayscale = convert_grayscale(original)
    expected_filtered = apply_median_filter(expected_grayscale)
    expected_enhanced = apply_clahe(expected_filtered)
    assert np.array_equal(
        expected_enhanced,
        apply_clahe(
            expected_filtered,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
        ),
    )
    for invalid_arguments in (
        {"clip_limit": 0},
        {"clip_limit": float("inf")},
        {"tile_grid_size": (0, 8)},
        {"tile_grid_size": (8,)},
    ):
        try:
            apply_clahe(expected_filtered, **invalid_arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for CLAHE arguments {invalid_arguments!r}"
            )

    assert stages["original"].ndim == 3 and stages["original"].shape[2] == 3
    assert np.array_equal(stages["original"], original)
    assert np.array_equal(stages["grayscale"], expected_grayscale)
    assert np.array_equal(stages["filtered"], expected_filtered)
    assert np.array_equal(stages["enhanced"], expected_enhanced)
    assert np.array_equal(preprocess_image(str(TEST_IMAGE)), stages["enhanced"])
    assert metrics["processing_time_seconds"] >= 0.0

    for stage_name in ("grayscale", "filtered", "enhanced"):
        image = stages[stage_name]
        high_frequency_estimate = round(
            float(cv2.Laplacian(image, cv2.CV_64F).var()), 2
        )
        expected_metrics = {
            "mean_brightness": round(float(np.mean(image)), 2),
            "contrast": round(float(np.std(image)), 2),
            "high_frequency_estimate": high_frequency_estimate,
            "noise_estimate": high_frequency_estimate,
            "dynamic_range": int(image.max()) - int(image.min()),
        }
        assert metrics[stage_name] == expected_metrics
        assert (
            metrics[stage_name]["high_frequency_estimate"]
            == metrics[stage_name]["noise_estimate"]
        )
    assert metrics["high_frequency_reduction"] == metrics["noise_reduction"]

    grayscale_before_map = stages["grayscale"].copy()
    filtered_before_map = stages["filtered"].copy()
    enhanced_before_map = stages["enhanced"].copy()
    filtering_change_raw, filtering_change_display = create_filtering_change_map(
        stages["grayscale"], stages["filtered"]
    )
    assert np.array_equal(
        filtering_change_raw,
        cv2.absdiff(stages["grayscale"], stages["filtered"]),
    )
    assert filtering_change_display.dtype == np.uint8
    if filtering_change_raw.max() > filtering_change_raw.min():
        assert filtering_change_display.min() == 0
        assert filtering_change_display.max() == 255
    assert np.array_equal(stages["grayscale"], grayscale_before_map)
    assert np.array_equal(stages["filtered"], filtered_before_map)
    assert np.array_equal(stages["enhanced"], enhanced_before_map)

    roi_views, bounds = extract_matching_center_rois(
        {
            "grayscale": stages["grayscale"],
            "filtered": stages["filtered"],
            "change_map": filtering_change_display,
            "enhanced": stages["enhanced"],
        }
    )
    x_start, y_start, x_end, y_end = bounds
    expected_roi_shape = (y_end - y_start, x_end - x_start)
    assert all(roi.shape[:2] == expected_roi_shape for roi in roi_views.values())
    assert np.array_equal(
        roi_views["enhanced"], stages["enhanced"][y_start:y_end, x_start:x_end]
    )
    for invalid_fraction in (0, 1.1, "0.3"):
        try:
            extract_matching_center_rois({"image": stages["enhanced"]}, invalid_fraction)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for ROI fraction {invalid_fraction!r}")

    # Regression: simulate Streamlit retaining the previous helper module
    # while rerunning the newly updated interface.
    delattr(preprocessing_presentation_module, "create_filtering_change_map")

    upload = UploadedFile(
        UploadedFileRec(
            "pcb-gui-test",
            TEST_IMAGE.name,
            "image/jpeg",
            TEST_IMAGE.read_bytes(),
        ),
        None,
    )
    app = AppTest.from_file(str(GUI_PATH), default_timeout=45)
    app.session_state["defective_upload"] = upload
    app.run(timeout=45)
    assert not app.exception, [str(item.value) for item in app.exception]
    assert hasattr(preprocessing_presentation_module, "create_filtering_change_map")

    run_button = next(
        button for button in app.button if "Run Inspection" in button.label
    )
    run_button.click().run(timeout=45)
    assert not app.exception, [str(item.value) for item in app.exception]

    result = app.session_state["pcb_result"]
    evaluation = result["evaluation"]
    assert result["template_name"] == TEST_REFERENCE.name
    measured_preprocessing_time = result["test_metrics"]["processing_time_seconds"]
    assert measured_preprocessing_time > 0.0
    assert evaluation["total_defects"] > 0
    assert all("severity_score" in defect for defect in evaluation["defects"])
    assert all("priority_rank" in defect for defect in evaluation["defects"])
    assert all("spatial_region" in defect for defect in evaluation["defects"])
    assert [tab.label for tab in app.tabs] == [
        "🖼️ Image Pre-processing",
        "🔍 Defect Segmentation",
        "📊 Feature Analysis",
        "✅ Severity & Spatial Analysis",
        "📄 Inspection Report",
    ]
    assert len(app.dataframe) >= 2

    visible_markdown = "\n".join(str(item.value) for item in app.markdown)
    for expected_label in (
        "Image Pre-processing Pipeline",
        "Image Quality Improvement",
        "Local Processing Comparison",
        "A. Median Filtering Effect",
        "FILTERING CHANGE MAP",
        "B. Contrast Enhancement",
        "Pixel Intensity Distribution",
        "Original vs Preprocessed Output",
        "Pre-processing Summary",
    ):
        assert expected_label in visible_markdown, expected_label
    assert f"{measured_preprocessing_time:.3f}s" in visible_markdown
    visible_captions = "\n".join(str(item.value) for item in app.caption)
    assert (
        "should not be interpreted directly as noise"
        in f"{visible_markdown}\n{visible_captions}"
    )
    result_metrics = result["test_metrics"]
    assert (
        f"{result_metrics['grayscale']['noise_estimate']:.0f} → "
        f"{result_metrics['filtered']['noise_estimate']:.0f}"
    ) in visible_markdown
    assert (
        f"{result_metrics['filtered']['contrast']:.1f} → "
        f"{result_metrics['enhanced']['contrast']:.1f}"
    ) in visible_markdown
    assert (
        f"{result_metrics['filtered']['dynamic_range']} → "
        f"{result_metrics['enhanced']['dynamic_range']}"
    ) in visible_markdown
    visible_content = visible_markdown + "\n" + "\n".join(
        tab.label for tab in app.tabs
    )
    assert not re.search(r"\bModules? [1-5]\b", visible_content, re.IGNORECASE)

    # Regression: a hot-reloaded session may contain an older preprocessing
    # metrics dictionary and grayscale data under the historical "original" key.
    result["test_metrics"].pop("processing_time_seconds")
    result["test_stages"]["original"] = result["test_stages"]["grayscale"]
    app.session_state["pcb_result"] = result
    app.run(timeout=45)
    assert not app.exception, [str(item.value) for item in app.exception]
    stale_result_markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "Pre-processing Time" in stale_result_markdown
    assert "N/A" in stale_result_markdown

    print(
        f"Streamlit AppTest passed: {len(app.tabs)} tabs, "
        f"{evaluation['total_defects']} severity-assessed defects, "
        f"{len(app.dataframe)} result tables, matching ROI {bounds}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


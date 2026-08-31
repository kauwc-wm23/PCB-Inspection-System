
import os
import sys
import time
import io
import importlib
import json
import hashlib
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.dataset_paths import SUPPORTED_IMAGE_EXTENSIONS, find_reference_image
from modules import preprocessing as _preprocessing
from modules import segmentation as _segmentation
from modules import reporting as _reporting
from gui import preprocessing_presentation as _preprocessing_presentation

if not hasattr(_preprocessing, "get_preprocessing_stages_from_array"):
    _preprocessing = importlib.reload(_preprocessing)

get_preprocessing_stages_from_array = (
    _preprocessing.get_preprocessing_stages_from_array
)

from modules import inspection_pipeline as _inspection_pipeline

_required_pipeline_helpers = (
    "CalibrationPipelineError",
    "create_compact_error_result",
    "create_compact_inspection_result",
    "run_inspection_arrays",
)
if not all(
    hasattr(_inspection_pipeline, helper_name)
    for helper_name in _required_pipeline_helpers
):
    _inspection_pipeline = importlib.reload(_inspection_pipeline)

CalibrationPipelineError = _inspection_pipeline.CalibrationPipelineError
create_compact_error_result = _inspection_pipeline.create_compact_error_result
create_compact_inspection_result = (
    _inspection_pipeline.create_compact_inspection_result
)
run_inspection_arrays = _inspection_pipeline.run_inspection_arrays

_required_reporting_helpers = (
    "build_inspection_conclusion",
    "generate_inspection_summary",
)
if not all(
    hasattr(_reporting, helper_name)
    for helper_name in _required_reporting_helpers
):
    _reporting = importlib.reload(_reporting)

build_inspection_conclusion = _reporting.build_inspection_conclusion
generate_inspection_summary = _reporting.generate_inspection_summary

_required_segmentation_helpers = (
    "analyse_morphology_effects",
    "count_foreground_components",
    "extract_change_detail_roi",
)
if not all(
    hasattr(_segmentation, helper_name)
    for helper_name in _required_segmentation_helpers
):
    _segmentation = importlib.reload(_segmentation)

extract_change_detail_roi = _segmentation.extract_change_detail_roi
get_segmentation_stages = _segmentation.get_segmentation_stages

if not hasattr(_preprocessing_presentation, "create_filtering_change_map"):
    _preprocessing_presentation = importlib.reload(_preprocessing_presentation)

create_filtering_change_map = (
    _preprocessing_presentation.create_filtering_change_map
)
extract_matching_center_rois = (
    _preprocessing_presentation.extract_matching_center_rois
)

OUTPUT_DIR_PRE = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")
OUTPUT_DIR_SEG = os.path.join(PROJECT_ROOT, "outputs", "segmentation")
RESULT_SCHEMA_VERSION = 3
BULK_SCHEMA_VERSION = 1
BULK_MAX_IMAGES = 10
os.makedirs(OUTPUT_DIR_PRE, exist_ok=True)
os.makedirs(OUTPUT_DIR_SEG, exist_ok=True)


def _build_bulk_items(uploaded_files):

    items = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        content = uploaded_file.getvalue()
        content_hash = hashlib.sha256(content).hexdigest()
        items.append({
            "item_id": f"{index:02d}-{content_hash[:12]}",
            "index": index,
            "filename": uploaded_file.name,
            "size": uploaded_file.size,
            "content": content,
            "content_hash": content_hash,
        })
    return items


def _bulk_signature(items):

    return tuple(
        (item["index"], item["filename"], item["content_hash"])
        for item in items
    )


def _bulk_summary(results):

    successful = [result for result in results if result["status"] == "SUCCESS"]
    passed = sum(
        result["inspection_result"] == "PASS"
        for result in successful
    )
    defective = sum(
        result["inspection_result"] == "DEFECTIVE"
        for result in successful
    )
    processing_times = [result["processing_time"] for result in successful]
    return {
        "submitted": len(results),
        "processed": len(results),
        "passed": passed,
        "defective": defective,
        "errors": len(results) - len(successful),
        "total_defects": sum(result["defect_count"] for result in successful),
        "average_processing_time": (
            sum(processing_times) / len(processing_times)
            if processing_times
            else 0.0
        ),
    }


def _process_bulk_item(item):

    started = time.perf_counter()

    def failure(status, message, reference_filename="N/A", calibration_status="N/A"):
        summary = create_compact_error_result(
            item_id=item["item_id"],
            filename=item["filename"],
            status=status,
            error_message=message,
            reference_filename=reference_filename,
            calibration_status=calibration_status,
        )
        summary["processing_time"] = time.perf_counter() - started
        return summary, None

    extension = os.path.splitext(item["filename"])[1].lower()
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        return failure(
            "UNSUPPORTED_FORMAT",
            "Supported formats are JPG, JPEG, and PNG.",
        )

    encoded = np.frombuffer(item["content"], np.uint8)
    test_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if test_bgr is None:
        return failure("DECODE_ERROR", "The uploaded image could not be decoded.")

    reference_path = find_reference_image(item["filename"])
    if reference_path is None:
        return failure(
            "REFERENCE_UNRESOLVED",
            "No matching PCB_USED reference was found for the filename board ID.",
        )
    if not reference_path.is_file():
        return failure(
            "REFERENCE_MISSING",
            "The resolved PCB reference file is missing.",
            reference_filename=reference_path.name,
        )

    reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        return failure(
            "REFERENCE_MISSING",
            "The resolved PCB reference could not be decoded.",
            reference_filename=reference_path.name,
        )

    try:
        detailed_result = run_inspection_arrays(
            test_bgr,
            reference_bgr,
            item["filename"],
            reference_path.name,
        )
    except CalibrationPipelineError as error:
        calibration_status = error.calibration_status
        status = (
            "CALIBRATION_FAILED"
            if calibration_status == "FAILED"
            else "CALIBRATION_UNVERIFIED"
        )
        return failure(
            status,
            str(error.metadata.get("message", error)),
            reference_filename=reference_path.name,
            calibration_status=calibration_status,
        )
    except Exception as error:
        return failure(
            "PROCESSING_ERROR",
            str(error) or "The inspection pipeline could not process this image.",
            reference_filename=reference_path.name,
        )

    return (
        create_compact_inspection_result(detailed_result, item["item_id"]),
        detailed_result,
    )


def _clear_active_inspection_results():

    st.session_state.pop("pcb_result", None)
    st.session_state.pop("bulk_detail_result", None)


def _as_feature_display_bgr(image):

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A valid source image is required for feature visualisation.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("Feature visualisation requires a grayscale or colour image.")


def _create_feature_overview_image(image, defects):

    canvas = _as_feature_display_bgr(image)
    image_height, image_width = canvas.shape[:2]
    annotation_colour = (225, 150, 55)

    for defect in defects:
        bounding_box = defect.get("bounding_box", {})
        raw_x_start = int(bounding_box.get("x", 0))
        raw_y_start = int(bounding_box.get("y", 0))
        box_width = max(0, int(bounding_box.get("width", 0)))
        box_height = max(0, int(bounding_box.get("height", 0)))
        x_start = max(0, raw_x_start)
        y_start = max(0, raw_y_start)
        x_end = min(
            image_width - 1,
            raw_x_start + box_width - 1,
        )
        y_end = min(
            image_height - 1,
            raw_y_start + box_height - 1,
        )
        if x_end < x_start or y_end < y_start:
            continue

        cv2.rectangle(
            canvas,
            (x_start, y_start),
            (x_end, y_end),
            annotation_colour,
            2,
        )
        label = f"D{defect.get('id', '')}"
        label_origin = (x_start, max(y_start - 7, 18))
        cv2.putText(
            canvas,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (8, 14, 22),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            annotation_colour,
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _create_feature_detail_roi(image, defect, padding_fraction=0.75):

    canvas = _as_feature_display_bgr(image)
    image_height, image_width = canvas.shape[:2]
    bounding_box = defect.get("bounding_box", {})
    box_x = int(bounding_box.get("x", 0))
    box_y = int(bounding_box.get("y", 0))
    box_width = max(0, int(bounding_box.get("width", 0)))
    box_height = max(0, int(bounding_box.get("height", 0)))
    if box_width == 0 or box_height == 0:
        return None, None

    x_padding = max(
        4,
        int(np.ceil(box_width * padding_fraction)),
        int(np.ceil(image_width * 0.005)),
    )
    y_padding = max(
        4,
        int(np.ceil(box_height * padding_fraction)),
        int(np.ceil(image_height * 0.005)),
    )
    roi_x_start = max(0, box_x - x_padding)
    roi_y_start = max(0, box_y - y_padding)
    roi_x_end = min(image_width, box_x + box_width + x_padding)
    roi_y_end = min(image_height, box_y + box_height + y_padding)
    if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
        return None, None

    roi = canvas[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()
    local_x_start = max(0, box_x - roi_x_start)
    local_y_start = max(0, box_y - roi_y_start)
    local_x_end = min(roi.shape[1] - 1, local_x_start + box_width - 1)
    local_y_end = min(roi.shape[0] - 1, local_y_start + box_height - 1)
    annotation_colour = (225, 150, 55)
    cv2.rectangle(
        roi,
        (local_x_start, local_y_start),
        (local_x_end, local_y_end),
        annotation_colour,
        2,
    )

    centroid = defect.get("centroid") or defect.get("location") or {}
    centroid_x = int(round(float(centroid.get("x", box_x + box_width / 2))))
    centroid_y = int(round(float(centroid.get("y", box_y + box_height / 2))))
    local_centroid = (
        min(max(centroid_x - roi_x_start, 0), roi.shape[1] - 1),
        min(max(centroid_y - roi_y_start, 0), roi.shape[0] - 1),
    )
    cv2.drawMarker(
        roi,
        local_centroid,
        (80, 220, 240),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=2,
    )

    bounds = (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), bounds


def _severity_colour_bgr(severity_level):

    return {
        "LOW": (94, 197, 34),
        "MEDIUM": (11, 158, 245),
        "HIGH": (68, 68, 239),
    }.get(str(severity_level).upper(), (190, 170, 140))


def _create_severity_map_image(
    image,
    assessed_defects,
    show_midpoint_guides=True,
):

    canvas = _as_feature_display_bgr(image)
    image_height, image_width = canvas.shape[:2]
    guide_colour = (115, 105, 95)
    annotation_scale = max(0.75, min(1.15, image_width / 2600.0))
    box_thickness = 4
    text_thickness = 2
    label_padding = 5
    occupied_label_boxes = []
    if show_midpoint_guides:
        cv2.line(
            canvas,
            (image_width, 0),
            (image_width, (image_height - 1) * 2),
            guide_colour,
            1,
            cv2.LINE_AA,
            1,
        )
        cv2.line(
            canvas,
            (0, image_height),
            ((image_width - 1) * 2, image_height),
            guide_colour,
            1,
            cv2.LINE_AA,
            1,
        )

    for defect in assessed_defects:
        bounding_box = defect.get("bounding_box", {})
        box_x = int(bounding_box.get("x", 0))
        box_y = int(bounding_box.get("y", 0))
        box_width = max(0, int(bounding_box.get("width", 0)))
        box_height = max(0, int(bounding_box.get("height", 0)))
        x_start = max(0, box_x)
        y_start = max(0, box_y)
        x_end = min(image_width - 1, box_x + box_width - 1)
        y_end = min(image_height - 1, box_y + box_height - 1)
        if x_end < x_start or y_end < y_start:
            continue

        annotation_colour = _severity_colour_bgr(defect.get("severity_level"))
        cv2.rectangle(
            canvas,
            (x_start, y_start),
            (x_end, y_end),
            annotation_colour,
            box_thickness,
        )
        centroid = defect.get("centroid") or defect.get("location") or {}
        centroid_point = (
            min(max(int(round(float(centroid.get("x", x_start)))), 0), image_width - 1),
            min(max(int(round(float(centroid.get("y", y_start)))), 0), image_height - 1),
        )
        cv2.drawMarker(
            canvas,
            centroid_point,
            annotation_colour,
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=3,
        )
        label = f"D{defect.get('id')} | {defect.get('severity_level', 'N/A')}"
        (text_width, text_height), text_baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            annotation_scale,
            text_thickness,
        )
        label_width = text_width + 2 * label_padding
        label_height = text_height + text_baseline + 2 * label_padding
        candidate_positions = (
            (x_start, y_start - label_height - 8),
            (x_start, y_end + 8),
            (x_end - label_width + 1, y_start - label_height - 8),
            (x_end - label_width + 1, y_end + 8),
        )
        label_box = None
        for candidate_x, candidate_y in candidate_positions:
            label_x = min(max(candidate_x, 0), max(image_width - label_width, 0))
            label_y = min(max(candidate_y, 0), max(image_height - label_height, 0))
            candidate_box = (
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height,
            )
            overlaps_existing_label = any(
                candidate_box[0] < occupied_box[2]
                and candidate_box[2] > occupied_box[0]
                and candidate_box[1] < occupied_box[3]
                and candidate_box[3] > occupied_box[1]
                for occupied_box in occupied_label_boxes
            )
            if not overlaps_existing_label:
                label_box = candidate_box
                break
        if label_box is None:
            label_box = candidate_box
        occupied_label_boxes.append(label_box)

        cv2.rectangle(
            canvas,
            (label_box[0], label_box[1]),
            (label_box[2], label_box[3]),
            (8, 14, 22),
            cv2.FILLED,
        )
        cv2.rectangle(
            canvas,
            (label_box[0], label_box[1]),
            (label_box[2], label_box[3]),
            annotation_colour,
            2,
        )
        label_origin = (
            label_box[0] + label_padding,
            label_box[1] + label_padding + text_height,
        )
        cv2.putText(
            canvas,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            annotation_scale,
            (8, 14, 22),
            text_thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            annotation_scale,
            annotation_colour,
            text_thickness,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _create_severity_detail_roi(image, assessed_defect):

    roi_rgb, bounds = _create_feature_detail_roi(image, assessed_defect)
    if roi_rgb is None or bounds is None:
        return None, None

    roi = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    bounding_box = assessed_defect.get("bounding_box", {})
    box_x = int(bounding_box.get("x", 0)) - bounds[0]
    box_y = int(bounding_box.get("y", 0)) - bounds[1]
    box_width = max(0, int(bounding_box.get("width", 0)))
    box_height = max(0, int(bounding_box.get("height", 0)))
    annotation_colour = _severity_colour_bgr(
        assessed_defect.get("severity_level")
    )
    cv2.rectangle(
        roi,
        (max(box_x, 0), max(box_y, 0)),
        (
            min(box_x + box_width - 1, roi.shape[1] - 1),
            min(box_y + box_height - 1, roi.shape[0] - 1),
        ),
        annotation_colour,
        2,
    )
    label = (
        f"D{assessed_defect.get('id')} | "
        f"{assessed_defect.get('severity_level', 'N/A')}"
    )
    cv2.putText(
        roi,
        label,
        (8, min(22, max(roi.shape[0] - 5, 8))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (8, 14, 22),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        roi,
        label,
        (8, min(22, max(roi.shape[0] - 5, 8))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        annotation_colour,
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), bounds


st.set_page_config(
    page_title="PCB Defect Inspection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.html("""
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
[data-testid="stAppViewContainer"] { background: #0a1019; color: #e5edf8; }
[data-testid="stHeader"] { background: rgba(10, 16, 25, 0.92); }
[data-testid="stMainBlockContainer"] {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
[data-testid="stSidebar"] {
    background: #0d1520;
    border-right: 1px solid #223044;
}
h1 { font-size: 2rem !important; line-height: 1.2 !important; font-weight: 680 !important; }
h2 { font-size: 1.3rem !important; line-height: 1.3 !important; font-weight: 650 !important; }
h3 { font-size: 1.02rem !important; line-height: 1.35 !important; font-weight: 620 !important; }
p, li, label { font-size: 0.9rem; }
[data-testid="stCaptionContainer"] { color: #91a0b5; font-size: 0.78rem; }
[data-testid="stMetric"] {
    min-height: 104px;
    background: #111b29;
    border: 1px solid #25364c;
    border-radius: 7px;
    padding: 0.8rem 0.9rem;
}
[data-testid="stMetricLabel"] { color: #91a0b5; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] {
    color: #f1f6fd;
    font-variant-numeric: tabular-nums;
    font-size: 1.55rem;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem; }
[data-testid="stFileUploaderDropzone"] {
    background: #101a28;
    border: 1px dashed #35506e;
    border-radius: 7px;
}
[data-testid="stImage"] img {
    border: 1px solid #25364c;
    border-radius: 6px;
}
[data-testid="stTabs"] {
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    width: 100% !important;
    max-width: 100% !important;
    height: 52px !important;
    box-sizing: border-box !important;
    gap: 12px !important;
    margin: 0 !important;
    padding: 4px !important;
    overflow: hidden !important;
    background-color: #0b1520 !important;
    border: 1px solid #26384b !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [role="tablist"] button[role="tab"] {
    flex: 1 1 0 !important;
    width: 20% !important;
    min-width: 0 !important;
    max-width: 20% !important;
    height: 44px !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    padding: 0.45rem 0.75rem !important;
    justify-content: center !important;
    border: 0 !important;
    border-radius: 4px !important;
    background-color: transparent !important;
    color: #a7b4c4 !important;
    opacity: 1 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
    letter-spacing: 0.005em !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    transition: background-color 0.16s ease, color 0.16s ease !important;
}
[data-testid="stTabs"] [role="tablist"] button[role="tab"]:hover {
    background-color: #122234 !important;
    color: #e2ebf6 !important;
}
[data-testid="stTabs"] [role="tablist"] button[role="tab"]:focus,
[data-testid="stTabs"] [role="tablist"] button[role="tab"]:focus-visible {
    outline: none !important;
    background-color: #122234 !important;
    color: #e2ebf6 !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [role="tablist"] button[role="tab"] p {
    margin: 0 !important;
    color: inherit !important;
    width: 100% !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    line-height: inherit !important;
    text-align: center !important;
    white-space: nowrap !important;
}
[data-testid="stTabs"] [role="tablist"] button[role="tab"][aria-selected="true"] {
    background-color: #14283c !important;
    color: #eef5ff !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [role="tablist"] button[role="tab"][aria-selected="true"] p {
    color: #eef5ff !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: block !important;
    height: 2px !important;
    background-color: #4f96dc !important;
    border: 0 !important;
    border-radius: 2px !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
    height: 0 !important;
    background-color: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #25364c;
    border-radius: 6px;
    overflow: hidden;
}
[data-testid="stExpander"] {
    background: #101925;
    border: 1px solid #25364c;
    border-radius: 6px;
}
.stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 600; }
.stButton > button[kind="primary"] { background: #2368ad; border-color: #3279bf; }
hr { border-color: #223044 !important; margin: 1.35rem 0 !important; }
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMetric"] { min-height: 94px; padding: 0.65rem; }
    [data-testid="stTabs"] [role="tablist"] button[role="tab"] {
        padding: 0.4rem 0.15rem !important;
        font-size: 0.68rem !important;
    }
}
</style>
""")


with st.sidebar:
    st.title("PCB Inspection")
    st.caption("Engineering quality workstation")
    st.divider()
    st.subheader("Inspection Workflow")
    st.markdown("**01  Image Pre-processing**")
    st.caption("Prepare and enhance the test and reference images.")
    st.markdown("**02  Defect Segmentation**")
    st.caption("Locate candidate structural differences.")
    st.markdown("**03  Feature Analysis**")
    st.caption("Measure connected defect regions.")
    st.markdown("**04  Severity & Spatial Analysis**")
    st.caption("Assess relative severity, priority, and position.")
    st.markdown("**05  Inspection Report**")
    st.caption("Review and export the inspection results.")
    st.divider()
    st.subheader("Production Pipeline")
    st.markdown("Colour → Grayscale → 5×5 Median → CLAHE")
    st.caption("CLAHE clip limit 2.0 · tile grid 8×8")
    st.caption("Dataset: Ironbrotherstyle PCB-DATASET")


st.caption("PCB QUALITY CONTROL")
st.title("PCB Defect Inspection System")
st.caption("Automated structural comparison and defect analysis")


workflow_columns = st.columns(5, gap="small")
for workflow_column, number, label in zip(
    workflow_columns,
    ("01", "02", "03", "04", "05"),
    ("Pre-process", "Segment", "Analyse", "Assess", "Report"),
):
    with workflow_column:
        st.caption(number)
        st.markdown(f"**{label}**")

st.divider()


st.subheader("Inspection Input")
st.caption("Load a test PCB image and its clean reference. Dataset references are matched automatically when possible.")

inspection_mode = st.radio(
    "Inspection Mode",
    ("Single Image", "Bulk Images"),
    horizontal=True,
    key="inspection_mode",
    on_change=_clear_active_inspection_results,
)

defective_file = None
template_file = None
bulk_selected_item = None
bulk_display_label_by_id = {}
bulk_items = []
bulk_current_signature = ()

if inspection_mode == "Single Image":
    up1, up2 = st.columns(2, gap="medium")

    with up1:
        st.markdown("**Test PCB Image**")
        st.caption("Defective or test board to inspect")
        defective_file = st.file_uploader(
            "Choose test PCB image",
            type=["jpg", "jpeg", "png"],
            help="Upload the defective/test PCB image",
            key="defective_upload",
        )

    with up2:
        st.markdown("**Reference PCB**")
        st.caption("Optional when the dataset board ID can be matched")
        template_file = st.file_uploader(
            "Choose clean reference PCB",
            type=["jpg", "jpeg", "png"],
            help="A PCB_USED reference is selected automatically when the dataset filename has a known board ID.",
            key="template_upload",
        )
else:
    st.markdown("**Bulk PCB Images**")
    st.caption(
        "Upload up to 10 PCB images. Each image is inspected sequentially "
        "against its automatically matched PCB_USED reference."
    )
    bulk_files = st.file_uploader(
        "Choose PCB images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Select or drag and drop multiple PCB images. Folder upload is not enabled.",
        key="bulk_uploads",
    )
    bulk_items = _build_bulk_items(bulk_files)
    bulk_current_signature = _bulk_signature(bulk_items)

    selection_over_limit = len(bulk_items) > BULK_MAX_IMAGES
    if selection_over_limit:
        st.error(
            f"This prototype supports a maximum of {BULK_MAX_IMAGES} images per "
            f"batch. Remove {len(bulk_items) - BULK_MAX_IMAGES} image(s) before "
            "starting the inspection."
        )
    elif bulk_items:
        st.caption(
            f"{len(bulk_items)} image(s) selected · maximum {BULK_MAX_IMAGES}"
        )
    else:
        st.info("Upload multiple PCB images to prepare a bulk inspection.")

    run_bulk = st.button(
        "▶ Run Bulk Inspection",
        type="primary",
        disabled=not bulk_items or selection_over_limit,
        width="stretch",
    )

    if run_bulk:
        st.session_state.pop("bulk_detail_result", None)
        st.session_state.pop("bulk_selected_item", None)
        bulk_results = []
        first_success_detail = None
        first_success_item = None
        progress = st.progress(0.0, text="Preparing bulk inspection...")

        for item_index, item in enumerate(bulk_items, start=1):
            progress.progress(
                (item_index - 1) / len(bulk_items),
                text=(
                    f"Processing {item_index} of {len(bulk_items)} · "
                    f"{item['filename']}"
                ),
            )
            try:
                summary, detailed_result = _process_bulk_item(item)
            except Exception as error:
                summary = create_compact_error_result(
                    item_id=item["item_id"],
                    filename=item["filename"],
                    status="PROCESSING_ERROR",
                    error_message=(
                        str(error)
                        or "The inspection pipeline could not process this image."
                    ),
                )
                detailed_result = None
            bulk_results.append(summary)
            if detailed_result is not None and first_success_detail is None:
                first_success_detail = detailed_result
                first_success_item = item
            else:
                detailed_result = None

        progress.progress(1.0, text="Bulk inspection complete.")
        summary_totals = _bulk_summary(bulk_results)
        st.session_state["bulk_batch"] = {
            "schema_version": BULK_SCHEMA_VERSION,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signature": bulk_current_signature,
            "batch_summary": summary_totals,
            "results": bulk_results,
        }

        if first_success_detail is not None and first_success_item is not None:
            first_reference = find_reference_image(first_success_item["filename"])
            first_signature = (
                "Bulk Images",
                first_success_item["item_id"],
                first_success_item["content_hash"],
                first_reference.name,
                first_reference.stat().st_size,
            )
            first_success_detail.update({
                "input_signature": first_signature,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "bulk_item_id": first_success_item["item_id"],
            })
            st.session_state["bulk_detail_result"] = first_success_detail
            st.session_state["bulk_selected_item"] = first_success_item["item_id"]

    stored_batch = st.session_state.get("bulk_batch")
    batch_is_current = (
        stored_batch is not None
        and stored_batch.get("schema_version") == BULK_SCHEMA_VERSION
        and stored_batch.get("signature") == bulk_current_signature
    )

    if stored_batch is not None and not batch_is_current:
        st.info("The selected files changed. Run Bulk Inspection to refresh the results.")

    if batch_is_current:
        batch_summary = stored_batch["batch_summary"]
        batch_results = stored_batch["results"]

        filename_counts = {}
        for result in batch_results:
            filename = result["filename"]
            filename_counts[filename] = filename_counts.get(filename, 0) + 1

        filename_positions = {}
        for result in batch_results:
            filename = result["filename"]
            filename_positions[filename] = filename_positions.get(filename, 0) + 1
            bulk_display_label_by_id[result["item_id"]] = (
                f"{filename} ({filename_positions[filename]})"
                if filename_counts[filename] > 1
                else filename
            )

        st.divider()
        st.markdown("### Bulk Inspection Summary")
        summary_columns = st.columns(6, gap="small")
        summary_columns[0].metric("Images Processed", batch_summary["processed"])
        summary_columns[1].metric("Passed", batch_summary["passed"])
        summary_columns[2].metric("Defective", batch_summary["defective"])
        summary_columns[3].metric("Errors", batch_summary["errors"])
        summary_columns[4].metric("Total Defects", batch_summary["total_defects"])
        summary_columns[5].metric(
            "Average Processing Time",
            f"{batch_summary['average_processing_time']:.2f} s",
        )

        table_rows = [
            {
                "Image": result["filename"],
                "Reference": result["reference_filename"],
                "Calibration": result["calibration_status"],
                "Warp": "YES" if result["warp_applied"] else "NO",
                "Defects": result["defect_count"],
                "Area": f"{result['total_defect_area']:,} px²",
                "Coverage": f"{result['coverage_percentage']:.4f}%",
                "Highest Severity": result["highest_severity"],
                "Region": result["dominant_spatial_region"],
                "Result": result["inspection_result"],
                "Time": f"{result['processing_time']:.2f} s",
                "Error": result["error_message"],
            }
            for result in batch_results
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

        bulk_export = {
            "schema_version": BULK_SCHEMA_VERSION,
            "generated_at": stored_batch["generated_at"],
            "batch_summary": batch_summary,
            "results": batch_results,
        }
        st.download_button(
            "Download Bulk Inspection Results (JSON)",
            data=json.dumps(bulk_export, indent=2, default=str),
            file_name="bulk_inspection_results.json",
            mime="application/json",
            type="secondary",
        )

        successful_results = [
            result for result in batch_results if result["status"] == "SUCCESS"
        ]
        if successful_results:
            successful_item_ids = [
                result["item_id"] for result in successful_results
            ]
            selected_item_id = st.selectbox(
                "Select Inspection Result",
                successful_item_ids,
                format_func=lambda item_id: bulk_display_label_by_id[item_id],
                key="bulk_selected_item",
            )
            bulk_selected_item = next(
                item for item in bulk_items if item["item_id"] == selected_item_id
            )
        else:
            st.info("No successful image is available for detailed inspection.")

if inspection_mode == "Single Image" and defective_file is not None:
    defective_name = defective_file.name
    defective_size = defective_file.size
    defective_content = defective_file.getvalue()
elif inspection_mode == "Bulk Images" and bulk_selected_item is not None:
    defective_name = bulk_selected_item["filename"]
    defective_size = bulk_selected_item["size"]
    defective_content = bulk_selected_item["content"]
else:
    defective_name = None
    defective_size = 0
    defective_content = None

auto_template_path = find_reference_image(defective_name) if defective_name else None
template_available = template_file is not None or auto_template_path is not None

if inspection_mode == "Single Image" and defective_name is None:
    st.info("Upload a test PCB image to begin. A matching PCB_USED reference is selected automatically for dataset images.")
elif inspection_mode == "Single Image" and not template_available:
    st.warning("No matching PCB_USED reference was found. Upload the corresponding clean template.")
elif inspection_mode == "Single Image" and template_file is None:
    st.info(f"Using matching dataset reference automatically: {auto_template_path.name}")


if defective_name is not None and template_available:

    test_bytes = np.frombuffer(defective_content, np.uint8)
    test_bgr = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    if template_file is not None:
        tmpl_bytes = np.frombuffer(template_file.getvalue(), np.uint8)
        template_bgr = cv2.imdecode(tmpl_bytes, cv2.IMREAD_COLOR)
        template_name = template_file.name
        template_size = template_file.size
        template_source = "Uploaded clean template"
    else:
        template_bgr = cv2.imread(str(auto_template_path), cv2.IMREAD_COLOR)
        template_name = auto_template_path.name
        template_size = auto_template_path.stat().st_size
        template_source = "Auto-matched from dataset/PCB_USED"

    if test_bgr is None or template_bgr is None:
        st.error("❌ Unable to decode one of the uploaded images.")
        st.stop()

    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    if inspection_mode == "Single Image":
        st.divider()
        st.subheader("Inspection Context")
        st.caption("Confirm the active image pair before running the inspection pipeline.")

        context_columns = st.columns(4, gap="small")
        context_columns[0].metric("Test Image", defective_name)
        context_columns[1].metric("Reference PCB", template_name)
        context_columns[2].metric(
            "Image Size",
            f"{test_bgr.shape[1]} × {test_bgr.shape[0]} px",
        )
        context_columns[3].metric("Inspection State", "Ready")

        st.markdown("### Input Pair Review")

        rv1, rv2 = st.columns(2, gap="medium")

        with rv1:
            st.image(test_rgb, caption=f"Defective/Test — {defective_name}", width="stretch")
            st.caption(
                f"Defective / Test PCB · {test_bgr.shape[1]} × {test_bgr.shape[0]} px · "
                f"BGR, 3 channels · {defective_size / 1024:.1f} KB · 8-bit per channel"
            )

        with rv2:
            st.image(template_rgb, caption=f"Template — {template_name}", width="stretch")
            st.caption(
                f"{template_source} · {template_bgr.shape[1]} × {template_bgr.shape[0]} px · "
                f"BGR, 3 channels · {template_size / 1024:.1f} KB · 8-bit per channel"
            )

        st.divider()
        st.subheader("Run Inspection")
        st.caption("Execute preprocessing, segmentation, measurement, assessment, and reporting in sequence.")

        input_signature = (
            defective_name,
            defective_size,
            template_name,
            template_size,
        )
        detail_state_key = "pcb_result"

        col_btn, col_reset, col_hint = st.columns([1, 1, 4], gap="small")
        with col_btn:
            run_btn = st.button("▶ Run Inspection", type="primary", width="stretch")
        with col_reset:
            reset_btn = st.button("↺ Reset", width="stretch")
            if reset_btn:
                st.session_state.pop(detail_state_key, None)
                st.rerun()
        with col_hint:
            st.caption(
                "Image Pre-processing → Defect Segmentation → Feature Analysis → "
                "Severity Assessment → Inspection Report"
            )
    else:
        input_signature = (
            "Bulk Images",
            bulk_selected_item["item_id"],
            bulk_selected_item["content_hash"],
            template_name,
            template_size,
        )
        detail_state_key = "bulk_detail_result"
        active_detail = st.session_state.get(detail_state_key)
        run_btn = not (
            active_detail is not None
            and active_detail.get("input_signature") == input_signature
            and active_detail.get("result_schema_version")
            == RESULT_SCHEMA_VERSION
        )
        st.caption(
            "Detailed inspection · "
            f"{bulk_display_label_by_id.get(bulk_selected_item['item_id'], defective_name)}"
        )

    if run_btn:
        try:
            with st.spinner("🔧 Running PCB Inspection Pipeline..."):
                detailed_result = run_inspection_arrays(
                    test_bgr,
                    template_bgr,
                    defective_name,
                    template_name,
                )
            detailed_result.update({
                "input_signature": input_signature,
                "result_schema_version": RESULT_SCHEMA_VERSION,
            })
            if inspection_mode == "Bulk Images":
                detailed_result["bulk_item_id"] = bulk_selected_item["item_id"]
            st.session_state[detail_state_key] = detailed_result

        except Exception as err:
            st.error(f"❌ Inspection failed: {err}")


    if (
        detail_state_key in st.session_state
        and st.session_state[detail_state_key].get("input_signature") == input_signature
        and st.session_state[detail_state_key].get("result_schema_version")
        == RESULT_SCHEMA_VERSION
    ):

        result = st.session_state[detail_state_key]

        test_stages = result["test_stages"]
        test_metrics = result["test_metrics"]
        template_stages = result["template_stages"]
        template_metrics = result["template_metrics"]
        calibration_metadata = result["calibration_metadata"]
        processed_test = result["processed_test"]
        processed_template = result["processed_template"]
        seg_stages = result["seg_stages"]
        seg_metrics = result["seg_metrics"]
        evaluation = result["evaluation"]
        proc_time = result["proc_time"]

        output_prefix = (
            f"{result['bulk_item_id']}_"
            if result.get("bulk_item_id")
            else ""
        )
        out_name_pre = f"preprocessed_{output_prefix}{result['test_name']}"
        out_path_pre = os.path.join(OUTPUT_DIR_PRE, out_name_pre)
        cv2.imwrite(out_path_pre, processed_test)

        out_name_seg = (
            f"segmented_{output_prefix}"
            f"{os.path.splitext(result['test_name'])[0]}.png"
        )
        out_path_seg = os.path.join(OUTPUT_DIR_SEG, out_name_seg)
        if "overlay" in seg_stages:
            cv2.imwrite(out_path_seg, seg_stages["overlay"])

        completed_defect_count = evaluation["total_defect_count"]
        completed_region_word = (
            "region" if completed_defect_count == 1 else "regions"
        )
        st.info(
            f"Inspection completed · {completed_defect_count} potential defect "
            f"{completed_region_word} detected · {proc_time:.2f} s"
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Image Pre-processing",
            "Defect Segmentation",
            "Feature Analysis",
            "Severity & Spatial Analysis",
            "Inspection Report",
        ])

        with tab1:

            st.markdown("### 1. Image Calibration / Rectification")
            st.caption(
                "Reference-relative geometric registration for spatial consistency."
            )

            calibration_status = calibration_metadata["status"]
            calibration_status_display = calibration_status.replace("_", " ")
            test_dimensions = (
                f"{calibration_metadata['test_width']} × "
                f"{calibration_metadata['test_height']} px"
            )
            reference_dimensions = (
                f"{calibration_metadata['reference_width']} × "
                f"{calibration_metadata['reference_height']} px"
            )
            dimensions_match = test_dimensions == reference_dimensions
            calibration_metrics = st.columns(4, gap="small")
            calibration_metrics[0].metric(
                "Calibration Status",
                calibration_status_display,
            )
            calibration_metrics[1].metric(
                "Alignment Method",
                "ORB + RANSAC",
            )
            calibration_metrics[2].metric(
                "Warp Applied",
                "Yes" if calibration_metadata["warp_applied"] else "No",
            )
            calibration_metrics[3].metric(
                "Image Dimensions",
                test_dimensions if dimensions_match else "Test ≠ Reference",
            )

            scale_value = calibration_metadata.get("scale")
            rotation_value = calibration_metadata.get("rotation_deg")
            translation_x = calibration_metadata.get("translation_x")
            translation_y = calibration_metadata.get("translation_y")
            inlier_ratio = calibration_metadata.get("inlier_ratio")
            transform_metrics = st.columns(4, gap="small")
            transform_metrics[0].metric(
                "Rotation",
                f"{rotation_value:.4f}°" if rotation_value is not None else "N/A",
            )
            transform_metrics[1].metric(
                "Scale",
                f"{scale_value:.6f}" if scale_value is not None else "N/A",
            )
            transform_metrics[2].metric(
                "Translation",
                (
                    f"{translation_x:+.2f}, {translation_y:+.2f} px"
                    if translation_x is not None and translation_y is not None
                    else "N/A"
                ),
            )
            transform_metrics[3].metric(
                "RANSAC Inliers",
                (
                    f"{calibration_metadata['inlier_count']}/"
                    f"{calibration_metadata['match_count']} ({inlier_ratio:.1%})"
                    if inlier_ratio is not None
                    else "N/A"
                ),
            )

            if calibration_status == "ALREADY_ALIGNED":
                st.success(calibration_metadata["message"])
            else:
                st.info(calibration_metadata["message"])

            calibrated_test_rgb = cv2.cvtColor(
                _as_feature_display_bgr(test_stages["original"]),
                cv2.COLOR_BGR2RGB,
            )
            calibration_views = st.columns(3, gap="medium")
            with calibration_views[0]:
                st.markdown("**TEST IMAGE**")
                st.caption("Before calibration")
                st.image(test_rgb, width="stretch")
            with calibration_views[1]:
                st.markdown("**REFERENCE IMAGE**")
                st.caption("Target coordinate system")
                st.image(template_rgb, width="stretch")
            with calibration_views[2]:
                st.markdown("**CALIBRATED TEST IMAGE**")
                st.caption(
                    "Original pixels preserved"
                    if not calibration_metadata["warp_applied"]
                    else "Validated transform applied"
                )
                st.image(calibrated_test_rgb, width="stretch")

            with st.expander("Calibration Details", expanded=False):
                reprojection_error = calibration_metadata.get(
                    "reprojection_error"
                )
                overlap_ratio = calibration_metadata.get("overlap_ratio")
                dimension_details = (
                    ""
                    if dimensions_match
                    else (
                        f"**Test dimensions:** {test_dimensions}  \n"
                        f"**Reference dimensions:** {reference_dimensions}  \n"
                    )
                )
                st.markdown(
                    dimension_details
                    + f"**Method:** {calibration_metadata['method']}  \n"
                    f"**Reliable matches:** {calibration_metadata['match_count']}  \n"
                    f"**RANSAC inliers:** {calibration_metadata['inlier_count']}  \n"
                    f"**Reprojection error:** "
                    f"{f'{reprojection_error:.3f} px' if reprojection_error is not None else 'N/A'}  \n"
                    f"**Valid overlap:** "
                    f"{f'{overlap_ratio:.2%}' if overlap_ratio is not None else 'N/A'}"
                )
                st.info(
                    "Physical pixel-to-mm scaling is unavailable because the dataset "
                    "does not provide a validated physical reference. Measurements "
                    "therefore remain in pixels and pixel²."
                )

            st.divider()
            st.markdown("### 2. Image Enhancement")
            st.caption("CALIBRATE / ALIGN → ENHANCE → SEGMENT")
            st.markdown("#### Image Pre-processing Pipeline")
            st.caption("Prepare the PCB image for structural comparison.")

            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            stage_defs = [
                (sc1, "original", "01", "Original Colour", "Raw colour PCB image."),
                (sc2, "grayscale", "02", "Grayscale", "Single-channel intensity image."),
                (sc3, "filtered", "03", "Median Filtered", "5×5 median-filtered image."),
                (sc4, "enhanced", "04", "CLAHE Enhanced", "Locally contrast-enhanced image."),
            ]

            for col, key, num, name, desc in stage_defs:
                with col:
                    with st.container(height=112, border=False):
                        st.caption(f"STAGE {num}")
                        st.markdown(f"**{name}**")
                        st.caption(desc)
                    if key == "original":
                        st.image(calibrated_test_rgb, width="stretch")
                    else:
                        st.image(
                            test_stages[key], clamp=True, channels="GRAY", width="stretch"
                        )

            st.divider()
            st.markdown("### Image Quality Improvement")
            st.caption("Measured values from the actual preprocessing stages.")

            m_gray = test_metrics["grayscale"]
            m_enh = test_metrics["enhanced"]
            m_filt = test_metrics["filtered"]
            preprocessing_time = test_metrics.get("processing_time_seconds")
            preprocessing_time_available = (
                isinstance(
                    preprocessing_time,
                    (int, float, np.integer, np.floating),
                )
                and not isinstance(preprocessing_time, (bool, np.bool_))
                and np.isfinite(preprocessing_time)
                and preprocessing_time >= 0
            )
            preprocessing_time_display = (
                f"{float(preprocessing_time):.3f}s"
                if preprocessing_time_available
                else "N/A"
            )
            preprocessing_time_summary = (
                f"{float(preprocessing_time):.3f} seconds"
                if preprocessing_time_available
                else "N/A"
            )

            brightness_delta = m_enh["mean_brightness"] - m_gray["mean_brightness"]
            contrast_gain = 0.0
            if m_filt["contrast"] > 0:
                contrast_gain = (
                    (m_enh["contrast"] - m_filt["contrast"])
                    / m_filt["contrast"]
                ) * 100
            contrast_gain_display = f"{contrast_gain:+.1f}% vs median"

            high_frequency_reduction = test_metrics.get(
                "high_frequency_reduction", test_metrics["noise_reduction"]
            )
            gray_high_frequency = m_gray.get(
                "high_frequency_estimate", m_gray["noise_estimate"]
            )
            filtered_high_frequency = m_filt.get(
                "high_frequency_estimate", m_filt["noise_estimate"]
            )
            if high_frequency_reduction > 0:
                high_frequency_summary = (
                    f"High-frequency estimate reduced by {high_frequency_reduction:.1f}%."
                )
            elif high_frequency_reduction < 0:
                high_frequency_summary = (
                    f"High-frequency estimate increased by {abs(high_frequency_reduction):.1f}%."
                )
            else:
                high_frequency_summary = "High-frequency estimate was unchanged."
            if high_frequency_reduction > 0:
                high_frequency_delta_display = (
                    f"{high_frequency_reduction:.1f}% lower"
                )
            elif high_frequency_reduction < 0:
                high_frequency_delta_display = (
                    f"{abs(high_frequency_reduction):.1f}% higher"
                )
            else:
                high_frequency_delta_display = "No change"

            dynamic_range_change = m_enh["dynamic_range"] - m_filt["dynamic_range"]
            metric_row_one = st.columns(3, gap="small")
            metric_row_one[0].metric(
                "Mean Intensity",
                f"{m_gray['mean_brightness']:.1f} → {m_enh['mean_brightness']:.1f}",
                f"{brightness_delta:+.1f} levels",
            )
            metric_row_one[1].metric(
                "Contrast",
                f"{m_filt['contrast']:.1f} → {m_enh['contrast']:.1f}",
                contrast_gain_display,
            )
            metric_row_one[2].metric(
                "High-Frequency Estimate",
                f"{gray_high_frequency:.0f} → {filtered_high_frequency:.0f}",
                high_frequency_delta_display,
                delta_color="off",
            )
            metric_row_two = st.columns(3, gap="small")
            metric_row_two[0].metric(
                "Dynamic Range",
                f"{m_filt['dynamic_range']} → {m_enh['dynamic_range']}",
                f"{dynamic_range_change:+d} levels",
            )
            metric_row_two[1].metric(
                "Pre-processing Time",
                preprocessing_time_display,
            )
            st.markdown(
                f"Measured comparison: {gray_high_frequency:.0f} → {filtered_high_frequency:.0f} "
                f"high-frequency estimate · {m_filt['contrast']:.1f} → {m_enh['contrast']:.1f} "
                f"contrast · {m_filt['dynamic_range']} → {m_enh['dynamic_range']} dynamic range · "
                f"Pre-processing Time {preprocessing_time_display}."
            )

            st.divider()
            st.markdown("### Local Processing Comparison")
            st.caption("All views use the same proportional region of interest for a valid local comparison.")

            filtering_change_raw, filtering_change_display = (
                create_filtering_change_map(
                    test_stages["grayscale"], test_stages["filtered"]
                )
            )
            roi_views, roi_bounds = extract_matching_center_rois(
                {
                    "grayscale": test_stages["grayscale"],
                    "filtered": test_stages["filtered"],
                    "change_map": filtering_change_display,
                    "enhanced": test_stages["enhanced"],
                }
            )
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
            st.markdown("#### A. Median Filtering Effect")
            roi_col1, roi_col2, roi_col3 = st.columns(3, gap="medium")
            with roi_col1:
                st.markdown("**GRAYSCALE INPUT**")
                st.caption("Before median filtering")
                st.image(
                    roi_views["grayscale"], clamp=True, channels="GRAY", width="stretch"
                )
            with roi_col2:
                st.markdown("**MEDIAN FILTERED**")
                st.caption("After 5×5 filtering")
                st.image(
                    roi_views["filtered"], clamp=True, channels="GRAY", width="stretch"
                )
            with roi_col3:
                st.markdown("**FILTERING CHANGE MAP**")
                st.caption("Display-normalized modified pixels")
                st.image(
                    roi_views["change_map"], clamp=True, channels="GRAY", width="stretch"
                )
            st.caption(
                f"All local views use the identical central ROI: x={roi_x1}:{roi_x2}, "
                f"y={roi_y1}:{roi_y2} ({roi_x2 - roi_x1}×{roi_y2 - roi_y1} px). "
                f"The change map is display-normalized |Grayscale − Median|; its raw "
                f"maximum change is {int(filtering_change_raw.max())} intensity levels and "
                "it is not used by the processing pipeline. Bright regions indicate stronger "
                "filtering changes and may include potential noise, compression artefacts, "
                "real PCB edges, or fine detail; they should not be interpreted directly as noise."
            )

            st.metric(
                "High-Frequency Estimate · Before → After",
                f"{gray_high_frequency:.0f} → {filtered_high_frequency:.0f}",
                f"{high_frequency_reduction:+.1f}% change",
            )
            st.caption(high_frequency_summary)

            st.markdown("#### B. Contrast Enhancement")
            clahe_col1, clahe_col2 = st.columns(2, gap="medium")
            with clahe_col1:
                st.markdown("**MEDIAN FILTERED**")
                st.caption("Before CLAHE")
                st.image(
                    roi_views["filtered"], clamp=True, channels="GRAY", width="stretch"
                )
            with clahe_col2:
                st.markdown("**CLAHE ENHANCED**")
                st.caption("After local contrast enhancement")
                st.image(
                    roi_views["enhanced"], clamp=True, channels="GRAY", width="stretch"
                )

            enhancement_metrics = st.columns(2, gap="small")
            enhancement_metrics[0].metric(
                "Contrast · Before → After",
                f"{m_filt['contrast']:.1f} → {m_enh['contrast']:.1f}",
                contrast_gain_display,
            )
            enhancement_metrics[1].metric(
                "Dynamic Range · Before → After",
                f"{m_filt['dynamic_range']} → {m_enh['dynamic_range']}",
                f"{dynamic_range_change:+d} levels",
            )

            st.divider()
            st.markdown("### Pixel Intensity Distribution")

            fig, ax = plt.subplots(figsize=(12, 3.2))
            fig.patch.set_facecolor("#0b0f19")
            ax.set_facecolor("#0e1520")

            ax.hist(test_stages["grayscale"].ravel(), bins=256, range=(0,255),
                    color="#5fa8ff", alpha=0.55, label="Original Grayscale", density=True)
            ax.hist(test_stages["enhanced"].ravel(), bins=256, range=(0,255),
                    color="#22c55e", alpha=0.65, label="Final CLAHE Enhanced", density=True)

            ax.set_xlabel("Pixel Intensity (0 = black · 255 = white)", color="#4a6890", fontsize=9)
            ax.set_ylabel("Normalised Frequency", color="#4a6890", fontsize=9)
            ax.tick_params(colors="#2a4060", labelsize=8)
            ax.spines[["top","right","left","bottom"]].set_color("#1a2840")
            ax.set_xlim(0, 255)
            ax.set_title(
                "Before Enhancement vs After Enhancement",
                color="#3a5070", fontsize=8.5, pad=8
            )
            legend = ax.legend(fontsize=8.5, framealpha=0.15,
                               labelcolor="white", facecolor="#0e1520")
            for line in legend.get_lines():
                line.set_linewidth(2)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                        facecolor="#0b0f19")
            buf.seek(0)
            st.image(buf, width="stretch")
            plt.close(fig)
            st.caption(
                "The histogram shows how CLAHE redistributes local pixel intensities, "
                "supporting local contrast enhancement. This does not automatically imply "
                "better segmentation for every image."
            )

            st.divider()
            st.markdown("### Original vs Preprocessed Output")
            st.caption("Full-frame input and final production preprocessing output.")

            fc1, fc2 = st.columns(2, gap="medium")
            with fc1:
                st.markdown("**INPUT · ORIGINAL COLOUR**")
                st.image(calibrated_test_rgb, width="stretch")

            with fc2:
                st.markdown("**OUTPUT · CLAHE ENHANCED**")
                st.image(processed_test, clamp=True, channels="GRAY", width="stretch")

            st.divider()
            st.markdown("### Pre-processing Summary")
            st.success("Image ready for defect segmentation")
            st.markdown(
                "- Reference-relative calibration verified\n"
                "- Grayscale conversion completed\n"
                "- 5×5 median filtering applied to suppress potential impulse noise and small local variations\n"
                "- Local contrast enhanced using CLAHE\n"
                "- Final single-channel image prepared for defect segmentation"
            )
            st.caption(
                f"Processing time: {preprocessing_time_summary} · "
                f"Output: {processed_test.shape[1]}×{processed_test.shape[0]} px · "
                f"Range: {int(processed_test.min())}–{int(processed_test.max())}"
            )

            _, dl_col, _ = st.columns([2, 1, 2])
            with dl_col:
                ok, enc = cv2.imencode(".png", processed_test)
                if ok:
                    st.download_button(
                        label="⬇ Download Preprocessed Image",
                        data=enc.tobytes(),
                        file_name=f"preprocessed_{os.path.splitext(result['test_name'])[0]}.png",
                        mime="image/png",
                        width="stretch",
                    )

        with tab2:

            st.markdown("### Defect Segmentation")
            st.caption("Compare the preprocessed test PCB with its reference and localise candidate defects.")

            st.markdown("#### Main Segmentation Result")
            main_overlay = seg_stages.get("overlay")
            _, main_result_column, _ = st.columns([0.21, 0.58, 0.21])
            with main_result_column:
                if main_overlay is None:
                    st.info("Final overlay is not available from segmentation.py.")
                elif len(main_overlay.shape) == 3:
                    st.image(
                        cv2.cvtColor(main_overlay, cv2.COLOR_BGR2RGB),
                        caption="Detected candidate defect regions with bounding boxes",
                        width="stretch",
                    )
                else:
                    st.image(
                        main_overlay,
                        clamp=True,
                        channels="GRAY",
                        caption="Detected candidate defect regions with bounding boxes",
                        width="stretch",
                    )

            st.divider()
            st.markdown("#### Segmentation Overview")

            thresh = seg_metrics.get("threshold", "Auto")
            count = seg_metrics.get("defect_count", seg_metrics.get("detected_regions", 0))
            area_px = seg_metrics.get("defect_area_px", 0)
            area_pct = seg_metrics.get("defect_area_pct", 0.0)

            segmentation_metrics = st.columns(4, gap="small")
            segmentation_metrics[0].metric("Otsu Threshold", thresh)
            segmentation_metrics[1].metric("Defect Regions", count)
            segmentation_metrics[2].metric("Defect Area", f"{area_px} px")
            segmentation_metrics[3].metric("Surface Coverage", f"{area_pct:.4f}%")
            st.caption(
                "The threshold is selected automatically; region count, contour area, and "
                "surface coverage are reported directly by segmentation.py."
            )

            st.divider()
            st.markdown("#### Morphological Processing")
            st.caption(
                "Morphological processing refines the binary defect mask before contour detection."
            )

            morphology_explanations = st.columns(2, gap="medium")
            with morphology_explanations[0]:
                st.markdown("**Opening**")
                st.caption("Removes small isolated foreground responses.")
            with morphology_explanations[1]:
                st.markdown("**Closing**")
                st.caption("Reconnects nearby foreground regions and fills small gaps.")

            pre_morphology_mask = seg_stages.get("otsu_binary")
            opened_mask = seg_stages.get("opening")
            closed_mask = seg_stages.get("morphology")
            opening_removed_map = seg_stages.get("opening_removed")
            closing_added_map = seg_stages.get("closing_added")
            morphology_analysis = seg_metrics.get("morphology_analysis")
            morphology_configuration = seg_metrics.get("morphology_configuration")

            morphology_masks_available = all(
                mask is not None
                for mask in (pre_morphology_mask, opened_mask, closed_mask)
            )
            morphology_evidence_available = (
                morphology_masks_available
                and opening_removed_map is not None
                and closing_added_map is not None
                and morphology_analysis is not None
            )

            if morphology_masks_available:
                st.markdown("##### Processing Sequence")
                sequence_columns = st.columns(3, gap="small")
                sequence_items = (
                    (
                        sequence_columns[0],
                        "BEFORE MORPHOLOGY",
                        "Binary Mask",
                        pre_morphology_mask,
                    ),
                    (
                        sequence_columns[1],
                        "AFTER OPENING",
                        "Opened Mask",
                        opened_mask,
                    ),
                    (
                        sequence_columns[2],
                        "AFTER CLOSING",
                        "Final Morphology Mask",
                        closed_mask,
                    ),
                )
                for column, stage_label, mask_label, mask in sequence_items:
                    with column:
                        st.caption(stage_label)
                        st.markdown(f"**{mask_label}**")
                        _, sequence_image_column, _ = st.columns([0.14, 0.72, 0.14])
                        with sequence_image_column:
                            st.image(
                                mask,
                                clamp=True,
                                channels="GRAY",
                                width="stretch",
                            )
            else:
                st.info(
                    "Morphology stage masks are unavailable. Run the inspection again "
                    "to refresh this result."
                )

            if morphology_evidence_available:
                opening_metrics = morphology_analysis["opening"]
                closing_metrics = morphology_analysis["closing"]
                opening_detail, opening_bounds = extract_change_detail_roi(
                    opening_removed_map
                )
                closing_detail, closing_bounds = extract_change_detail_roi(
                    closing_added_map
                )

                st.markdown("#### Morphology Impact")
                morphology_impact = [
                    {
                        "Operation": "Opening",
                        "Pixel Change": f"{opening_metrics['pixel_change']:,} removed",
                        "Change %": f"{opening_metrics['change_percentage']:.3f}%",
                        "Components": (
                            f"{opening_metrics['components_before']} → "
                            f"{opening_metrics['components_after']}"
                        ),
                    },
                    {
                        "Operation": "Closing",
                        "Pixel Change": f"{closing_metrics['pixel_change']:,} added",
                        "Change %": f"{closing_metrics['change_percentage']:.3f}%",
                        "Components": (
                            f"{closing_metrics['components_before']} → "
                            f"{closing_metrics['components_after']}"
                        ),
                    },
                ]
                st.dataframe(
                    morphology_impact,
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Pixel changes and foreground-component counts are measured from the "
                    "original-resolution runtime masks; component counts exclude background."
                )

                st.markdown("#### Morphology Change Detail")
                change_detail_columns = st.columns(2, gap="medium")
                with change_detail_columns[0]:
                    st.markdown("**Opening Effect**")
                    st.markdown("**Opening Change Detail**")
                    if opening_metrics["pixel_change"] == 0 or opening_detail is None:
                        st.info("No foreground pixels were removed by Opening for this image.")
                        st.caption(
                            f"{opening_metrics['pixel_change']:,} removed · "
                            f"{opening_metrics['change_percentage']:.3f}%"
                        )
                    else:
                        _, opening_detail_column, _ = st.columns([0.12, 0.76, 0.12])
                        with opening_detail_column:
                            st.image(
                                opening_detail,
                                clamp=True,
                                channels="GRAY",
                                width="stretch",
                            )
                        st.caption(
                            f"Opening changed {opening_metrics['pixel_change']:,} pixels "
                            f"({opening_metrics['change_percentage']:.3f}% of the foreground)."
                        )
                        if opening_metrics["pixel_change"] <= 10:
                            st.caption(
                                "Only a small number of pixels were changed in this operation."
                            )
                    st.caption(
                        "Bright pixels show foreground responses removed by Opening. These "
                        "changes may include isolated responses or fine structures and should "
                        "not automatically be interpreted as noise."
                    )

                with change_detail_columns[1]:
                    st.markdown("**Closing Effect**")
                    st.markdown("**Closing Change Detail**")
                    if closing_metrics["pixel_change"] == 0 or closing_detail is None:
                        st.info(
                            "No foreground pixels were added by Closing for this image."
                        )
                        st.caption(
                            f"{closing_metrics['pixel_change']:,} added · "
                            f"{closing_metrics['change_percentage']:.3f}%"
                        )
                    else:
                        _, closing_detail_column, _ = st.columns([0.12, 0.76, 0.12])
                        with closing_detail_column:
                            st.image(
                                closing_detail,
                                clamp=True,
                                channels="GRAY",
                                width="stretch",
                            )
                        st.caption(
                            f"Closing changed {closing_metrics['pixel_change']:,} pixels "
                            f"({closing_metrics['change_percentage']:.3f}% of the foreground)."
                        )
                        if closing_metrics["pixel_change"] <= 10:
                            st.caption(
                                "Only a small number of pixels were changed in this operation."
                            )
                    st.caption(
                        "Bright pixels, when present, show foreground pixels added while "
                        "reconnecting nearby areas or filling small gaps. These changes should "
                        "not automatically be interpreted as beneficial corrections."
                    )

                with st.expander("View full morphology change maps"):
                    full_map_columns = st.columns(2, gap="medium")
                    with full_map_columns[0]:
                        st.markdown("**Opening — Removed Pixels**")
                        st.image(
                            opening_removed_map,
                            clamp=True,
                            channels="GRAY",
                            width="stretch",
                        )
                    with full_map_columns[1]:
                        st.markdown("**Closing — Added Pixels**")
                        st.image(
                            closing_added_map,
                            clamp=True,
                            channels="GRAY",
                            width="stretch",
                        )
                    st.caption(
                        "Bright pixels indicate where each morphology operation changed "
                        "the binary mask."
                    )

                with st.expander("Morphology measurement details"):
                    st.markdown("**Morphology Summary**")
                    st.caption("Complete runtime values used by the impact summary.")
                    measurement_columns = st.columns(2, gap="medium")
                    with measurement_columns[0]:
                        st.markdown("**Opening**")
                        st.metric(
                            "Foreground Before Opening",
                            f"{opening_metrics['foreground_before']:,}",
                        )
                        st.metric(
                            "Foreground After Opening",
                            f"{opening_metrics['foreground_after']:,}",
                        )
                        st.metric(
                            "Pixels Removed by Opening",
                            f"{opening_metrics['pixel_change']:,}",
                        )
                        st.metric(
                            "Foreground Removed",
                            f"{opening_metrics['change_percentage']:.3f}%",
                        )
                        st.metric(
                            "Components Before Opening",
                            opening_metrics["components_before"],
                        )
                        st.metric(
                            "Components After Opening",
                            opening_metrics["components_after"],
                        )
                    with measurement_columns[1]:
                        st.markdown("**Closing**")
                        st.metric(
                            "Foreground Before Closing",
                            f"{closing_metrics['foreground_before']:,}",
                        )
                        st.metric(
                            "Foreground After Closing",
                            f"{closing_metrics['foreground_after']:,}",
                        )
                        st.metric(
                            "Pixels Added by Closing",
                            f"{closing_metrics['pixel_change']:,}",
                        )
                        st.metric(
                            "Foreground Added",
                            f"{closing_metrics['change_percentage']:.3f}%",
                        )
                        st.metric(
                            "Components Before Closing",
                            closing_metrics["components_before"],
                        )
                        st.metric(
                            "Components After Closing",
                            closing_metrics["components_after"],
                        )
                    st.caption(
                        "Foreground components exclude the background. A reduction can "
                        "indicate that foreground regions became connected, but it does "
                        "not by itself prove successful gap repair."
                    )
            else:
                st.info(
                    "Morphology measurements are unavailable for this stored result. "
                    "Run the inspection again to generate them from the production masks."
                )

            with st.expander("Morphology Configuration"):
                if morphology_configuration is None:
                    st.info("Configuration metadata is unavailable for this stored result.")
                else:
                    configuration_columns = st.columns(2, gap="medium")
                    for column, operation in zip(
                        configuration_columns,
                        ("opening", "closing"),
                    ):
                        settings = morphology_configuration[operation]
                        kernel_height, kernel_width = settings["kernel_size"]
                        with column:
                            st.markdown(f"**{operation.title()}**")
                            st.write(
                                f"Kernel: {settings['kernel_shape']} "
                                f"{kernel_width}×{kernel_height}"
                            )
                            st.write(f"Iterations: {settings['iterations']}")
                    st.caption(
                        "Read-only production configuration. No segmentation parameters "
                        "can be changed from this view."
                    )

            st.divider()
            st.markdown("#### Supporting Segmentation Stages")
            st.caption("The complete production sequence remains available for technical review.")

            stage_order = [
                ("test_image", "Stage 1", "Preprocessed Test", "Pre-processing output"),
                ("template_image", "Stage 2", "Preprocessed Template", "Reference input"),
                ("difference", "Stage 3", "Absolute Difference", "Test vs template"),
                ("otsu_binary", "Stage 4", "Otsu Binary", "Automatic thresholding"),
                ("opening", "Stage 5", "Opening", "Isolated-response suppression"),
                ("morphology", "Stage 6", "Closing", "Gap filling and reconnection"),
            ]

            with st.expander("View all six production stages"):
                for stage_row in (stage_order[:3], stage_order[3:]):
                    stage_columns = st.columns(3, gap="small")
                    for column, item in zip(stage_columns, stage_row):
                        key, number, name, description = item
                        with column:
                            st.caption(number.upper())
                            st.markdown(f"**{name}**")
                            st.caption(description)
                            stage_image = seg_stages.get(key)
                            if stage_image is None:
                                st.info("Stage output is not available from segmentation.py.")
                            elif len(stage_image.shape) == 3:
                                st.image(
                                    cv2.cvtColor(stage_image, cv2.COLOR_BGR2RGB),
                                    width="stretch",
                                )
                            else:
                                st.image(
                                    stage_image,
                                    clamp=True,
                                    channels="GRAY",
                                    width="stretch",
                                )

            with st.expander("Segmentation Pipeline Explanation"):
                st.markdown("**Step 1 · Absolute Difference**")
                st.write(
                    "Compares the test PCB with its matched reference and highlights "
                    "structural differences."
                )
                st.markdown("**Step 2 · Otsu Thresholding**")
                st.write(
                    "Automatically converts the difference image into a binary "
                    "candidate-defect mask."
                )
                st.caption(f"Selected threshold: {thresh}")
                st.markdown("**Step 3 · Morphological Processing**")
                st.write(
                    "Opening removes small isolated responses, while Closing reconnects "
                    "nearby foreground regions and fills small gaps."
                )
                st.markdown("**Step 4 · Contour Detection**")
                st.write(
                    "Extracts connected candidate regions and draws bounding boxes for "
                    "localisation."
                )
                st.caption(f"Detected regions: {count}")

            st.divider()
            _, dl_col2, _ = st.columns([2, 1, 2])
            with dl_col2:
                overlay = seg_stages.get("overlay")
                ok, enc2 = (False, None) if overlay is None else cv2.imencode(".png", overlay)
                if ok:
                    st.download_button(
                        label="⬇ Download Defect Detection Result",
                        data=enc2.tobytes(),
                        file_name=out_name_seg,
                        mime="image/png",
                        width="stretch",
                    )

        with tab3:

            st.markdown("### Feature Analysis")
            st.caption("Measure and inspect the geometric properties of detected defect regions.")

            defects = result.get("defects", [])
            analysis_metrics = result.get("analysis_metrics", {})
            total_defects = analysis_metrics.get("total_defects", 0)
            total_area = analysis_metrics.get("total_defect_area", 0)
            average_area = analysis_metrics.get("average_area", 0.0)
            largest_defect = analysis_metrics.get("largest_defect")
            largest_defect_display = (
                f"D{largest_defect.get('id')} · {largest_defect.get('area', 0):,} px²"
                if largest_defect
                else "N/A"
            )

            st.markdown("#### Feature Analysis Overview")
            feature_metrics = st.columns(4, gap="small")
            feature_metrics[0].metric("Detected Regions", total_defects)
            feature_metrics[1].metric("Total Defect Area", f"{total_area:,} px²")
            feature_metrics[2].metric("Largest Defect", largest_defect_display)
            feature_metrics[3].metric(
                "Average Defect Area",
                f"{average_area:,.1f} px²" if total_defects else "N/A",
            )

            if not defects or total_defects == 0:
                st.info(
                    "No candidate defect regions are available for feature analysis under "
                    "the current segmentation criteria."
                )
            else:
                feature_source_image = test_stages.get("original")
                if feature_source_image is None:
                    feature_source_image = seg_stages.get("test_image", processed_test)

                st.divider()
                st.markdown("#### Detected Defect Regions")
                st.caption(
                    "Existing feature IDs and bounding boxes are overlaid on the original PCB image."
                )
                feature_overview = _create_feature_overview_image(
                    feature_source_image,
                    defects,
                )
                _, feature_overview_column, _ = st.columns([0.24, 0.52, 0.24])
                with feature_overview_column:
                    st.image(
                        feature_overview,
                        caption=(
                            "Detected candidate regions labelled with their canonical "
                            "Feature Analysis IDs."
                        ),
                        width="stretch",
                    )

                st.divider()
                st.markdown("#### Defect Size Comparison")
                st.caption("Compare the relative pixel area of the detected candidate regions.")
                st.markdown("##### Defect Area Comparison")

                defect_labels = [f"D{defect.get('id')}" for defect in defects]
                defect_areas = [int(defect.get("area", 0)) for defect in defects]
                area_figure, area_axis = plt.subplots(figsize=(8.2, 3.0))
                area_figure.patch.set_facecolor("#0b0f19")
                area_axis.set_facecolor("#0e1520")
                area_bars = area_axis.bar(
                    defect_labels,
                    defect_areas,
                    color="#3b82f6",
                    width=0.58,
                )
                area_axis.set_xlabel("Defect ID", color="#91a0b5", fontsize=9)
                area_axis.set_ylabel("Area (px²)", color="#91a0b5", fontsize=9)
                area_axis.tick_params(colors="#91a0b5", labelsize=8.5)
                area_axis.spines[["top", "right"]].set_visible(False)
                area_axis.spines[["left", "bottom"]].set_color("#2a3d54")
                area_axis.grid(axis="y", color="#26384b", alpha=0.55, linewidth=0.7)
                area_axis.set_axisbelow(True)
                area_ceiling = max(defect_areas) if defect_areas else 0
                area_axis.set_ylim(0, max(1, area_ceiling * 1.18))
                for area_bar, area_value in zip(area_bars, defect_areas):
                    area_axis.text(
                        area_bar.get_x() + area_bar.get_width() / 2,
                        area_bar.get_height() + max(area_ceiling * 0.025, 0.5),
                        f"{area_value:,}",
                        ha="center",
                        va="bottom",
                        color="#d9e6f5",
                        fontsize=8,
                    )
                area_figure.tight_layout()
                area_buffer = io.BytesIO()
                area_figure.savefig(
                    area_buffer,
                    format="png",
                    dpi=130,
                    bbox_inches="tight",
                    facecolor="#0b0f19",
                )
                area_buffer.seek(0)
                _, area_chart_column, _ = st.columns([0.12, 0.76, 0.12])
                with area_chart_column:
                    st.image(area_buffer, width="stretch")
                plt.close(area_figure)

                st.divider()
                st.markdown("#### Individual Defect Analysis")
                defect_ids = [defect.get("id") for defect in defects]
                selected_defect_id = st.selectbox(
                    "Select Defect",
                    options=defect_ids,
                    format_func=lambda defect_id: f"D{defect_id}",
                    key="feature_analysis_selected_defect",
                )
                selected_defect = next(
                    defect
                    for defect in defects
                    if defect.get("id") == selected_defect_id
                )
                selected_roi, selected_roi_bounds = _create_feature_detail_roi(
                    feature_source_image,
                    selected_defect,
                )
                selected_bbox = selected_defect.get("bounding_box", {})
                selected_centroid = (
                    selected_defect.get("centroid")
                    or selected_defect.get("location")
                    or {}
                )

                defect_detail_columns = st.columns([0.52, 0.48], gap="medium")
                with defect_detail_columns[0]:
                    st.caption("SELECTED DEFECT ROI")
                    if selected_roi is None:
                        st.info("A valid display ROI is unavailable for this defect.")
                    else:
                        st.image(
                            selected_roi,
                            caption=(
                                "Display-only crop with the existing bounding box and "
                                "centroid marker."
                            ),
                            width="stretch",
                        )
                        st.caption(f"Padded ROI bounds: {selected_roi_bounds}")
                with defect_detail_columns[1]:
                    st.caption("EXTRACTED MEASUREMENTS")
                    selected_metric_row_one = st.columns(2, gap="small")
                    selected_metric_row_one[0].metric(
                        "Defect ID",
                        f"D{selected_defect.get('id')}",
                    )
                    selected_metric_row_one[1].metric(
                        "Area",
                        f"{selected_defect.get('area', 0):,} px²",
                    )
                    selected_metric_row_two = st.columns(2, gap="small")
                    selected_metric_row_two[0].metric(
                        "Width",
                        f"{selected_defect.get('width', 0)} px",
                    )
                    selected_metric_row_two[1].metric(
                        "Height",
                        f"{selected_defect.get('height', 0)} px",
                    )
                    st.markdown(
                        "**Centroid:** "
                        f"({selected_centroid.get('x', 0):.1f}, "
                        f"{selected_centroid.get('y', 0):.1f})"
                    )
                    st.markdown(
                        "**Bounding Box (x, y, w, h):** "
                        f"({selected_bbox.get('x', 0)}, {selected_bbox.get('y', 0)}, "
                        f"{selected_bbox.get('width', 0)}, "
                        f"{selected_bbox.get('height', 0)})"
                    )

                st.divider()
                st.markdown("#### Defect Measurements")
                st.caption(
                    "Complete geometric measurements for every canonical defect ID."
                )

                table_data = []
                for defect in defects:
                    bbox = defect.get("bounding_box", {})
                    centroid = defect.get("centroid") or defect.get("location") or {}
                    table_data.append({
                        "Defect ID": f"D{defect.get('id', '')}",
                        "Area (px²)": defect.get("area", 0),
                        "Width (px)": defect.get("width", 0),
                        "Height (px)": defect.get("height", 0),
                        "Centroid (x, y)": (
                            f"({centroid.get('x', 0):.1f}, "
                            f"{centroid.get('y', 0):.1f})"
                        ),
                        "Bounding Box (x, y, w, h)": (
                            f"({bbox.get('x', 0)}, {bbox.get('y', 0)}, "
                            f"{bbox.get('width', 0)}, {bbox.get('height', 0)})"
                        ),
                    })

                st.dataframe(
                    table_data,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Defect ID": st.column_config.TextColumn(),
                        "Area (px²)": st.column_config.NumberColumn(format="%d"),
                        "Width (px)": st.column_config.NumberColumn(format="%d"),
                        "Height (px)": st.column_config.NumberColumn(format="%d"),
                        "Centroid (x, y)": st.column_config.TextColumn(),
                        "Bounding Box (x, y, w, h)": st.column_config.TextColumn(),
                    }
                )

            st.divider()
            with st.expander("Feature Extraction Method"):
                st.markdown("**Step 1 · Connected Region Identification**")
                st.write(
                    "The final segmentation mask is restricted to contours already accepted "
                    "by segmentation, then 8-connected components assign sequential IDs."
                )
                st.markdown("**Step 2 · Bounding Box Measurement**")
                st.write(
                    "Connected-component statistics provide each region's position, width, "
                    "and height."
                )
                st.markdown("**Step 3 · Area Measurement**")
                st.write(
                    "The connected-component pixel count quantifies the spatial extent of "
                    "each candidate region."
                )
                st.markdown("**Step 4 · Centroid Calculation**")
                st.write(
                    "The connected-component centroid provides the approximate centre "
                    "position of each detected region."
                )
                st.markdown("**Step 5 · Downstream Analysis**")
                st.write(
                    "The same IDs and geometric measurements pass to Severity & Spatial "
                    "Analysis for relative severity assessment, priority ranking, and "
                    "spatial localisation."
                )
                st.caption(
                    "Measurements are extracted using OpenCV connectedComponentsWithStats()."
                )

        with tab4:

            st.markdown("### Severity & Spatial Analysis")
            st.caption(
                "Assess relative geometric severity, priority, and spatial distribution "
                "of detected defect regions."
            )

            assessed_defects = evaluation.get("defects", [])
            assessed_count = evaluation.get("total_defect_count", 0)
            highest_severity = evaluation.get("highest_severity_level", "N/A")
            priority_id = evaluation.get("highest_priority_defect_id", "N/A")
            priority_display = (
                "N/A" if priority_id == "N/A" else f"D{priority_id}"
            )
            dominant_region = evaluation.get("most_concentrated_region", "N/A")

            st.markdown("#### Assessment Overview")
            overview_metrics = st.columns(4, gap="small")
            overview_metrics[0].metric("Assessed Defects", assessed_count)
            overview_metrics[1].metric("Highest Severity", highest_severity)
            overview_metrics[2].metric(
                "Highest Priority Defect",
                priority_display,
            )
            overview_metrics[3].metric("Dominant Spatial Region", dominant_region)
            st.caption(
                "Severity describes relative geometric extent only. It does not estimate "
                "electrical failure, defect type, or manufacturing risk."
            )

            severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for assessed_defect in assessed_defects:
                severity_level = assessed_defect.get("severity_level")
                if severity_level in severity_counts:
                    severity_counts[severity_level] += 1

            st.markdown("##### Severity Distribution")
            distribution_metrics = st.columns(3, gap="small")
            for distribution_column, severity_level in zip(
                distribution_metrics,
                ("LOW", "MEDIUM", "HIGH"),
            ):
                with distribution_column:
                    st.caption(severity_level)
                    st.markdown(f"**{severity_counts[severity_level]}**")

            st.divider()
            st.markdown("#### Defect Priority Ranking")
            st.caption(
                "Ordered only by the priority ranks already produced by the existing "
                "severity assessment."
            )
            if assessed_defects:
                ranked_defects = sorted(
                    assessed_defects,
                    key=lambda defect: defect.get("priority_rank", float("inf")),
                )
                highest_priority_defect = ranked_defects[0]
                st.info(
                    f"Priority #1 · D{highest_priority_defect.get('id')} · "
                    f"{highest_priority_defect.get('severity_level', 'N/A')} relative "
                    "geometric severity"
                )
                priority_rows = [
                    {
                        "Priority Rank": f"#{defect.get('priority_rank')}",
                        "Defect ID": f"D{defect.get('id')}",
                        "Severity": defect.get("severity_level", "N/A"),
                        "Severity Score": f"{defect.get('severity_score', 0.0):.6f}",
                        "Area (px²)": defect.get("area", 0),
                        "Spatial Region": defect.get("spatial_region", "N/A"),
                    }
                    for defect in ranked_defects
                ]
                st.dataframe(
                    priority_rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Priority Rank": st.column_config.TextColumn(),
                        "Defect ID": st.column_config.TextColumn(),
                        "Severity": st.column_config.TextColumn(),
                        "Severity Score": st.column_config.TextColumn(),
                        "Area (px²)": st.column_config.NumberColumn(format="%d"),
                        "Spatial Region": st.column_config.TextColumn(),
                    },
                )
            else:
                st.info(
                    "No valid defects were detected, so no priority ranking is available."
                )

            st.divider()
            st.markdown("#### Spatial Distribution")
            st.caption(
                "Defect centroids are assigned against the horizontal and vertical image "
                "midpoints using the existing quadrant rule."
            )
            spatial_distribution = evaluation.get("spatial_distribution", {})
            spatial_metrics = st.columns(4, gap="small")
            for metric_column, region in zip(
                spatial_metrics,
                ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"),
            ):
                metric_column.metric(region, spatial_distribution.get(region, 0))

            spatial_summary_columns = st.columns(3, gap="small")
            spatial_summaries = (
                (
                    "CONCENTRATED REGION",
                    evaluation.get("most_concentrated_region", "N/A"),
                ),
                (
                    "LARGEST-DEFECT REGION",
                    evaluation.get("largest_defect_region", "N/A"),
                ),
                (
                    "HIGHEST-PRIORITY REGION",
                    evaluation.get("highest_priority_defect_region", "N/A"),
                ),
            )
            for summary_column, (summary_label, summary_value) in zip(
                spatial_summary_columns,
                spatial_summaries,
            ):
                with summary_column:
                    st.caption(summary_label)
                    st.markdown(f"**{summary_value}**")

            st.divider()
            st.markdown("#### PCB Severity Map")
            if assessed_defects:
                severity_source_image = test_stages.get("original")
                if severity_source_image is None:
                    severity_source_image = seg_stages.get(
                        "test_image",
                        processed_test,
                    )
                severity_map = _create_severity_map_image(
                    severity_source_image,
                    assessed_defects,
                )
                _, severity_map_column, _ = st.columns([0.24, 0.52, 0.24])
                with severity_map_column:
                    st.image(
                        severity_map,
                        caption=(
                            "Canonical defect IDs, stored LOW / MEDIUM / HIGH labels, "
                            "existing bounding boxes, centroids, and midpoint guides."
                        ),
                        width="stretch",
                    )
                st.caption(
                    "Map colours distinguish the stored severity labels; each annotation "
                    "also states its label so colour is not the only indicator."
                )
            else:
                st.info("No assessed defect regions are available for severity mapping.")

            st.divider()
            st.markdown("#### Individual Defect Assessment")
            if assessed_defects:
                defect_ids = [defect.get("id") for defect in assessed_defects]
                default_defect_index = (
                    defect_ids.index(priority_id)
                    if priority_id in defect_ids
                    else 0
                )
                selected_defect_id = st.selectbox(
                    "Select Defect",
                    options=defect_ids,
                    index=default_defect_index,
                    format_func=lambda defect_id: f"D{defect_id}",
                    key="severity_analysis_selected_defect",
                )
                selected_defect = next(
                    defect
                    for defect in assessed_defects
                    if defect.get("id") == selected_defect_id
                )

                severity_source_image = test_stages.get("original")
                if severity_source_image is None:
                    severity_source_image = seg_stages.get(
                        "test_image",
                        processed_test,
                    )
                selected_roi, _ = _create_severity_detail_roi(
                    severity_source_image,
                    selected_defect,
                )

                detail_columns = st.columns([0.52, 0.48], gap="medium")
                with detail_columns[0]:
                    st.caption("SELECTED DEFECT ROI")
                    if selected_roi is None:
                        st.info("A valid display ROI is unavailable for this defect.")
                    else:
                        st.image(
                            selected_roi,
                            caption=(
                                "Display-only crop using the existing bounding box, "
                                "centroid, ID, and severity label."
                            ),
                            width="stretch",
                        )
                with detail_columns[1]:
                    st.caption("STORED ASSESSMENT")
                    assessment_row_one = st.columns(2, gap="small")
                    assessment_row_one[0].metric(
                        "Defect ID",
                        f"D{selected_defect.get('id')}",
                    )
                    assessment_row_one[1].metric(
                        "Priority Rank",
                        f"#{selected_defect.get('priority_rank')}",
                    )
                    assessment_row_two = st.columns(2, gap="small")
                    assessment_row_two[0].metric(
                        "Geometric Severity",
                        selected_defect.get("severity_level", "N/A"),
                    )
                    assessment_row_two[1].metric(
                        "Severity Score",
                        f"{selected_defect.get('severity_score', 0.0):.6f}",
                    )
                    assessment_row_three = st.columns(3, gap="small")
                    assessment_row_three[0].metric(
                        "Area",
                        f"{selected_defect.get('area', 0):,} px²",
                    )
                    assessment_row_three[1].metric(
                        "Width",
                        f"{selected_defect.get('width', 0)} px",
                    )
                    assessment_row_three[2].metric(
                        "Height",
                        f"{selected_defect.get('height', 0)} px",
                    )
                    st.metric(
                        "Spatial Region",
                        selected_defect.get("spatial_region", "N/A"),
                    )

                st.markdown("##### Severity Score Breakdown")
                weights = evaluation.get("severity_weights", {})
                component_labels = ["Area", "Width", "Height"]
                component_ratios = [
                    float(selected_defect.get("area_ratio", 0.0)),
                    float(selected_defect.get("width_ratio", 0.0)),
                    float(selected_defect.get("height_ratio", 0.0)),
                ]
                component_weights = [
                    float(weights.get("area", 0.0)),
                    float(weights.get("width", 0.0)),
                    float(weights.get("height", 0.0)),
                ]
                weighted_contributions = [
                    component_weight * component_ratio
                    for component_weight, component_ratio in zip(
                        component_weights,
                        component_ratios,
                    )
                ]

                ratio_metrics = st.columns(3, gap="small")
                for ratio_column, label, ratio, weight in zip(
                    ratio_metrics,
                    component_labels,
                    component_ratios,
                    component_weights,
                ):
                    ratio_column.metric(
                        f"{label} Ratio",
                        f"{ratio:.6f}",
                        help=f"Existing {label.lower()} ratio with weight {weight:.2f}.",
                    )

                breakdown_figure, breakdown_axis = plt.subplots(figsize=(8.0, 2.7))
                breakdown_figure.patch.set_facecolor("#0b0f19")
                breakdown_axis.set_facecolor("#0e1520")
                breakdown_bars = breakdown_axis.barh(
                    component_labels,
                    weighted_contributions,
                    color="#3b82f6",
                    height=0.50,
                )
                breakdown_axis.set_xlabel(
                    "Weighted contribution to severity score",
                    color="#91a0b5",
                    fontsize=9,
                )
                breakdown_axis.tick_params(colors="#91a0b5", labelsize=8.5)
                breakdown_axis.spines[["top", "right"]].set_visible(False)
                breakdown_axis.spines[["left", "bottom"]].set_color("#2a3d54")
                breakdown_axis.grid(
                    axis="x",
                    color="#26384b",
                    alpha=0.55,
                    linewidth=0.7,
                )
                breakdown_axis.set_axisbelow(True)
                contribution_ceiling = max(weighted_contributions, default=0.0)
                breakdown_axis.set_xlim(0, max(1e-6, contribution_ceiling * 1.28))
                for contribution_bar, contribution_value in zip(
                    breakdown_bars,
                    weighted_contributions,
                ):
                    breakdown_axis.text(
                        contribution_bar.get_width()
                        + max(contribution_ceiling * 0.025, 1e-8),
                        contribution_bar.get_y() + contribution_bar.get_height() / 2,
                        f"{contribution_value:.6f}",
                        ha="left",
                        va="center",
                        color="#d9e6f5",
                        fontsize=8,
                    )
                breakdown_figure.tight_layout()
                breakdown_buffer = io.BytesIO()
                breakdown_figure.savefig(
                    breakdown_buffer,
                    format="png",
                    dpi=130,
                    bbox_inches="tight",
                    facecolor="#0b0f19",
                )
                breakdown_buffer.seek(0)
                _, breakdown_chart_column, _ = st.columns([0.12, 0.76, 0.12])
                with breakdown_chart_column:
                    st.image(breakdown_buffer, width="stretch")
                plt.close(breakdown_figure)

                st.metric(
                    "Final Relative Geometric Severity Score",
                    f"{selected_defect.get('severity_score', 0.0):.6f}",
                )
                dominant_component_index = int(np.argmax(weighted_contributions))
                dominant_component = component_labels[dominant_component_index]
                dominant_contribution = weighted_contributions[
                    dominant_component_index
                ]
                st.info(
                    f"D{selected_defect.get('id')} is Priority "
                    f"#{selected_defect.get('priority_rank')} under the existing "
                    f"geometric ranking. Its largest weighted score contribution is "
                    f"{dominant_component.lower()} ({dominant_contribution:.6f}). "
                    "This is a relative geometric comparison, not an electrical or "
                    "manufacturing-risk prediction."
                )
            else:
                st.info(
                    "No valid defects were detected, so individual severity assessment "
                    "is not available."
                )

            st.divider()
            thresholds = evaluation.get("severity_thresholds", {})
            weights = evaluation.get("severity_weights", {})
            with st.expander("Severity Assessment Method"):
                st.markdown("**Relative Geometric Severity Formula**")
                st.write(
                    "area ratio = defect area / image area; width ratio = defect width / "
                    "image width; height ratio = defect height / image height."
                )
                st.markdown(
                    f"Score = {weights.get('area', 0.0):.2f} × area ratio + "
                    f"{weights.get('width', 0.0):.2f} × width ratio + "
                    f"{weights.get('height', 0.0):.2f} × height ratio."
                )

                st.markdown("**Prototype Severity Thresholds**")
                st.markdown(
                    f"- **LOW:** score ≤ "
                    f"{thresholds.get('low_max_score', 0.0):.4f}\n"
                    f"- **MEDIUM:** {thresholds.get('low_max_score', 0.0):.4f} "
                    f"< score ≤ {thresholds.get('medium_max_score', 0.0):.4f}\n"
                    f"- **HIGH:** score > "
                    f"{thresholds.get('medium_max_score', 0.0):.4f}"
                )
                st.warning(
                    "These thresholds provide relative geometric categorisation for this "
                    "prototype; they are not validated manufacturing acceptance limits."
                )

                st.markdown("**Priority Ranking Rule**")
                st.write(
                    "Existing ranks order defects by severity score descending, then area "
                    "descending, defect identifier ascending, and original input order."
                )

                st.markdown("**Spatial Quadrant Assignment**")
                st.write(
                    "A centroid left of the image midpoint is LEFT; otherwise it is RIGHT. "
                    "A centroid above the image midpoint is TOP; otherwise it is BOTTOM. "
                    "A centroid exactly on a midpoint therefore belongs to RIGHT and/or "
                    "BOTTOM."
                )

                st.markdown("**Interpretation Limitation**")
                st.info(
                    "The assessment compares geometric extent among detected regions. It "
                    "does not diagnose defect type, electrical behaviour, reliability, or "
                    "production acceptance."
                )

        with tab5:

            st.markdown("### Inspection Report")
            st.caption(
                "Final inspection decision, consolidated assessment results, and "
                "export actions."
            )

            test_name = result.get("test_name", "test_image.jpg")
            proc_time = result.get("proc_time", 0.0)

            inspection_report = result["inspection_report"]
            inspection_conclusion = build_inspection_conclusion(inspection_report)
            report_defects = inspection_report.get("defects", [])

            st.markdown("#### Inspection Summary")
            total_defects = inspection_report["total_defects"]
            total_area = inspection_report["total_defect_area"]
            avg_area = inspection_report["average_defect_area"]
            largest_area = inspection_report["largest_defect_area"]
            detected_coverage = inspection_report["defect_coverage_percentage"]
            status = inspection_report["inspection_status"]

            if status.startswith("NORMAL"):
                st.success(f"Inspection Result: {status}")
            else:
                st.error(f"Inspection Result: {status}")
            report_metrics = st.columns(4, gap="small")
            report_metrics[0].metric("Total Defects", total_defects)
            report_metrics[1].metric(
                "Total Defect Area",
                f"{total_area:,} px²",
            )
            report_metrics[2].metric(
                "Detected Coverage",
                f"{detected_coverage:.4f}%",
            )
            report_metrics[3].metric("Processing Time", f"{proc_time:.2f} s")

            st.divider()
            st.markdown("#### Inspection Conclusion")
            st.info(inspection_conclusion)

            st.divider()
            st.markdown("#### Final Inspection Visualisation")
            report_source_image = test_stages.get("original")
            if report_source_image is None:
                report_source_image = result.get("seg_stages", {}).get(
                    "test_image",
                    processed_test,
                )

            if report_source_image is None:
                st.info("A source PCB image is not available for report visualisation.")
            else:
                final_report_visualisation = _create_severity_map_image(
                    report_source_image,
                    report_defects,
                    show_midpoint_guides=False,
                )
                _, report_visualisation_column, _ = st.columns([0.24, 0.52, 0.24])
                with report_visualisation_column:
                    st.image(
                        final_report_visualisation,
                        width="stretch",
                        caption=(
                            "Canonical defect IDs and stored relative geometric severity "
                            "labels on the original PCB image."
                            if report_defects
                            else "Original PCB image with no valid defect annotations."
                        ),
                    )
                if report_defects:
                    st.caption(
                        "Annotations use the same canonical IDs and bounding boxes as "
                        "Feature Analysis and Severity & Spatial Analysis."
                    )

            st.divider()
            st.markdown("#### Defect Assessment Summary")

            if not report_defects or total_defects == 0:
                st.info(
                    "No individual defect measurements are available because no valid "
                    "defect regions were detected."
                )
            else:
                assessment_rows = [
                    {
                        "Defect ID": f"D{defect.get('id')}",
                        "Area": f"{defect.get('area', 0):,} px²",
                        "Size": (
                            f"{defect.get('width', 0)} × "
                            f"{defect.get('height', 0)} px"
                        ),
                        "Severity Score": (
                            f"{defect.get('severity_score', 0.0):.6f}"
                        ),
                        "Severity": defect.get("severity_level", "N/A"),
                        "Priority": f"#{defect.get('priority_rank')}",
                        "Region": defect.get("spatial_region", "N/A"),
                    }
                    for defect in report_defects
                ]

                st.dataframe(
                    assessment_rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Defect ID": st.column_config.TextColumn(),
                        "Area": st.column_config.TextColumn(),
                        "Size": st.column_config.TextColumn(),
                        "Severity Score": st.column_config.TextColumn(),
                        "Severity": st.column_config.TextColumn(),
                        "Priority": st.column_config.TextColumn(),
                        "Region": st.column_config.TextColumn(),
                    },
                )

            st.divider()
            with st.expander("Detailed Measurements"):
                st.markdown(
                    f"**Average Defect Area:** {avg_area:.1f} px² · "
                    f"**Largest Defect Area:** {largest_area:,} px²"
                )
                if not report_defects or total_defects == 0:
                    st.info("No detailed defect measurements are available.")
                else:
                    detailed_rows = []
                    for defect in report_defects:
                        centroid = (
                            defect.get("centroid")
                            or defect.get("location")
                            or {}
                        )
                        bounding_box = defect.get("bounding_box", {})
                        detailed_rows.append({
                            "Defect ID": f"D{defect.get('id')}",
                            "Centroid X": f"{centroid.get('x', 0):.1f}",
                            "Centroid Y": f"{centroid.get('y', 0):.1f}",
                            "Bounding Box (x, y, w, h)": (
                                f"({bounding_box.get('x', 0)}, "
                                f"{bounding_box.get('y', 0)}, "
                                f"{bounding_box.get('width', 0)}, "
                                f"{bounding_box.get('height', 0)})"
                            ),
                            "Area Ratio (%)": (
                                f"{defect.get('area_ratio', 0.0) * 100:.4f}"
                            ),
                            "Width (px)": defect.get("width", 0),
                            "Height (px)": defect.get("height", 0),
                            "Area (px²)": defect.get("area", 0),
                        })
                    st.dataframe(
                        detailed_rows,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Defect ID": st.column_config.TextColumn(),
                            "Centroid X": st.column_config.TextColumn(),
                            "Centroid Y": st.column_config.TextColumn(),
                            "Bounding Box (x, y, w, h)": (
                                st.column_config.TextColumn()
                            ),
                            "Area Ratio (%)": st.column_config.TextColumn(),
                            "Width (px)": st.column_config.NumberColumn(format="%d"),
                            "Height (px)": st.column_config.NumberColumn(format="%d"),
                            "Area (px²)": st.column_config.NumberColumn(format="%d"),
                        },
                    )

            st.divider()
            st.markdown("#### Report Information")
            report_information_row_one = st.columns(2, gap="small")
            with report_information_row_one[0]:
                st.caption("TEST IMAGE")
                st.markdown(f"**{inspection_report['test_filename']}**")
            with report_information_row_one[1]:
                st.caption("REFERENCE PCB")
                st.markdown(f"**{inspection_report['template_filename']}**")
            report_information_row_two = st.columns(2, gap="small")
            with report_information_row_two[0]:
                st.caption("PROCESSING TIME")
                st.markdown(
                    f"**{inspection_report['processing_time']:.2f} seconds**"
                )
            with report_information_row_two[1]:
                st.caption("GENERATED TIME")
                st.markdown(f"**{inspection_report['timestamp']}**")

            report_payload = json.dumps(inspection_report, indent=2, default=str)
            st.download_button(
                "Download Inspection Report (JSON)",
                data=report_payload,
                file_name=(
                    f"inspection_report_{os.path.splitext(test_name)[0]}.json"
                ),
                mime="application/json",
                type="secondary",
            )

            if inspection_mode == "Single Image":
                try:
                    from modules.pdf_reporting import build_inspection_pdf

                    pdf_payload = build_inspection_pdf(
                        inspection_report,
                        inspection_conclusion,
                    )
                    if not (
                        isinstance(pdf_payload, bytes)
                        and pdf_payload.startswith(b"%PDF-")
                    ):
                        raise ValueError("PDF generation returned invalid data.")
                except Exception:
                    st.warning(
                        "PDF export is unavailable for this result. "
                        "JSON export remains available."
                    )
                else:
                    test_stem = os.path.splitext(os.path.basename(test_name))[0]
                    st.download_button(
                        "Download Inspection Report (PDF)",
                        data=pdf_payload,
                        file_name=f"{test_stem}_inspection_report.pdf",
                        mime="application/pdf",
                        type="secondary",
                    )

else:
    if inspection_mode == "Single Image":
        st.info("👆 Upload a defective/test PCB image to get started; add a template if no dataset reference matches.")

"""
=============================================================================
Module      : segmentation.py
Project     : PCB Defect Inspection System

Description :
    Module 2 — Defect Segmentation and Detection.

    Processing Pipeline:
        1. Compare preprocessed defective PCB with defect-free template
        2. Compute absolute image difference
        3. Apply Otsu automatic thresholding
        4. Apply Morphological Opening
        5. Apply Morphological Closing
        6. Detect defect contours
        7. Visualise detected defect regions

Output:
    Segmented defect regions represented by contours.

Notes:
    - Both test and template images should be preprocessed using the same
      Module 1 pipeline before entering this module.
    - Template and test images must represent the same PCB type/orientation.
    - Feature extraction such as area, width, height, bounding-box
      measurements and defect location is handled by Module 3.
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


# =============================================================================
# Constants / Hyperparameters
# =============================================================================

KERNEL_OPEN_SIZE = 3
KERNEL_CLOSE_SIZE = 3

MIN_DEFECT_AREA = 15
MAX_DEFECT_AREA = 5000

# Keep disabled unless weak defect regions need slight expansion.
EXTRA_DILATION_ITERATIONS = 0


# =============================================================================
# Helper
# =============================================================================

def _to_gray(image: np.ndarray) -> np.ndarray:
    """
    Ensure input image is uint8 single-channel grayscale.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if image.dtype != np.uint8:
        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    return image


# =============================================================================
# Step 1 — Reference Difference
# =============================================================================

def compute_difference(
    test_image: np.ndarray,
    template_image: np.ndarray
) -> np.ndarray:
    """
    Compute the absolute pixel-level difference between the defective
    PCB image and the corresponding defect-free template.

    Parameters
    ----------
    test_image : np.ndarray
        Preprocessed defective PCB image.

    template_image : np.ndarray
        Preprocessed defect-free template image.

    Returns
    -------
    np.ndarray
        Absolute difference image.
    """

    test_gray = _to_gray(test_image)
    template_gray = _to_gray(template_image)

    # Ensure both images have the same dimensions.
    if template_gray.shape != test_gray.shape:
        template_gray = cv2.resize(
            template_gray,
            (test_gray.shape[1], test_gray.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    difference = cv2.absdiff(
        test_gray,
        template_gray
    )

    return difference


# =============================================================================
# Step 2 — Otsu Thresholding
# =============================================================================

def apply_otsu_threshold(
    difference_image: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Convert the difference image into a binary defect mask using
    Otsu's automatic global thresholding.

    Returns
    -------
    binary : np.ndarray
        Binary defect candidate mask.

    otsu_value : float
        Automatically selected Otsu threshold.
    """

    gray = _to_gray(difference_image)

    # Slight smoothing suppresses isolated high-frequency differences
    # before thresholding.
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    otsu_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Optional dilation.
    if EXTRA_DILATION_ITERATIONS > 0:

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        binary = cv2.dilate(
            binary,
            kernel,
            iterations=EXTRA_DILATION_ITERATIONS
        )

    return binary, float(otsu_value)


# =============================================================================
# Step 3 — Morphological Processing
# =============================================================================

def apply_morphological_processing(
    binary: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply morphological Opening followed by Closing.

    Opening:
        Removes small isolated foreground noise.

    Closing:
        Fills small gaps and improves continuity of detected regions.

    Returns
    -------
    opened : np.ndarray
        Binary mask after morphological opening.

    closed : np.ndarray
        Binary mask after morphological closing.
    """

    binary = _to_gray(binary)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_OPEN_SIZE, KERNEL_OPEN_SIZE)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_CLOSE_SIZE, KERNEL_CLOSE_SIZE)
    )

    # Opening = erosion followed by dilation.
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # Closing = dilation followed by erosion.
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    return opened, closed


# =============================================================================
# Step 4 — Contour Detection
# =============================================================================

def detect_defect_contours(
    cleaned_mask: np.ndarray
) -> List[np.ndarray]:
    """
    Detect independent defect regions from the cleaned binary mask.

    The detected contours represent the final segmentation output
    of Module 2.

    Quantitative feature extraction from these contours is handled
    separately by Module 3.
    """

    cleaned_mask = _to_gray(cleaned_mask)

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Remove extremely small noise.
        if area < MIN_DEFECT_AREA:
            continue

        # Remove abnormally large difference regions.
        if area > MAX_DEFECT_AREA:
            continue

        valid_contours.append(contour)

    return valid_contours


# =============================================================================
# Step 5 — Defect Overlay
# =============================================================================

def create_defect_overlay(
    test_image: np.ndarray,
    contours: List[np.ndarray]
) -> np.ndarray:
    """
    Create a visual representation of the regions detected by Module 2.

    Rectangles are used only for visual localisation in the GUI.
    Their dimensions, areas and coordinates are NOT extracted or
    reported as Module 2 features.

    Feature analysis is handled by Module 3.
    """

    test_gray = _to_gray(test_image)

    overlay = cv2.cvtColor(
        test_gray,
        cv2.COLOR_GRAY2BGR
    )

    for index, contour in enumerate(contours, start=1):

        # Used only to draw the detected region.
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            overlay,
            f"Defect {index}",
            (x, max(y - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return overlay


# =============================================================================
# Complete Module 2 Pipeline
# =============================================================================

def segment_image(
    test_image: np.ndarray,
    template_image: np.ndarray
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    float
]:
    """
    Execute the complete Module 2 segmentation pipeline.

    Returns
    -------
    difference :
        Absolute difference image.

    binary :
        Otsu thresholded binary mask.

    opened :
        Binary mask after morphological opening.

    cleaned_mask :
        Final binary mask after morphological closing.

    contours :
        Detected defect regions.

    otsu_value :
        Automatically selected Otsu threshold.
    """

    # -------------------------------------------------------------------------
    # Step 1 — Absolute Difference
    # -------------------------------------------------------------------------

    difference = compute_difference(
        test_image,
        template_image
    )

    # -------------------------------------------------------------------------
    # Step 2 — Otsu Thresholding
    # -------------------------------------------------------------------------

    binary, otsu_value = apply_otsu_threshold(
        difference
    )

    # -------------------------------------------------------------------------
    # Step 3 — Morphological Opening + Closing
    # -------------------------------------------------------------------------

    opened, cleaned_mask = apply_morphological_processing(
        binary
    )

    # -------------------------------------------------------------------------
    # Step 4 — Contour Detection
    # -------------------------------------------------------------------------

    contours = detect_defect_contours(
        cleaned_mask
    )

    return (
        difference,
        binary,
        opened,
        cleaned_mask,
        contours,
        otsu_value
    )


# =============================================================================
# GUI Adapter
# =============================================================================

def get_segmentation_stages(
    test_image: np.ndarray,
    template_image: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run Module 2 and provide intermediate segmentation results
    and overall segmentation statistics for the GUI.

    Detailed feature extraction for individual defects is handled
    by Module 3.
    """

    (
        difference,
        binary,
        opened,
        cleaned_mask,
        contours,
        otsu_value
    ) = segment_image(
        test_image,
        template_image
    )

    # Stage 7 — visualisation only
    overlay = create_defect_overlay(
        test_image,
        contours
    )

    # -------------------------------------------------------------------------
    # Overall segmentation statistics
    # -------------------------------------------------------------------------

    defect_area_px = int(
        sum(
            cv2.contourArea(contour)
            for contour in contours
        )
    )

    image_height, image_width = cleaned_mask.shape[:2]
    image_area = image_height * image_width

    defect_area_pct = (
        (defect_area_px / image_area) * 100
        if image_area > 0
        else 0.0
    )

    # -------------------------------------------------------------------------
    # GUI stages
    # -------------------------------------------------------------------------

    stages = {
        "test_image": _to_gray(test_image),
        "template_image": _to_gray(template_image),
        "difference": difference,
        "otsu_binary": binary,
        "opening": opened,
        "morphology": cleaned_mask,
        "overlay": overlay
    }

    # -------------------------------------------------------------------------
    # Module 2 overall metrics
    # -------------------------------------------------------------------------

    metrics = {
        "threshold": round(otsu_value, 2),
        "defect_count": len(contours),
        "defect_area_px": defect_area_px,
        "defect_area_pct": round(defect_area_pct, 4),
        "contours": contours
    }

    return stages, metrics
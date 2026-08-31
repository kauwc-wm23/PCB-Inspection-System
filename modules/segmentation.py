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
from typing import Tuple, List, Dict, Any, Optional


# =============================================================================
# Constants / Hyperparameters
# =============================================================================

KERNEL_OPEN_SIZE = 3
KERNEL_CLOSE_SIZE = 3

MIN_DEFECT_AREA = 15
MAX_DEFECT_AREA = 50000

# JPEG recompression creates weak differences across otherwise identical
# PCB-DATASET image pairs. Suppress that measured low-level noise before Otsu.
DIFFERENCE_NOISE_FLOOR = 6

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

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Input image has an unsupported channel count.")
    elif image.ndim != 2:
        raise ValueError("Input image must be 2D grayscale or 3D colour.")

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

    # A resized reference would invalidate annotation coordinates and can
    # create false differences. Require a correctly matched image pair.
    if template_gray.shape != test_gray.shape:
        raise ValueError(
            "Test and template images must have identical dimensions; "
            f"received {test_gray.shape} and {template_gray.shape}."
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
    difference_image: np.ndarray,
    noise_floor: int = DIFFERENCE_NOISE_FLOOR,
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

    if not isinstance(noise_floor, int) or not 0 <= noise_floor <= 255:
        raise ValueError("noise_floor must be an integer from 0 to 255.")

    # Slight smoothing suppresses isolated high-frequency differences
    # before thresholding.
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Discard measured low-level JPEG/recompression differences. Otsu remains
    # responsible for automatically separating the remaining candidate pixels.
    noise_suppressed = blurred.copy()
    noise_suppressed[noise_suppressed < noise_floor] = 0

    otsu_value, binary = cv2.threshold(
        noise_suppressed,
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
    cleaned_mask: np.ndarray,
    min_area: float = MIN_DEFECT_AREA,
    max_area: Optional[float] = MAX_DEFECT_AREA,
) -> List[np.ndarray]:
    """
    Detect independent defect regions from the cleaned binary mask.

    The detected contours represent the final segmentation output
    of Module 2.

    Quantitative feature extraction from these contours is handled
    separately by Module 3.
    """

    cleaned_mask = _to_gray(cleaned_mask)

    if min_area < 0:
        raise ValueError("min_area cannot be negative.")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area must be greater than or equal to min_area.")

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Remove extremely small noise.
        if area < min_area:
            continue

        # Remove abnormally large difference regions.
        if max_area is not None and area > max_area:
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
    template_image: np.ndarray,
    noise_floor: int = DIFFERENCE_NOISE_FLOOR,
    min_defect_area: float = MIN_DEFECT_AREA,
    max_defect_area: Optional[float] = MAX_DEFECT_AREA,
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
        difference,
        noise_floor=noise_floor,
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
        cleaned_mask,
        min_area=min_defect_area,
        max_area=max_defect_area,
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
    template_image: np.ndarray,
    noise_floor: int = DIFFERENCE_NOISE_FLOOR,
    min_defect_area: float = MIN_DEFECT_AREA,
    max_defect_area: Optional[float] = MAX_DEFECT_AREA,
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
        template_image,
        noise_floor=noise_floor,
        min_defect_area=min_defect_area,
        max_defect_area=max_defect_area,
    )

    # Stage 7 — visualisation only
    overlay = create_defect_overlay(
        test_image,
        contours
    )

    # -------------------------------------------------------------------------
    # Overall segmentation statistics
    # -------------------------------------------------------------------------

    valid_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(valid_mask, contours, -1, 255, cv2.FILLED)
    defect_area_px = int(cv2.countNonZero(valid_mask))

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
        "noise_floor": noise_floor,
        "min_defect_area": min_defect_area,
        "max_defect_area": max_defect_area,
        "defect_count": len(contours),
        "defect_area_px": defect_area_px,
        "defect_area_pct": round(defect_area_pct, 4),
        "contours": contours
    }

    return stages, metrics

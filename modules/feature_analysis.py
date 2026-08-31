"""
=============================================================================
Module      : feature_analysis.py
Project     : PCB Defect Inspection System

Description :
    Module 3 — Defect Feature Extraction and Inspection Analysis.

    Processing Pipeline:
        1. Accept cleaned binary mask from Module 2 (segmentation)
        2. Apply Connected Component Analysis
        3. Extract features for each defect region:
           - Defect ID
           - Area (in pixels)
           - Width (bounding box width in pixels)
           - Height (bounding box height in pixels)
           - Bounding box coordinates (x, y, width, height)
           - Centroid location (centre_x, centre_y)
        4. Perform inspection-level analysis:
           - Total defect count
           - Largest defect metrics
           - Average defect size

    This module performs ONLY feature extraction and does NOT:
        - Perform preprocessing or segmentation
        - Generate GUI visualizations
        - Generate reports or PDF files
        - Classify defect types

    The output is structured data (lists/dictionaries) for Module 4 to
    consume and display.

Usage:
    from modules.feature_analysis import analyse_features
    
    # After segmentation from Module 2
    defects, inspection_stats = analyse_features(cleaned_binary_mask)

Author:
    PCB Inspection Team - Feature Analysis Module
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_binary_uint8(mask: np.ndarray) -> np.ndarray:
    """
    Ensure input mask is binary (0 or 255) uint8 format.
    
    Parameters
    ----------
    mask : np.ndarray
        Input mask, may be uint8 or other format.
    
    Returns
    -------
    np.ndarray
        Binary mask guaranteed to be uint8.
    
    Raises
    ------
    ValueError
        If mask is None or invalid.
    """
    if mask is None:
        raise ValueError("Input mask is None.")
    
    if not isinstance(mask, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if mask.size == 0:
        raise ValueError("Input mask is empty.")
    
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Input mask has an unsupported channel count.")
    elif mask.ndim != 2:
        raise ValueError("Input mask must be a 2D or colour image array.")

    # Connected-component analysis treats all non-zero values as foreground.
    # Make that behaviour explicit and return a true 0/255 uint8 mask.
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _reconstruct_mask_from_contours(
    reference_mask: np.ndarray,
    valid_contours: List[np.ndarray]
) -> np.ndarray:
    """
    Reconstruct a binary mask by drawing only valid contours.

    This function creates an empty binary mask matching the reference
    image shape and fills it with only the contours that Module 2
    has already validated. This ensures Module 3 analyses only
    defect regions that passed Module 2's area filtering.

    Parameters
    ----------
    reference_mask : np.ndarray
        Reference binary mask to obtain shape (height, width).
    valid_contours : List[np.ndarray]
        List of contours pre-filtered by Module 2.
        Each contour is a numpy array of shape (N, 1, 2).

    Returns
    -------
    np.ndarray
        Reconstructed binary uint8 mask with only valid contours drawn.
    """
    if not valid_contours:
        # No contours to draw, return empty mask
        return np.zeros(reference_mask.shape, dtype=np.uint8)

    # Create empty mask with same shape as reference
    reconstructed = np.zeros(reference_mask.shape, dtype=np.uint8)

    # Draw all valid contours filled with 255
    cv2.drawContours(
        reconstructed,
        valid_contours,
        -1,  # Draw all contours
        255,  # White fill
        cv2.FILLED  # Filled contour
    )

    return reconstructed


# =============================================================================
# Main Feature Extraction
# =============================================================================

def analyse_features(
    binary_mask: np.ndarray,
    valid_contours: Optional[List[np.ndarray]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract defect features from a cleaned binary segmentation mask.

    This function performs Connected Component Analysis (CCA) on the
    input binary mask and extracts quantitative features for each
    detected defect region.

    Parameters
    ----------
    binary_mask : np.ndarray
        Cleaned binary segmentation mask from Module 2 (segmentation).
        Expected to be uint8 with values 0 (background) and 255 (defect).
        Should be the output of morphological closing in Module 2.

    valid_contours : List[np.ndarray], optional
        Pre-filtered contours from Module 2. If provided, only these
        contours will be analysed. This ensures consistency with Module 2's
        area filtering (MIN_DEFECT_AREA, MAX_DEFECT_AREA).
        If None (default), all components in binary_mask are analysed.
        Default: None.

    Returns
    -------
    defects : List[Dict[str, Any]]
        List of dictionaries, one per detected defect.
        Each dictionary contains:
            {
                "id": int,                          # Defect ID (1-indexed)
                "area": int,                        # Area in pixels
                "width": int,                       # Bounding box width in pixels
                "height": int,                      # Bounding box height in pixels
                "bounding_box": {
                    "x": int,                       # Top-left x coordinate
                    "y": int,                       # Top-left y coordinate
                    "width": int,                   # Width in pixels
                    "height": int                   # Height in pixels
                },
                "location": {
                    "x": float,                     # Centroid x coordinate
                    "y": float                      # Centroid y coordinate
                }
            }

    inspection_stats : Dict[str, Any]
        Inspection-level analysis containing:
            {
                "total_defects": int,               # Total number of defects found
                "largest_defect": Dict or None,     # Defect dict with largest area
                "smallest_defect": Dict or None,    # Defect dict with smallest area
                "average_area": float,              # Average defect area
                "total_defect_area": int            # Total area of all defects
            }

    Raises
    ------
    ValueError
        If binary_mask is None, empty, or invalid.

    Notes
    -----
    - Background (label 0) is automatically skipped.
    - Requires uint8 binary mask (0 and 255 values).
    - Uses cv2.connectedComponentsWithStats for analysis.
    - If valid_contours are provided, reconstructs mask to ensure only
      Module 2 pre-filtered regions are analysed.
    """

    # Ensure input is valid
    mask = _ensure_binary_uint8(binary_mask)

    # =========================================================================
    # Optionally reconstruct mask from valid contours
    # =========================================================================

    if valid_contours is not None:
        # Reconstruct mask to contain only pre-filtered contours from Module 2
        mask = _reconstruct_mask_from_contours(mask, valid_contours)

    # =========================================================================
    # Connected Component Analysis
    # =========================================================================

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
        ltype=cv2.CV_32S
    )

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    defects: List[Dict[str, Any]] = []

    # Iterate through detected components (skip background label 0)
    for label_id in range(1, num_labels):

        # Extract statistics for this component
        # stats format: [x, y, width, height, area]
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        # Extract centroid
        centre_x = float(centroids[label_id, 0])
        centre_y = float(centroids[label_id, 1])

        # Build defect dictionary
        defect = {
            "id": label_id,  # Sequential ID starting from 1
            "area": area,
            "width": width,
            "height": height,
            "bounding_box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
                "location": {
                    "x": centre_x,
                    "y": centre_y
                },
                "centroid": {
                    "x": centre_x,
                    "y": centre_y
                }
        }

        defects.append(defect)

    # =========================================================================
    # Inspection-level Analysis
    # =========================================================================

    inspection_stats: Dict[str, Any] = {
        "total_defects": len(defects),
        "largest_defect": None,
        "smallest_defect": None,
        "average_area": 0.0,
        "total_defect_area": 0
    }

    if len(defects) > 0:

        # Find largest and smallest defects by area
        largest = max(defects, key=lambda d: d["area"])
        smallest = min(defects, key=lambda d: d["area"])

        inspection_stats["largest_defect"] = largest
        inspection_stats["smallest_defect"] = smallest

        # Compute total and average area
        total_area = sum(d["area"] for d in defects)
        inspection_stats["total_defect_area"] = total_area
        inspection_stats["average_area"] = total_area / len(defects)

    return defects, inspection_stats


# =============================================================================
# Additional Utility Functions
# =============================================================================

def get_defect_summary(
    defects: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a high-level summary of detected defects.

    Parameters
    ----------
    defects : List[Dict[str, Any]]
        List of defect dictionaries as returned by analyse_features().

    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing:
            {
                "count": int,
                "total_area": int,
                "average_area": float,
                "max_area": int,
                "min_area": int
            }
    """
    if not defects:
        return {
            "count": 0,
            "total_area": 0,
            "average_area": 0.0,
            "max_area": 0,
            "min_area": 0
        }

    areas = [d["area"] for d in defects]

    return {
        "count": len(defects),
        "total_area": sum(areas),
        "average_area": sum(areas) / len(defects),
        "max_area": max(areas),
        "min_area": min(areas)
    }


def filter_defects_by_area(
    defects: List[Dict[str, Any]],
    min_area: int = 0,
    max_area: int = float('inf')
) -> List[Dict[str, Any]]:
    """
    Filter defects based on area thresholds.

    Parameters
    ----------
    defects : List[Dict[str, Any]]
        List of defect dictionaries as returned by analyse_features().
    min_area : int, optional
        Minimum area in pixels (inclusive). Default is 0.
    max_area : int, optional
        Maximum area in pixels (inclusive). Default is infinity.

    Returns
    -------
    List[Dict[str, Any]]
        Filtered list of defects within the specified area range.
    """
    return [
        d for d in defects
        if min_area <= d["area"] <= max_area
    ]


def get_defect_bounding_region(
    defects: List[Dict[str, Any]],
    image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Compute the overall bounding region encompassing all defects.

    Parameters
    ----------
    defects : List[Dict[str, Any]]
        List of defect dictionaries as returned by analyse_features().
    image_shape : Tuple[int, int]
        Shape of the original image (height, width).

    Returns
    -------
    Tuple[int, int, int, int]
        Overall bounding box as (x, y, width, height).
        Returns (0, 0, 0, 0) if no defects exist.
    """
    if not defects:
        return (0, 0, 0, 0)

    bboxes = [d["bounding_box"] for d in defects]

    if image_shape is None or len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width.")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_shape dimensions must be positive.")

    x_min = max(0, min(bbox["x"] for bbox in bboxes))
    y_min = max(0, min(bbox["y"] for bbox in bboxes))

    x_max = min(image_width, max(bbox["x"] + bbox["width"] for bbox in bboxes))
    y_max = min(image_height, max(bbox["y"] + bbox["height"] for bbox in bboxes))

    width = x_max - x_min
    height = y_max - y_min

    return (x_min, y_min, width, height)

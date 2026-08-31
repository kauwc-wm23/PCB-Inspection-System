"""
=============================================================================
Module      : preprocessing.py
Project     : PCB Inspection System

Description :
    Image Pre-processing and Calibration Module.

    This module prepares raw PCB images before they are passed to
    downstream modules such as segmentation and defect analysis.

    Processing Pipeline:
        1. Load PCB image
        2. Optional image resizing (disabled by default)
        3. Convert image to grayscale
        4. Apply median filtering to suppress potential impulse noise and
           small local intensity variations
        5. Apply CLAHE histogram equalisation for contrast enhancement

    Additional utility:
        get_preprocessing_stages() - returns all intermediate images
        and computed image quality metrics for analysis and reporting.

Usage:
    from modules.preprocessing import preprocess_image
    processed = preprocess_image("path/to/pcb_image.jpg")

    from modules.preprocessing import get_preprocessing_stages
    stages, metrics = get_preprocessing_stages("path/to/pcb_image.jpg")

Author:
    PCB Inspection Team - Image Pre-processing Module
=============================================================================
"""

import cv2
import numpy as np
from math import isfinite
from numbers import Real
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Tuple, Union


# =============================================================================
# Constants
# =============================================================================

MEDIAN_KERNEL_SIZE = 5

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (8, 8)


# =============================================================================
# Image Loading
# =============================================================================

def _validate_image(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate a non-empty grayscale/BGR/BGRA OpenCV image array."""
    if image is None:
        raise ValueError(f"{name} is None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")
    if image.size == 0:
        raise ValueError(f"{name} is empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D grayscale or 3D colour image.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} has an unsupported channel count: {image.shape[2]}.")
    return image


def load_image(image_path: Union[str, PathLike]) -> np.ndarray:
    """
    Load PCB image from file path.

    Parameters
    ----------
    image_path : str
        Path to PCB image.

    Returns
    -------
    np.ndarray
        Loaded BGR image.
    """

    if not isinstance(image_path, (str, PathLike)) or not str(image_path).strip():
        raise ValueError("image_path must be a non-empty filesystem path.")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    return image



# =============================================================================
# Optional Calibration / Standardisation
# =============================================================================

def resize_image(
    image: np.ndarray,
    target_size=None
) -> np.ndarray:
    """
    Resize image only when required.

    PCB-DATASET contains annotation coordinates, therefore automatic resizing
    is avoided to prevent coordinate mismatch during defect evaluation.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    target_size : tuple, optional
        Desired size (width, height).

    Returns
    -------
    np.ndarray
        Resized or original image.
    """

    image = _validate_image(image)

    if target_size is None:
        return image

    if (
        not isinstance(target_size, (tuple, list))
        or len(target_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in target_size)
    ):
        raise ValueError("target_size must be a positive (width, height) pair.")

    resized = cv2.resize(
        image,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    return resized



# =============================================================================
# Grayscale Conversion
# =============================================================================

def convert_grayscale(
    image: np.ndarray
) -> np.ndarray:
    """
    Convert BGR image into grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input colour image.

    Returns
    -------
    np.ndarray
        Grayscale image.
    """

    image = _validate_image(image)

    if image.ndim == 2:
        return image

    if image.shape[2] == 1:
        return image[:, :, 0]

    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY

    gray = cv2.cvtColor(
        image,
        conversion
    )

    return gray



# =============================================================================
# Local Variation Suppression
# =============================================================================

def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = MEDIAN_KERNEL_SIZE
) -> np.ndarray:
    """
    Suppress potential impulse noise and small local intensity variations.

    Median filtering is used because it is well suited to impulse-like noise
    and can preserve edges better than linear smoothing. On an uncorrupted
    real image, however, changed pixels are not automatically noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    kernel_size : int
        Median filter kernel size.

    Returns
    -------
    np.ndarray
        Median-filtered image.
    """

    image = _validate_image(image)
    if image.ndim != 2:
        raise ValueError("Median filtering requires a single-channel image.")

    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Median kernel size must be a positive odd integer."
        )

    filtered = cv2.medianBlur(
        image,
        kernel_size
    )

    return filtered



# =============================================================================
# Contrast Enhancement
# =============================================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.

    CLAHE improves visibility of PCB structural
    patterns and small defect regions under
    uneven illumination conditions.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    clip_limit : float
        Positive CLAHE contrast-limiting threshold.

    tile_grid_size : tuple of int
        Positive ``(columns, rows)`` contextual tile grid.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image.
    """

    image = _validate_image(image)
    if image.ndim != 2:
        raise ValueError("CLAHE requires a single-channel image.")
    if (
        not isinstance(clip_limit, Real)
        or isinstance(clip_limit, bool)
        or not isfinite(float(clip_limit))
        or clip_limit <= 0
    ):
        raise ValueError("CLAHE clip_limit must be a positive finite number.")
    if (
        not isinstance(tile_grid_size, (tuple, list))
        or len(tile_grid_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in tile_grid_size
        )
    ):
        raise ValueError("CLAHE tile_grid_size must be a positive integer pair.")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )

    enhanced = clahe.apply(image)

    return enhanced


def apply_histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """Apply the Module 1 histogram-equalisation stage.

    CLAHE is retained as the project's histogram-equalisation method because
    it limits local contrast amplification while improving local PCB contrast.
    """
    return apply_clahe(image)



# =============================================================================
# Image Quality Metrics (internal helper)
# =============================================================================

def _compute_metrics(image: np.ndarray) -> dict:
    """
    Compute image quality metrics for a single-channel image.

    Metrics
    -------
    mean_brightness : float
        Mean pixel intensity (0-255).  Higher = brighter.

    contrast : float
        Standard deviation of pixel intensities.
        Higher std = more contrast (wider tonal range).

    high_frequency_estimate : float
        Laplacian variance — a proxy for high-frequency image content.
        It responds to noise, compression artefacts, edges, and fine PCB detail,
        so it is not a direct physical-noise measurement.

    noise_estimate : float
        Backward-compatible alias for ``high_frequency_estimate``.

    dynamic_range : int
        Difference between max and min pixel value.
        Higher = better use of available bit depth.

    Parameters
    ----------
    image : np.ndarray
        Single-channel uint8 image.

    Returns
    -------
    dict
        Dictionary of computed metric values.
    """

    # Ensure grayscale
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image

    mean_brightness = float(np.mean(img))
    contrast        = float(np.std(img))

    # Laplacian variance is a high-frequency proxy, not a noise ground truth.
    lap             = cv2.Laplacian(img, cv2.CV_64F)
    high_frequency_estimate = float(lap.var())

    dynamic_range   = int(img.max()) - int(img.min())

    return {
        "mean_brightness": round(mean_brightness, 2),
        "contrast":        round(contrast, 2),
        "high_frequency_estimate": round(high_frequency_estimate, 2),
        # Retained so older sessions and external callers continue to work.
        "noise_estimate":  round(high_frequency_estimate, 2),
        "dynamic_range":   dynamic_range,
    }


# =============================================================================
# Complete Pre-processing Pipeline
# =============================================================================

def preprocess_image(
    image_path: str
) -> np.ndarray:
    """
    Execute complete PCB image preprocessing pipeline.

    Pipeline:
        Load image
        Grayscale conversion
        Median filtering
        CLAHE enhancement

    Parameters
    ----------
    image_path : str
        PCB image path.

    Returns
    -------
    np.ndarray
        Fully processed grayscale PCB image.
    """

    # Step 1: Load image
    image = load_image(image_path)

    # Step 2: Optional resize
    # Disabled by default to preserve PCB-DATASET annotation coordinates.
    image = resize_image(image)

    # Step 3: Convert to grayscale
    gray = convert_grayscale(image)

    # Step 4: Suppress potential impulse noise and small local variations
    filtered = apply_median_filter(gray)

    # Step 5: Improve contrast
    enhanced = apply_histogram_equalisation(filtered)

    return enhanced


# =============================================================================
# Verbose Pipeline  —  returns intermediate stages + quality metrics
# =============================================================================

def get_preprocessing_stages(image_path: str) -> tuple:
    """
    Execute the full preprocessing pipeline and return every
    intermediate image together with per-stage quality metrics.

    This function is intended for analysis, reporting, and UI
    visualisation. It does NOT alter the behaviour of
    preprocess_image() — both functions share the same pipeline logic.

    Parameters
    ----------
    image_path : str
        Path to the raw PCB image file.

    Returns
    -------
    stages : dict
        Ordered dictionary of stage name -> np.ndarray.
        Keys:
            "original"  - original three-channel BGR colour image
            "grayscale" - after colour-to-gray conversion
            "filtered"  - after median filtering
            "enhanced"  - after CLAHE (final output)

    metrics : dict
        Per-stage quality metrics and improvement deltas.
        Keys:
            "original"          - metrics dict for original grayscale
            "grayscale"         - metrics dict for grayscale stage
            "filtered"          - metrics dict for filtered stage
            "enhanced"          - metrics dict for CLAHE output
            "contrast_gain"     - contrast improvement: enhanced vs original (%)
            "high_frequency_reduction" - Laplacian-variance change: gray vs filtered (%)
            "noise_reduction"   - backward-compatible alias for the above
            "dynamic_range_gain"- dynamic range improvement (pixels)
            "processing_time_seconds" - elapsed time for the preprocessing pipeline

    Example
    -------
    >>> stages, metrics = get_preprocessing_stages("image.jpg")
    >>> print(metrics["contrast_gain"])   # e.g. 12.5  (percent improvement)
    """

    # --- Run pipeline stages ---
    started = perf_counter()
    image    = load_image(image_path)
    image    = resize_image(image)
    gray     = convert_grayscale(image)
    filtered = apply_median_filter(gray)
    enhanced = apply_histogram_equalisation(filtered)
    processing_time_seconds = perf_counter() - started

    # Store all stages
    stages = {
        "original":  image.copy(),
        "grayscale": gray.copy(),
        "filtered":  filtered.copy(),
        "enhanced":  enhanced.copy(),
    }

    # --- Compute per-stage metrics ---
    m_orig  = _compute_metrics(gray)
    m_gray  = _compute_metrics(gray)
    m_filt  = _compute_metrics(filtered)
    m_enh   = _compute_metrics(enhanced)

    # --- Compute improvement deltas ---

    # Contrast gain: how much std dev improved (%) after full pipeline
    contrast_gain = 0.0
    if m_orig["contrast"] > 0:
        contrast_gain = round(
            ((m_enh["contrast"] - m_orig["contrast"]) / m_orig["contrast"]) * 100,
            1
        )

    # Percentage change in Laplacian variance. This quantifies filtering of
    # high-frequency content, which can include noise, artefacts, and real edges.
    high_frequency_reduction = 0.0
    if m_orig["high_frequency_estimate"] > 0:
        high_frequency_reduction = round(
            ((m_orig["high_frequency_estimate"] - m_filt["high_frequency_estimate"])
             / m_orig["high_frequency_estimate"]) * 100,
            1
        )

    dynamic_range_gain = m_enh["dynamic_range"] - m_orig["dynamic_range"]

    metrics = {
        "original":           m_orig,
        "grayscale":          m_gray,
        "filtered":           m_filt,
        "enhanced":           m_enh,
        "contrast_gain":      contrast_gain,
        "high_frequency_reduction": high_frequency_reduction,
        # Retained so older sessions and external callers continue to work.
        "noise_reduction":    high_frequency_reduction,
        "dynamic_range_gain": dynamic_range_gain,
        "processing_time_seconds": processing_time_seconds,
    }

    return stages, metrics

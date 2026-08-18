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
        4. Apply median filtering for noise reduction
        5. Apply CLAHE for contrast enhancement

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


# =============================================================================
# Constants
# =============================================================================

MEDIAN_KERNEL_SIZE = 5

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (8, 8)


# =============================================================================
# Image Loading
# =============================================================================

def load_image(image_path: str) -> np.ndarray:
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

    image = cv2.imread(image_path)

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

    DeepPCB contains annotation coordinates,
    therefore automatic resizing is avoided to prevent
    coordinate mismatch during defect evaluation.

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

    if target_size is None:
        return image

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

    if len(image.shape) == 2:
        return image

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray



# =============================================================================
# Noise Reduction
# =============================================================================

def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = MEDIAN_KERNEL_SIZE
) -> np.ndarray:
    """
    Reduce image noise using median filtering.

    Median filtering is applied because it can reduce
    noise while preserving important PCB edge structures.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    kernel_size : int
        Median filter kernel size.

    Returns
    -------
    np.ndarray
        Noise-reduced image.
    """

    if kernel_size % 2 == 0:
        raise ValueError(
            "Median kernel size must be odd."
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
    image: np.ndarray
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

    Returns
    -------
    np.ndarray
        Contrast-enhanced image.
    """

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(image)

    return enhanced



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

    noise_estimate : float
        Laplacian variance — measures high-frequency content.
        Lower value after median filtering indicates noise removed.

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

    # Laplacian variance as noise/sharpness proxy
    lap             = cv2.Laplacian(img, cv2.CV_64F)
    noise_estimate  = float(lap.var())

    dynamic_range   = int(img.max()) - int(img.min())

    return {
        "mean_brightness": round(mean_brightness, 2),
        "contrast":        round(contrast, 2),
        "noise_estimate":  round(noise_estimate, 2),
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
    # Disabled by default to preserve DeepPCB coordinates
    image = resize_image(image)

    # Step 3: Convert to grayscale
    gray = convert_grayscale(image)

    # Step 4: Remove noise
    filtered = apply_median_filter(gray)

    # Step 5: Improve contrast
    enhanced = apply_clahe(filtered)

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
        Ordered dictionary of stage name -> np.ndarray (uint8, grayscale).
        Keys:
            "original"  - BGR image converted to grayscale for fair comparison
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
            "noise_reduction"   - noise reduction: original vs filtered (%)
            "dynamic_range_gain"- dynamic range improvement (pixels)

    Example
    -------
    >>> stages, metrics = get_preprocessing_stages("image.jpg")
    >>> print(metrics["contrast_gain"])   # e.g. 12.5  (percent improvement)
    """

    # --- Run pipeline stages ---
    image    = load_image(image_path)
    image    = resize_image(image)
    gray     = convert_grayscale(image)
    filtered = apply_median_filter(gray)
    enhanced = apply_clahe(filtered)

    # Store all stages
    stages = {
        "original":  gray.copy(),    # original as grayscale for fair comparison
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

    # Noise reduction: Laplacian variance drop from original to filtered (%)
    # Lower Laplacian var after median = less high-freq noise
    noise_reduction = 0.0
    if m_orig["noise_estimate"] > 0:
        noise_reduction = round(
            ((m_orig["noise_estimate"] - m_filt["noise_estimate"])
             / m_orig["noise_estimate"]) * 100,
            1
        )

    dynamic_range_gain = m_enh["dynamic_range"] - m_orig["dynamic_range"]

    metrics = {
        "original":           m_orig,
        "grayscale":          m_gray,
        "filtered":           m_filt,
        "enhanced":           m_enh,
        "contrast_gain":      contrast_gain,
        "noise_reduction":    noise_reduction,
        "dynamic_range_gain": dynamic_range_gain,
    }

    return stages, metrics

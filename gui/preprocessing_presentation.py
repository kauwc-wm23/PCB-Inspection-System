"""Small presentation helpers for preprocessing comparisons in the GUI."""

from collections.abc import Mapping
from numbers import Real
from typing import Dict, Tuple

import cv2
import numpy as np


RoiBounds = Tuple[int, int, int, int]


def create_filtering_change_map(
    grayscale: np.ndarray, median_filtered: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the raw and display-normalized median-filtering change maps.

    The raw map is ``abs(grayscale - median_filtered)``. Normalization is
    applied only to a separate visualization array and never alters either
    preprocessing stage.
    """
    for name, image in (
        ("grayscale", grayscale),
        ("median_filtered", median_filtered),
    ):
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 2:
            raise ValueError(f"{name} must be a non-empty single-channel image.")
    if grayscale.shape != median_filtered.shape:
        raise ValueError("Change-map inputs must have identical dimensions.")
    if grayscale.dtype != median_filtered.dtype:
        raise ValueError("Change-map inputs must have identical data types.")

    absolute_difference = cv2.absdiff(grayscale, median_filtered)
    minimum = float(absolute_difference.min())
    maximum = float(absolute_difference.max())
    if maximum == 0:
        display_map = np.zeros_like(absolute_difference, dtype=np.uint8)
    elif maximum == minimum:
        display_map = np.full(absolute_difference.shape, 255, dtype=np.uint8)
    else:
        display_map = cv2.normalize(
            absolute_difference,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

    return absolute_difference, display_map


def extract_matching_center_rois(
    images: Mapping[str, np.ndarray], fraction: float = 0.32
) -> Tuple[Dict[str, np.ndarray], RoiBounds]:
    """Crop the same proportional central region from equally sized images.

    Returns the copied crops and ``(x_start, y_start, x_end, y_end)`` bounds.
    The proportional crop remains resolution-independent.
    """
    if not images:
        raise ValueError("At least one image is required for ROI comparison.")
    if (
        not isinstance(fraction, Real)
        or isinstance(fraction, bool)
        or not 0 < fraction <= 1
    ):
        raise ValueError("ROI fraction must be a number in the range (0, 1].")

    prepared: Dict[str, np.ndarray] = {}
    expected_shape = None
    for name, image in images.items():
        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
            or image.ndim not in (2, 3)
        ):
            raise ValueError(f"ROI source '{name}' must be a non-empty image array.")
        image_shape = image.shape[:2]
        if expected_shape is None:
            expected_shape = image_shape
        elif image_shape != expected_shape:
            raise ValueError("All ROI source images must have identical height and width.")
        prepared[name] = image

    image_height, image_width = expected_shape
    roi_width = max(1, int(round(image_width * float(fraction))))
    roi_height = max(1, int(round(image_height * float(fraction))))
    x_start = (image_width - roi_width) // 2
    y_start = (image_height - roi_height) // 2
    x_end = x_start + roi_width
    y_end = y_start + roi_height
    bounds = (x_start, y_start, x_end, y_end)

    crops = {
        name: image[y_start:y_end, x_start:x_end].copy()
        for name, image in prepared.items()
    }
    return crops, bounds

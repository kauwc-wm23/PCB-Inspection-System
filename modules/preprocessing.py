
import cv2
import numpy as np
from math import isfinite
from numbers import Real
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Tuple, Union



MEDIAN_KERNEL_SIZE = 5

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (8, 8)



def _validate_image(image: np.ndarray, name: str = "image") -> np.ndarray:
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




def resize_image(
    image: np.ndarray,
    target_size=None
) -> np.ndarray:

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




def convert_grayscale(
    image: np.ndarray
) -> np.ndarray:

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




def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = MEDIAN_KERNEL_SIZE
) -> np.ndarray:

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




def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:

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
    return apply_clahe(image)




def _compute_metrics(image: np.ndarray) -> dict:

    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image

    mean_brightness = float(np.mean(img))
    contrast        = float(np.std(img))

    lap             = cv2.Laplacian(img, cv2.CV_64F)
    high_frequency_estimate = float(lap.var())

    dynamic_range   = int(img.max()) - int(img.min())

    return {
        "mean_brightness": round(mean_brightness, 2),
        "contrast":        round(contrast, 2),
        "high_frequency_estimate": round(high_frequency_estimate, 2),
        "noise_estimate":  round(high_frequency_estimate, 2),
        "dynamic_range":   dynamic_range,
    }



def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    image = resize_image(image)
    gray = convert_grayscale(image)
    filtered = apply_median_filter(gray)
    enhanced = apply_histogram_equalisation(filtered)
    return enhanced


def preprocess_image(
    image_path: str
) -> np.ndarray:
    return preprocess_image_array(load_image(image_path))


def _preprocessing_stages_from_loaded_image(
    image: np.ndarray,
    started: float,
) -> tuple:
    image = resize_image(image)
    gray = convert_grayscale(image)
    filtered = apply_median_filter(gray)
    enhanced = apply_histogram_equalisation(filtered)
    processing_time_seconds = perf_counter() - started

    stages = {
        "original":  image.copy(),
        "grayscale": gray.copy(),
        "filtered":  filtered.copy(),
        "enhanced":  enhanced.copy(),
    }

    m_orig  = _compute_metrics(gray)
    m_gray  = _compute_metrics(gray)
    m_filt  = _compute_metrics(filtered)
    m_enh   = _compute_metrics(enhanced)


    contrast_gain = 0.0
    if m_orig["contrast"] > 0:
        contrast_gain = round(
            ((m_enh["contrast"] - m_orig["contrast"]) / m_orig["contrast"]) * 100,
            1
        )

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
        "noise_reduction":    high_frequency_reduction,
        "dynamic_range_gain": dynamic_range_gain,
        "processing_time_seconds": processing_time_seconds,
    }

    return stages, metrics


def get_preprocessing_stages_from_array(image: np.ndarray) -> tuple:
    return _preprocessing_stages_from_loaded_image(image, perf_counter())


def get_preprocessing_stages(image_path: str) -> tuple:
    started = perf_counter()
    image = load_image(image_path)
    return _preprocessing_stages_from_loaded_image(image, started)

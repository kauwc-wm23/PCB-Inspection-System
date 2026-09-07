
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional



KERNEL_OPEN_SIZE = 3
KERNEL_CLOSE_SIZE = 3
MORPHOLOGY_KERNEL_SHAPE = cv2.MORPH_ELLIPSE
MORPHOLOGY_KERNEL_SHAPE_LABEL = "Ellipse"
OPENING_ITERATIONS = 1
CLOSING_ITERATIONS = 1

MIN_DEFECT_AREA = 30
MAX_DEFECT_AREA = 50000

DIFFERENCE_NOISE_FLOOR = 6

EXTRA_DILATION_ITERATIONS = 0



def _to_gray(image: np.ndarray) -> np.ndarray:

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

    if np.issubdtype(image.dtype, np.bool_):
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    return image



def compute_difference(
    test_image: np.ndarray,
    template_image: np.ndarray
) -> np.ndarray:

    test_gray = _to_gray(test_image)
    template_gray = _to_gray(template_image)

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



def apply_otsu_threshold(
    difference_image: np.ndarray,
    noise_floor: int = DIFFERENCE_NOISE_FLOOR,
) -> Tuple[np.ndarray, float]:

    gray = _to_gray(difference_image)

    if not isinstance(noise_floor, int) or not 0 <= noise_floor <= 255:
        raise ValueError("noise_floor must be an integer from 0 to 255.")

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    noise_suppressed = blurred.copy()
    noise_suppressed[noise_suppressed < noise_floor] = 0

    otsu_value, binary = cv2.threshold(
        noise_suppressed,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

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



def apply_morphological_processing(
    binary: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    binary = _to_gray(binary)

    open_kernel = cv2.getStructuringElement(
        MORPHOLOGY_KERNEL_SHAPE,
        (KERNEL_OPEN_SIZE, KERNEL_OPEN_SIZE)
    )

    close_kernel = cv2.getStructuringElement(
        MORPHOLOGY_KERNEL_SHAPE,
        (KERNEL_CLOSE_SIZE, KERNEL_CLOSE_SIZE)
    )

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=OPENING_ITERATIONS,
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=CLOSING_ITERATIONS,
    )

    return opened, closed


def count_foreground_components(mask: np.ndarray) -> int:

    foreground = (_to_gray(mask) > 0).astype(np.uint8)
    label_count, _ = cv2.connectedComponents(foreground, connectivity=8)
    return max(int(label_count) - 1, 0)


def analyse_morphology_effects(
    pre_morphology: np.ndarray,
    opened: np.ndarray,
    closed: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:

    pre_mask = _to_gray(pre_morphology)
    opened_mask = _to_gray(opened)
    closed_mask = _to_gray(closed)

    if pre_mask.shape != opened_mask.shape or opened_mask.shape != closed_mask.shape:
        raise ValueError("Morphology masks must have identical dimensions.")

    pre_foreground = pre_mask > 0
    opened_foreground = opened_mask > 0
    closed_foreground = closed_mask > 0

    opening_removed = np.where(
        pre_foreground & ~opened_foreground,
        255,
        0,
    ).astype(np.uint8)
    closing_added = np.where(
        closed_foreground & ~opened_foreground,
        255,
        0,
    ).astype(np.uint8)

    opening_before = int(np.count_nonzero(pre_foreground))
    opening_after = int(np.count_nonzero(opened_foreground))
    opening_removed_count = int(cv2.countNonZero(opening_removed))
    opening_components_before = count_foreground_components(pre_mask)
    opening_components_after = count_foreground_components(opened_mask)

    closing_before = opening_after
    closing_after = int(np.count_nonzero(closed_foreground))
    closing_added_count = int(cv2.countNonZero(closing_added))
    closing_components_before = opening_components_after
    closing_components_after = count_foreground_components(closed_mask)

    change_maps = {
        "opening_removed": opening_removed,
        "closing_added": closing_added,
    }
    measurements = {
        "opening": {
            "foreground_before": opening_before,
            "foreground_after": opening_after,
            "pixel_change": opening_removed_count,
            "change_percentage": (
                (opening_removed_count / opening_before) * 100.0
                if opening_before > 0
                else 0.0
            ),
            "components_before": opening_components_before,
            "components_after": opening_components_after,
            "component_difference": (
                opening_components_before - opening_components_after
            ),
        },
        "closing": {
            "foreground_before": closing_before,
            "foreground_after": closing_after,
            "pixel_change": closing_added_count,
            "change_percentage": (
                (closing_added_count / closing_before) * 100.0
                if closing_before > 0
                else 0.0
            ),
            "components_before": closing_components_before,
            "components_after": closing_components_after,
            "component_difference": (
                closing_components_before - closing_components_after
            ),
        },
    }

    return change_maps, measurements


def extract_change_detail_roi(
    change_map: np.ndarray,
    padding_fraction: float = 0.20,
    minimum_image_padding_fraction: float = 0.02,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:

    if not isinstance(padding_fraction, (int, float)) or not 0 <= padding_fraction <= 1:
        raise ValueError("padding_fraction must be between 0 and 1.")
    if (
        not isinstance(minimum_image_padding_fraction, (int, float))
        or not 0 <= minimum_image_padding_fraction <= 1
    ):
        raise ValueError(
            "minimum_image_padding_fraction must be between 0 and 1."
        )

    display_map = _to_gray(change_map)
    changed_y, changed_x = np.nonzero(display_map)
    if changed_x.size == 0:
        return None, None

    image_height, image_width = display_map.shape
    x_start = int(changed_x.min())
    x_end = int(changed_x.max()) + 1
    y_start = int(changed_y.min())
    y_end = int(changed_y.max()) + 1

    change_width = x_end - x_start
    change_height = y_end - y_start
    x_padding = max(
        1,
        int(np.ceil(change_width * padding_fraction)),
        int(np.ceil(image_width * minimum_image_padding_fraction)),
    )
    y_padding = max(
        1,
        int(np.ceil(change_height * padding_fraction)),
        int(np.ceil(image_height * minimum_image_padding_fraction)),
    )

    x_start = max(0, x_start - x_padding)
    y_start = max(0, y_start - y_padding)
    x_end = min(image_width, x_end + x_padding)
    y_end = min(image_height, y_end + y_padding)
    bounds = (x_start, y_start, x_end, y_end)

    return display_map[y_start:y_end, x_start:x_end].copy(), bounds



def detect_defect_contours(
    cleaned_mask: np.ndarray,
    min_area: float = MIN_DEFECT_AREA,
    max_area: Optional[float] = MAX_DEFECT_AREA,
) -> List[np.ndarray]:

    cleaned_mask = _to_gray(cleaned_mask)

    if min_area < 0:
        raise ValueError("min_area cannot be negative.")

    if max_area is not None and max_area < min_area:
        raise ValueError(
            "max_area must be greater than or equal to min_area."
        )

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    image_height, image_width = cleaned_mask.shape[:2]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)


        if (
            x <= 1
            or y <= 1
            or x + w >= image_width - 1
            or y + h >= image_height - 1
        ):
            continue

        valid_contours.append(contour)

    return valid_contours



def create_defect_overlay(
    test_image: np.ndarray,
    contours: List[np.ndarray]
) -> np.ndarray:

    test_gray = _to_gray(test_image)

    overlay = cv2.cvtColor(
        test_gray,
        cv2.COLOR_GRAY2BGR
    )

    for index, contour in enumerate(contours, start=1):

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


    difference = compute_difference(
        test_image,
        template_image
    )


    binary, otsu_value = apply_otsu_threshold(
        difference,
        noise_floor=noise_floor,
    )


    opened, cleaned_mask = apply_morphological_processing(
        binary
    )


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



def get_segmentation_stages(
    test_image: np.ndarray,
    template_image: np.ndarray,
    noise_floor: int = DIFFERENCE_NOISE_FLOOR,
    min_defect_area: float = MIN_DEFECT_AREA,
    max_defect_area: Optional[float] = MAX_DEFECT_AREA,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:

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

    overlay = create_defect_overlay(
        test_image,
        contours
    )

    morphology_change_maps, morphology_analysis = analyse_morphology_effects(
        binary,
        opened,
        cleaned_mask,
    )


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


    stages = {
        "test_image": _to_gray(test_image),
        "template_image": _to_gray(template_image),
        "difference": difference,
        "otsu_binary": binary,
        "opening": opened,
        "morphology": cleaned_mask,
        "opening_removed": morphology_change_maps["opening_removed"],
        "closing_added": morphology_change_maps["closing_added"],
        "overlay": overlay
    }


    metrics = {
        "threshold": round(otsu_value, 2),
        "noise_floor": noise_floor,
        "min_defect_area": min_defect_area,
        "max_defect_area": max_defect_area,
        "defect_count": len(contours),
        "defect_area_px": defect_area_px,
        "defect_area_pct": round(defect_area_pct, 4),
        "contours": contours,
        "morphology_analysis": morphology_analysis,
        "morphology_configuration": {
            "opening": {
                "kernel_shape": MORPHOLOGY_KERNEL_SHAPE_LABEL,
                "kernel_size": (KERNEL_OPEN_SIZE, KERNEL_OPEN_SIZE),
                "iterations": OPENING_ITERATIONS,
            },
            "closing": {
                "kernel_shape": MORPHOLOGY_KERNEL_SHAPE_LABEL,
                "kernel_size": (KERNEL_CLOSE_SIZE, KERNEL_CLOSE_SIZE),
                "iterations": CLOSING_ITERATIONS,
            },
        },
    }
    
    return stages, metrics
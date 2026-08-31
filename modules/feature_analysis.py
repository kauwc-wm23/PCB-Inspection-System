
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional



def _ensure_binary_uint8(mask: np.ndarray) -> np.ndarray:
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

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _reconstruct_mask_from_contours(
    reference_mask: np.ndarray,
    valid_contours: List[np.ndarray]
) -> np.ndarray:
    if not valid_contours:
        return np.zeros(reference_mask.shape, dtype=np.uint8)

    reconstructed = np.zeros(reference_mask.shape, dtype=np.uint8)

    cv2.drawContours(
        reconstructed,
        valid_contours,
        -1,
        255,
        cv2.FILLED
    )

    return reconstructed



def analyse_features(
    binary_mask: np.ndarray,
    valid_contours: Optional[List[np.ndarray]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    mask = _ensure_binary_uint8(binary_mask)


    if valid_contours is not None:
        mask = _reconstruct_mask_from_contours(mask, valid_contours)


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
        ltype=cv2.CV_32S
    )


    defects: List[Dict[str, Any]] = []

    for label_id in range(1, num_labels):

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        centre_x = float(centroids[label_id, 0])
        centre_y = float(centroids[label_id, 1])

        defect = {
            "id": label_id,
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


    inspection_stats: Dict[str, Any] = {
        "total_defects": len(defects),
        "largest_defect": None,
        "smallest_defect": None,
        "average_area": 0.0,
        "total_defect_area": 0
    }

    if len(defects) > 0:

        largest = max(defects, key=lambda d: d["area"])
        smallest = min(defects, key=lambda d: d["area"])

        inspection_stats["largest_defect"] = largest
        inspection_stats["smallest_defect"] = smallest

        total_area = sum(d["area"] for d in defects)
        inspection_stats["total_defect_area"] = total_area
        inspection_stats["average_area"] = total_area / len(defects)

    return defects, inspection_stats



def get_defect_summary(
    defects: List[Dict[str, Any]]
) -> Dict[str, Any]:
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
    return [
        d for d in defects
        if min_area <= d["area"] <= max_area
    ]


def get_defect_bounding_region(
    defects: List[Dict[str, Any]],
    image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
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

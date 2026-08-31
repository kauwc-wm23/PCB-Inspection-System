
from math import isfinite
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_AREA_WEIGHT = 0.60
DEFAULT_WIDTH_WEIGHT = 0.20
DEFAULT_HEIGHT_WEIGHT = 0.20

LOW_MAX_SEVERITY_SCORE = 0.0025
MEDIUM_MAX_SEVERITY_SCORE = 0.0050

SPATIAL_REGIONS = (
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_RIGHT",
)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if isfinite(number):
            return number
    return None


def _positive_number(value: Any) -> Optional[float]:
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _validate_configuration(
    weights: Tuple[Any, Any, Any],
    low_score_threshold: float,
    medium_score_threshold: float,
) -> None:
    if any(_finite_number(weight) is None or weight < 0 for weight in weights):
        raise ValueError("Severity weights must be finite, non-negative numbers.")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Severity weights must sum to 1.0.")
    if (
        _finite_number(low_score_threshold) is None
        or _finite_number(medium_score_threshold) is None
        or low_score_threshold < 0
        or medium_score_threshold < low_score_threshold
    ):
        raise ValueError("Severity thresholds must satisfy 0 <= low <= medium.")


def _spatial_region(centroid_x: float, centroid_y: float, width: int, height: int) -> str:
    horizontal = "LEFT" if centroid_x < width / 2 else "RIGHT"
    vertical = "TOP" if centroid_y < height / 2 else "BOTTOM"
    return f"{vertical}_{horizontal}"


def _severity_level(score: float, low_threshold: float, medium_threshold: float) -> str:
    if score <= low_threshold:
        return "LOW"
    if score <= medium_threshold:
        return "MEDIUM"
    return "HIGH"


def _identifier_sort_key(identifier: Any) -> Tuple[int, Any]:
    numeric_identifier = _finite_number(identifier)
    if numeric_identifier is not None:
        return (0, numeric_identifier)
    return (1, str(identifier))


def assess_defect_severity(
    defects: Sequence[Mapping[str, Any]],
    image_shape: Sequence[int],
    processing_time: Optional[float] = None,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    width_weight: float = DEFAULT_WIDTH_WEIGHT,
    height_weight: float = DEFAULT_HEIGHT_WEIGHT,
    low_score_threshold: float = LOW_MAX_SEVERITY_SCORE,
    medium_score_threshold: float = MEDIUM_MAX_SEVERITY_SCORE,
) -> Dict[str, Any]:
    if defects is None:
        defects = []
    if not isinstance(defects, Sequence) or isinstance(defects, (str, bytes)):
        raise ValueError("defects must be a sequence of Module 3 dictionaries.")
    if image_shape is None or len(image_shape) < 2:
        raise ValueError("image_shape must contain image height and width.")

    try:
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Image height and width must be valid integers.") from error
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image height and width must be greater than zero.")

    weights = (area_weight, width_weight, height_weight)
    _validate_configuration(weights, low_score_threshold, medium_score_threshold)
    weights = tuple(float(weight) for weight in weights)
    low_score_threshold = float(low_score_threshold)
    medium_score_threshold = float(medium_score_threshold)
    image_area = image_width * image_height

    assessed_defects: List[Dict[str, Any]] = []
    invalid_defect_count = 0

    for original_index, defect in enumerate(defects):
        if not isinstance(defect, Mapping):
            invalid_defect_count += 1
            continue

        area = _positive_number(defect.get("area"))
        defect_width = _positive_number(defect.get("width"))
        defect_height = _positive_number(defect.get("height"))
        centroid = defect.get("centroid") or defect.get("location") or {}
        centroid_x = _finite_number(centroid.get("x")) if isinstance(centroid, Mapping) else None
        centroid_y = _finite_number(centroid.get("y")) if isinstance(centroid, Mapping) else None

        if None in (area, defect_width, defect_height, centroid_x, centroid_y):
            invalid_defect_count += 1
            continue

        area_ratio = area / image_area
        width_ratio = defect_width / image_width
        height_ratio = defect_height / image_height
        severity_score = (
            weights[0] * area_ratio
            + weights[1] * width_ratio
            + weights[2] * height_ratio
        )
        region = _spatial_region(centroid_x, centroid_y, image_width, image_height)

        enriched = dict(defect)
        enriched.update(
            {
                "id": defect.get("id", original_index + 1),
                "area": int(round(area)),
                "width": int(round(defect_width)),
                "height": int(round(defect_height)),
                "centroid": {"x": centroid_x, "y": centroid_y},
                "area_ratio": area_ratio,
                "width_ratio": width_ratio,
                "height_ratio": height_ratio,
                "severity_score": severity_score,
                "severity_level": _severity_level(
                    severity_score, low_score_threshold, medium_score_threshold
                ),
                "priority_rank": None,
                "spatial_region": region,
                "_original_index": original_index,
            }
        )
        assessed_defects.append(enriched)

    ranked_defects = sorted(
        assessed_defects,
        key=lambda item: (
            -item["severity_score"],
            -item["area"],
            _identifier_sort_key(item["id"]),
            item["_original_index"],
        ),
    )
    for priority_rank, defect in enumerate(ranked_defects, start=1):
        defect["priority_rank"] = priority_rank

    assessed_defects.sort(key=lambda item: item["_original_index"])
    for defect in assessed_defects:
        defect.pop("_original_index", None)

    spatial_distribution = {region: 0 for region in SPATIAL_REGIONS}
    for defect in assessed_defects:
        spatial_distribution[defect["spatial_region"]] += 1

    total_defects = len(assessed_defects)
    total_defect_area = sum(defect["area"] for defect in assessed_defects)
    valid_processing_time = _finite_number(processing_time)
    if valid_processing_time is not None and valid_processing_time < 0:
        valid_processing_time = None

    if total_defects:
        highest_priority = min(assessed_defects, key=lambda item: item["priority_rank"])
        largest_defect = max(
            assessed_defects,
            key=lambda item: (item["area"], -item["priority_rank"]),
        )
        maximum_concentration = max(spatial_distribution.values())
        concentrated_regions = [
            region
            for region in SPATIAL_REGIONS
            if spatial_distribution[region] == maximum_concentration
        ]
        most_concentrated_region = concentrated_regions[0]
        inspection_status = "DEFECTIVE"
        quality_decision = "FAIL"
        highest_severity_level = highest_priority["severity_level"]
        highest_priority_defect_id = highest_priority["id"]
        highest_priority_defect_region = highest_priority["spatial_region"]
        largest_defect_region = largest_defect["spatial_region"]
    else:
        concentrated_regions = []
        most_concentrated_region = "N/A"
        inspection_status = "NORMAL"
        quality_decision = "PASS"
        highest_severity_level = "N/A"
        highest_priority_defect_id = "N/A"
        highest_priority_defect_region = "N/A"
        largest_defect_region = "N/A"

    coverage = (total_defect_area / image_area) * 100.0
    return {
        "inspection_status": inspection_status,
        "quality_decision": quality_decision,
        "status_label": f"{inspection_status} / {quality_decision}",
        "total_defects": total_defects,
        "total_defect_count": total_defects,
        "invalid_defect_count": invalid_defect_count,
        "defects": assessed_defects,
        "severity": highest_severity_level,
        "highest_severity_level": highest_severity_level,
        "highest_priority_defect_id": highest_priority_defect_id,
        "highest_priority_defect_region": highest_priority_defect_region,
        "spatial_distribution": spatial_distribution,
        "most_concentrated_region": most_concentrated_region,
        "most_concentrated_regions": concentrated_regions,
        "largest_defect_region": largest_defect_region,
        "total_defect_area": total_defect_area,
        "largest_defect_area": max(
            (defect["area"] for defect in assessed_defects), default=0
        ),
        "smallest_defect_area": min(
            (defect["area"] for defect in assessed_defects), default=0
        ),
        "average_defect_area": total_defect_area / total_defects if total_defects else 0.0,
        "defect_coverage_percentage": coverage,
        "defect_density_per_megapixel": total_defects / (image_area / 1_000_000.0),
        "image_area": image_area,
        "image_width": image_width,
        "image_height": image_height,
        "processing_time": valid_processing_time,
        "severity_weights": {
            "area": weights[0],
            "width": weights[1],
            "height": weights[2],
            "basis": "Configurable prototype weights; not scientifically validated.",
        },
        "severity_thresholds": {
            "low_max_score": low_score_threshold,
            "medium_max_score": medium_score_threshold,
            "basis": (
                "Configurable prototype cutoffs informed by 44 detected regions from the "
                "first two images in each of six categories; PCB-DATASET has no severity labels."
            ),
        },
    }


def evaluate_inspection(
    defects: Sequence[Mapping[str, Any]],
    image_shape: Sequence[int],
    processing_time: Optional[float] = None,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    width_weight: float = DEFAULT_WIDTH_WEIGHT,
    height_weight: float = DEFAULT_HEIGHT_WEIGHT,
    low_score_threshold: float = LOW_MAX_SEVERITY_SCORE,
    medium_score_threshold: float = MEDIUM_MAX_SEVERITY_SCORE,
) -> Dict[str, Any]:
    return assess_defect_severity(
        defects=defects,
        image_shape=image_shape,
        processing_time=processing_time,
        area_weight=area_weight,
        width_weight=width_weight,
        height_weight=height_weight,
        low_score_threshold=low_score_threshold,
        medium_score_threshold=medium_score_threshold,
    )

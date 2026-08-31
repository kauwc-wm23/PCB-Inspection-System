from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


CALIBRATION_METHOD = "ORB + mutual ratio matching + RANSAC similarity"

ORB_FEATURE_COUNT = 4000
ORB_FAST_THRESHOLD = 10
REGISTRATION_MAX_DIMENSION = 3200
MATCH_RATIO_THRESHOLD = 0.75

MIN_MATCH_COUNT = 20
MIN_INLIER_COUNT = 15
MIN_INLIER_RATIO = 0.60
RANSAC_REPROJECTION_THRESHOLD_PX = 3.0
MAX_REPROJECTION_ERROR_PX = 2.0
RANSAC_MAX_ITERATIONS = 5000
RANSAC_CONFIDENCE = 0.999

MIN_SCALE = 0.95
MAX_SCALE = 1.05
MAX_ROTATION_DEG = 5.0
MAX_TRANSLATION_FRACTION = 0.05
MIN_OVERLAP_RATIO = 0.90
MAX_ASPECT_RATIO_RELATIVE_DIFFERENCE = 0.05

IDENTITY_MAX_SCALE_DIFFERENCE = 0.001
IDENTITY_MAX_ROTATION_DEG = 0.02
IDENTITY_MAX_TRANSLATION_PX = 0.25


@dataclass(frozen=True)
class CalibrationResult:
    calibrated_test: Optional[np.ndarray]
    valid_region_mask: Optional[np.ndarray]
    transform_matrix: Optional[np.ndarray]
    metadata: Dict[str, Any]

    @property
    def is_verified(self) -> bool:
        return self.metadata.get("status") in {"ALREADY_ALIGNED", "RECTIFIED"}


def _image_dimensions(image: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(image, np.ndarray) and image.ndim >= 2 and image.size > 0:
        return int(image.shape[1]), int(image.shape[0])
    return None, None


def _base_metadata(test_image: Any, reference_image: Any) -> Dict[str, Any]:
    test_width, test_height = _image_dimensions(test_image)
    reference_width, reference_height = _image_dimensions(reference_image)
    return {
        "status": "UNVERIFIED",
        "method": CALIBRATION_METHOD,
        "warp_applied": False,
        "test_width": test_width,
        "test_height": test_height,
        "reference_width": reference_width,
        "reference_height": reference_height,
        "match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": None,
        "reprojection_error": None,
        "scale": None,
        "rotation_deg": None,
        "translation_x": None,
        "translation_y": None,
        "overlap_ratio": None,
        "message": "Calibration has not been evaluated.",
    }


def _validate_image(image: Any, name: str) -> np.ndarray:
    if image is None:
        raise ValueError(f"{name} image is missing.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} image must be a NumPy array.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"{name} image must have nonzero two-dimensional geometry.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} image dimensions must be positive.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} image has an unsupported channel count.")
    return image


def validate_image_pair(
    test_image: np.ndarray,
    reference_image: np.ndarray,
) -> Dict[str, Any]:
    test = _validate_image(test_image, "Test")
    reference = _validate_image(reference_image, "Reference")
    if test.ndim != reference.ndim:
        raise ValueError("Test and reference images must use compatible channel layouts.")
    if test.ndim == 3 and test.shape[2] != reference.shape[2]:
        raise ValueError("Test and reference images must have the same channel count.")

    test_height, test_width = test.shape[:2]
    reference_height, reference_width = reference.shape[:2]
    test_aspect = test_width / test_height
    reference_aspect = reference_width / reference_height
    aspect_difference = abs(test_aspect - reference_aspect) / reference_aspect
    if aspect_difference > MAX_ASPECT_RATIO_RELATIVE_DIFFERENCE:
        raise ValueError(
            "Test and reference aspect ratios differ too much for bounded similarity registration."
        )

    return {
        "test_width": test_width,
        "test_height": test_height,
        "reference_width": reference_width,
        "reference_height": reference_height,
        "dimensions_match": test.shape[:2] == reference.shape[:2],
        "aspect_ratio_relative_difference": aspect_difference,
    }


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _registration_view(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    height, width = gray.shape
    scale = min(1.0, REGISTRATION_MAX_DIMENSION / max(height, width))
    if scale == 1.0:
        return gray, scale
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA), scale


def _ratio_matches(
    query_descriptors: np.ndarray,
    train_descriptors: np.ndarray,
) -> Dict[int, cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    neighbours = matcher.knnMatch(query_descriptors, train_descriptors, k=2)
    filtered = {}
    for candidates in neighbours:
        if len(candidates) != 2:
            continue
        first, second = candidates
        if first.distance < MATCH_RATIO_THRESHOLD * second.distance:
            filtered[first.queryIdx] = first
    return filtered


def _mutual_matches(
    test_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
) -> list:
    forward = _ratio_matches(test_descriptors, reference_descriptors)
    reverse = _ratio_matches(reference_descriptors, test_descriptors)
    return sorted(
        (
            match
            for test_index, match in forward.items()
            if match.trainIdx in reverse
            and reverse[match.trainIdx].trainIdx == test_index
        ),
        key=lambda match: match.distance,
    )


def estimate_alignment(
    test_image: np.ndarray,
    reference_image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    test_gray, test_scale = _registration_view(_to_grayscale(test_image))
    reference_gray, reference_scale = _registration_view(
        _to_grayscale(reference_image)
    )
    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURE_COUNT,
        fastThreshold=ORB_FAST_THRESHOLD,
    )
    test_keypoints, test_descriptors = orb.detectAndCompute(test_gray, None)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference_gray, None
    )
    diagnostics = {
        "match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": None,
        "reprojection_error": None,
    }
    if test_descriptors is None or reference_descriptors is None:
        diagnostics["failure_reason"] = "Insufficient ORB features for registration."
        return None, diagnostics

    matches = _mutual_matches(test_descriptors, reference_descriptors)
    diagnostics["match_count"] = len(matches)
    if len(matches) < MIN_MATCH_COUNT:
        diagnostics["failure_reason"] = (
            f"Only {len(matches)} reliable matches were found; at least "
            f"{MIN_MATCH_COUNT} are required."
        )
        return None, diagnostics

    test_points = np.float32(
        [
            (
                test_keypoints[match.queryIdx].pt[0] / test_scale,
                test_keypoints[match.queryIdx].pt[1] / test_scale,
            )
            for match in matches
        ]
    ).reshape(-1, 1, 2)
    reference_points = np.float32(
        [
            (
                reference_keypoints[match.trainIdx].pt[0] / reference_scale,
                reference_keypoints[match.trainIdx].pt[1] / reference_scale,
            )
            for match in matches
        ]
    ).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        test_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD_PX,
        maxIters=RANSAC_MAX_ITERATIONS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=20,
    )
    if matrix is None or inlier_mask is None:
        diagnostics["failure_reason"] = "RANSAC could not estimate a similarity transform."
        return None, diagnostics

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    diagnostics["inlier_count"] = inlier_count
    diagnostics["inlier_ratio"] = inlier_count / len(matches)
    if inlier_count:
        projected = cv2.transform(test_points[inliers], matrix).reshape(-1, 2)
        residuals = projected - reference_points[inliers].reshape(-1, 2)
        diagnostics["reprojection_error"] = float(
            np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))
        )
    return matrix.astype(np.float64), diagnostics


def _transform_properties(
    matrix: np.ndarray,
    test_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
) -> Dict[str, float]:
    a = float(matrix[0, 0])
    c = float(matrix[1, 0])
    scale = hypot(a, c)
    rotation_deg = degrees(atan2(c, a))
    test_height, test_width = test_shape[:2]
    reference_height, reference_width = reference_shape[:2]
    test_centre = np.float32([[[test_width / 2.0, test_height / 2.0]]])
    mapped_centre = cv2.transform(test_centre, matrix)[0, 0]
    translation_x = float(mapped_centre[0] - reference_width / 2.0)
    translation_y = float(mapped_centre[1] - reference_height / 2.0)

    test_corners = np.float32(
        [
            [0, 0],
            [test_width - 1, 0],
            [test_width - 1, test_height - 1],
            [0, test_height - 1],
        ]
    ).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(test_corners, matrix).reshape(-1, 2)
    reference_corners = np.float32(
        [
            [0, 0],
            [reference_width - 1, 0],
            [reference_width - 1, reference_height - 1],
            [0, reference_height - 1],
        ]
    )
    intersection_area, _ = cv2.intersectConvexConvex(
        transformed_corners.astype(np.float32),
        reference_corners,
    )
    reference_area = float((reference_width - 1) * (reference_height - 1))
    overlap_ratio = max(0.0, float(intersection_area) / reference_area)
    return {
        "scale": scale,
        "rotation_deg": rotation_deg,
        "translation_x": translation_x,
        "translation_y": translation_y,
        "overlap_ratio": min(overlap_ratio, 1.0),
    }


def validate_estimated_transform(
    matrix: np.ndarray,
    diagnostics: Dict[str, Any],
    test_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
) -> Tuple[bool, Dict[str, float], str]:
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        return False, {}, "The estimated transform is missing or non-finite."
    properties = _transform_properties(matrix, test_shape, reference_shape)
    values = tuple(properties.values())
    if not all(isfinite(value) for value in values):
        return False, properties, "The estimated transform diagnostics are non-finite."

    reasons = []
    match_count = int(diagnostics.get("match_count") or 0)
    inlier_count = int(diagnostics.get("inlier_count") or 0)
    inlier_ratio = diagnostics.get("inlier_ratio")
    reprojection_error = diagnostics.get("reprojection_error")
    if match_count < MIN_MATCH_COUNT:
        reasons.append(f"matches {match_count} < {MIN_MATCH_COUNT}")
    if inlier_count < MIN_INLIER_COUNT:
        reasons.append(f"inliers {inlier_count} < {MIN_INLIER_COUNT}")
    if inlier_ratio is None or inlier_ratio < MIN_INLIER_RATIO:
        reasons.append(f"inlier ratio below {MIN_INLIER_RATIO:.2f}")
    if (
        reprojection_error is None
        or not isfinite(float(reprojection_error))
        or reprojection_error > MAX_REPROJECTION_ERROR_PX
    ):
        reasons.append(f"reprojection error above {MAX_REPROJECTION_ERROR_PX:.1f} px")
    if not MIN_SCALE <= properties["scale"] <= MAX_SCALE:
        reasons.append(f"scale outside {MIN_SCALE:.2f}-{MAX_SCALE:.2f}")
    if abs(properties["rotation_deg"]) > MAX_ROTATION_DEG:
        reasons.append(f"rotation exceeds ±{MAX_ROTATION_DEG:.1f}°")
    reference_height, reference_width = reference_shape[:2]
    if abs(properties["translation_x"]) > MAX_TRANSLATION_FRACTION * reference_width:
        reasons.append("horizontal translation exceeds the prototype bound")
    if abs(properties["translation_y"]) > MAX_TRANSLATION_FRACTION * reference_height:
        reasons.append("vertical translation exceeds the prototype bound")
    if properties["overlap_ratio"] < MIN_OVERLAP_RATIO:
        reasons.append(f"overlap below {MIN_OVERLAP_RATIO:.2f}")
    if reasons:
        return False, properties, "; ".join(reasons) + "."
    return True, properties, "Estimated similarity transform passed validation."


def is_identity_alignment(
    properties: Dict[str, float],
    test_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
) -> bool:
    return (
        test_shape[:2] == reference_shape[:2]
        and abs(properties["scale"] - 1.0) <= IDENTITY_MAX_SCALE_DIFFERENCE
        and abs(properties["rotation_deg"]) <= IDENTITY_MAX_ROTATION_DEG
        and abs(properties["translation_x"]) <= IDENTITY_MAX_TRANSLATION_PX
        and abs(properties["translation_y"]) <= IDENTITY_MAX_TRANSLATION_PX
    )


def apply_validated_transform(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    reference_height, reference_width = reference_image.shape[:2]
    output_size = (reference_width, reference_height)
    warped = cv2.warpAffine(
        test_image,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    source_mask = np.full(test_image.shape[:2], 255, dtype=np.uint8)
    valid_region_mask = cv2.warpAffine(
        source_mask,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped[valid_region_mask == 0] = reference_image[valid_region_mask == 0]
    return warped, valid_region_mask


def calibrate_to_reference(
    test_image: np.ndarray,
    reference_image: np.ndarray,
) -> CalibrationResult:
    metadata = _base_metadata(test_image, reference_image)
    try:
        dimensions = validate_image_pair(test_image, reference_image)
    except ValueError as error:
        metadata.update(status="FAILED", message=str(error))
        return CalibrationResult(None, None, None, metadata)

    metadata.update(dimensions)
    matrix, diagnostics = estimate_alignment(test_image, reference_image)
    metadata.update(
        {
            key: diagnostics.get(key)
            for key in (
                "match_count",
                "inlier_count",
                "inlier_ratio",
                "reprojection_error",
            )
        }
    )
    if matrix is None:
        metadata.update(
            status="UNVERIFIED",
            message=diagnostics.get(
                "failure_reason", "A reliable similarity transform could not be estimated."
            ),
        )
        return CalibrationResult(None, None, None, metadata)

    valid, properties, validation_message = validate_estimated_transform(
        matrix,
        diagnostics,
        test_image.shape,
        reference_image.shape,
    )
    metadata.update(properties)
    if not valid:
        metadata.update(status="UNVERIFIED", message=validation_message)
        return CalibrationResult(None, None, matrix, metadata)

    if is_identity_alignment(properties, test_image.shape, reference_image.shape):
        valid_region_mask = np.full(test_image.shape[:2], 255, dtype=np.uint8)
        metadata.update(
            status="ALREADY_ALIGNED",
            warp_applied=False,
            message="Already aligned — original test image preserved.",
        )
        return CalibrationResult(test_image, valid_region_mask, matrix, metadata)

    calibrated_test, valid_region_mask = apply_validated_transform(
        test_image,
        reference_image,
        matrix,
    )
    metadata.update(
        status="RECTIFIED",
        warp_applied=True,
        overlap_ratio=float(np.count_nonzero(valid_region_mask))
        / valid_region_mask.size,
        message="Validated similarity transform applied in the reference coordinate system.",
    )
    return CalibrationResult(calibrated_test, valid_region_mask, matrix, metadata)


import argparse
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_paths import discover_dataset_images, find_annotation, find_reference_image
from modules.feature_analysis import analyse_features
from modules.preprocessing import preprocess_image
from modules.segmentation import get_segmentation_stages


BoundingBox = Tuple[int, int, int, int]


def load_pascal_voc_boxes(annotation_path: Path) -> List[BoundingBox]:
    root = ET.parse(annotation_path).getroot()
    boxes: List[BoundingBox] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        values = tuple(int(box.findtext(name, "0")) for name in ("xmin", "ymin", "xmax", "ymax"))
        if values[2] > values[0] and values[3] > values[1]:
            boxes.append(values)
    return boxes


def feature_boxes(defects: Sequence[Dict]) -> List[BoundingBox]:
    boxes: List[BoundingBox] = []
    for defect in defects:
        box = defect.get("bounding_box", {})
        x, y = int(box.get("x", 0)), int(box.get("y", 0))
        width, height = int(box.get("width", 0)), int(box.get("height", 0))
        if width > 0 and height > 0:
            boxes.append((x, y, x + width, y + height))
    return boxes


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    ground_truth: Sequence[BoundingBox],
    predictions: Sequence[BoundingBox],
    iou_threshold: float,
) -> Tuple[int, int, int, List[float]]:
    candidates = sorted(
        (
            (intersection_over_union(gt_box, pred_box), gt_index, pred_index)
            for gt_index, gt_box in enumerate(ground_truth)
            for pred_index, pred_box in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_ground_truth = set()
    matched_predictions = set()
    matched_ious: List[float] = []
    for score, gt_index, pred_index in candidates:
        if score < iou_threshold:
            break
        if gt_index in matched_ground_truth or pred_index in matched_predictions:
            continue
        matched_ground_truth.add(gt_index)
        matched_predictions.add(pred_index)
        matched_ious.append(score)

    true_positives = len(matched_ious)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives, matched_ious


def _empty_totals() -> Dict[str, object]:
    return {"images": 0, "tp": 0, "fp": 0, "fn": 0, "detected_images": 0, "ious": [], "seconds": 0.0}


def evaluate_images(image_paths: Iterable[Path], iou_threshold: float) -> Dict[str, Dict[str, object]]:
    totals: Dict[str, Dict[str, object]] = defaultdict(_empty_totals)

    for image_path in image_paths:
        annotation_path = find_annotation(image_path)
        reference_path = find_reference_image(image_path)
        if annotation_path is None or reference_path is None:
            print(f"[skip] Missing annotation/reference for {image_path}")
            continue

        started = time.perf_counter()
        processed_test = preprocess_image(str(image_path))
        processed_reference = preprocess_image(str(reference_path))
        stages, segmentation_metrics = get_segmentation_stages(processed_test, processed_reference)
        defects, _ = analyse_features(stages["morphology"], segmentation_metrics["contours"])
        elapsed = time.perf_counter() - started

        ground_truth = load_pascal_voc_boxes(annotation_path)
        predictions = feature_boxes(defects)
        tp, fp, fn, matched_ious = match_boxes(ground_truth, predictions, iou_threshold)

        category = image_path.parent.name
        stats = totals[category]
        stats["images"] += 1
        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn
        stats["detected_images"] += int(tp > 0)
        stats["ious"].extend(matched_ious)
        stats["seconds"] += elapsed

    return dict(totals)


def _metrics(stats: Dict[str, object]) -> Dict[str, float]:
    tp, fp, fn = int(stats["tp"]), int(stats["fp"]), int(stats["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    images = int(stats["images"])
    ious = stats["ious"]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "detection_rate": int(stats["detected_images"]) / images if images else 0.0,
        "average_seconds": float(stats["seconds"]) / images if images else 0.0,
    }


def print_results(results: Dict[str, Dict[str, object]], iou_threshold: float) -> None:
    print(f"PCB-DATASET bounding-box evaluation (IoU threshold={iou_threshold:.2f})")
    print("Category          Images   TP   FP   FN  Precision  Recall   F1  Mean IoU  Detection  Avg sec")
    overall = _empty_totals()
    for category in sorted(results):
        stats = results[category]
        metrics = _metrics(stats)
        print(
            f"{category:17} {stats['images']:6} {stats['tp']:4} {stats['fp']:4} {stats['fn']:4} "
            f"{metrics['precision']:9.3f} {metrics['recall']:7.3f} {metrics['f1']:5.3f} "
            f"{metrics['mean_iou']:9.3f} {metrics['detection_rate']:9.3f} {metrics['average_seconds']:8.3f}"
        )
        for key in ("images", "tp", "fp", "fn", "detected_images"):
            overall[key] += stats[key]
        overall["ious"].extend(stats["ious"])
        overall["seconds"] += stats["seconds"]

    metrics = _metrics(overall)
    print(
        f"{'OVERALL':17} {overall['images']:6} {overall['tp']:4} {overall['fp']:4} {overall['fn']:4} "
        f"{metrics['precision']:9.3f} {metrics['recall']:7.3f} {metrics['f1']:5.3f} "
        f"{metrics['mean_iou']:9.3f} {metrics['detection_rate']:9.3f} {metrics['average_seconds']:8.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="evaluate one category folder")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of images (0 means all)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    images = discover_dataset_images(args.category)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("No matching dataset images were found.", file=sys.stderr)
        return 1

    results = evaluate_images(images, args.iou_threshold)
    print_results(results, args.iou_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

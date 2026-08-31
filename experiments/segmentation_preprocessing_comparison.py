
import argparse
import csv
import json
import math
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluation import (
    feature_boxes,
    intersection_over_union,
    load_pascal_voc_boxes,
    match_boxes,
)
from modules.dataset_paths import (
    board_id_from_name,
    discover_dataset_images,
    find_annotation,
    find_reference_image,
)
from modules.feature_analysis import analyse_features
from modules.preprocessing import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MEDIAN_KERNEL_SIZE,
    convert_grayscale,
    load_image,
    preprocess_image,
    resize_image,
)
from modules.segmentation import segment_image


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "segmentation"
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_VARIANTS_PER_BOARD = 1
CONDITIONS = ("grayscale_only", "production_preprocessing")


def select_balanced_images(
    categories: Optional[Sequence[str]] = None,
    variants_per_board: int = DEFAULT_VARIANTS_PER_BOARD,
) -> List[Path]:
    available = discover_dataset_images()
    available_categories = sorted({path.parent.name for path in available})
    selected_categories = available_categories if not categories else sorted(set(categories))
    unknown = sorted(set(selected_categories) - set(available_categories))
    if unknown:
        raise ValueError(f"Unknown dataset categories: {', '.join(unknown)}")

    selected: List[Path] = []
    for category in selected_categories:
        board_groups: Dict[str, List[Path]] = defaultdict(list)
        for image_path in discover_dataset_images(category):
            board_groups[board_id_from_name(image_path)].append(image_path)
        for board_id in sorted(board_groups):
            selected.extend(sorted(board_groups[board_id])[:variants_per_board])
    return selected


def _grayscale_only(image_path: Path) -> np.ndarray:
    return convert_grayscale(resize_image(load_image(image_path)))


def _annotation_size(annotation_path: Path) -> Optional[Tuple[int, int]]:
    size = ET.parse(annotation_path).getroot().find("size")
    if size is None:
        return None
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    return (width, height) if width > 0 and height > 0 else None


def _run_condition(
    image_path: Path,
    reference_path: Path,
    condition: str,
) -> Dict:
    started_total = time.perf_counter()
    started_preprocessing = time.perf_counter()
    if condition == "grayscale_only":
        prepared_test = _grayscale_only(image_path)
        prepared_reference = _grayscale_only(reference_path)
    elif condition == "production_preprocessing":
        prepared_test = preprocess_image(str(image_path))
        prepared_reference = preprocess_image(str(reference_path))
    else:
        raise ValueError(f"Unsupported condition: {condition}")
    preprocessing_seconds = time.perf_counter() - started_preprocessing

    started_segmentation = time.perf_counter()
    _, _, _, cleaned_mask, contours, otsu_value = segment_image(
        prepared_test, prepared_reference
    )
    segmentation_seconds = time.perf_counter() - started_segmentation

    started_features = time.perf_counter()
    defects, _ = analyse_features(cleaned_mask, contours)
    feature_seconds = time.perf_counter() - started_features
    return {
        "predictions": feature_boxes(defects),
        "otsu_threshold": otsu_value,
        "preprocessing_seconds": preprocessing_seconds,
        "segmentation_seconds": segmentation_seconds,
        "feature_seconds": feature_seconds,
        "total_seconds": time.perf_counter() - started_total,
        "prepared_shape": prepared_test.shape,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summarize(rows: Sequence[Dict]) -> List[Dict]:
    output: List[Dict] = []
    categories = sorted({row["category"] for row in rows})
    for scope in categories + ["OVERALL"]:
        for condition in CONDITIONS:
            scoped = [
                row
                for row in rows
                if row["condition"] == condition
                and (scope == "OVERALL" or row["category"] == scope)
            ]
            if not scoped:
                continue
            tp = sum(row["tp"] for row in scoped)
            fp = sum(row["fp"] for row in scoped)
            fn = sum(row["fn"] for row in scoped)
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            f1_score = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            output.append(
                {
                    "scope": scope,
                    "condition": condition,
                    "images": len(scoped),
                    "ground_truth_boxes": sum(row["ground_truth_boxes"] for row in scoped),
                    "predicted_boxes": sum(row["predicted_boxes"] for row in scoped),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                    "mean_matched_iou": _safe_ratio(
                        sum(row["matched_iou_sum"] for row in scoped), tp
                    ),
                    "mean_best_ground_truth_iou": _safe_ratio(
                        sum(row["best_ground_truth_iou_sum"] for row in scoped),
                        sum(row["ground_truth_boxes"] for row in scoped),
                    ),
                    "std_per_image_best_ground_truth_iou": statistics.pstdev(
                        row["mean_best_ground_truth_iou"] for row in scoped
                    ),
                    "std_predicted_boxes": statistics.pstdev(
                        row["predicted_boxes"] for row in scoped
                    ),
                    "image_detection_rate": statistics.fmean(
                        row["image_detected"] for row in scoped
                    ),
                    "mean_preprocessing_seconds": statistics.fmean(
                        row["preprocessing_seconds"] for row in scoped
                    ),
                    "mean_segmentation_seconds": statistics.fmean(
                        row["segmentation_seconds"] for row in scoped
                    ),
                    "mean_feature_seconds": statistics.fmean(
                        row["feature_seconds"] for row in scoped
                    ),
                    "mean_total_seconds": statistics.fmean(
                        row["total_seconds"] for row in scoped
                    ),
                }
            )
    return output


def _paired_differences(rows: Sequence[Dict]) -> Tuple[List[Dict], Dict]:
    pairs: Dict[Tuple[str, str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        key = (row["category"], row["board_id"], row["image"])
        pairs[key][row["condition"]] = row

    paired_rows: List[Dict] = []
    for (category, board_id, image_name), conditions in sorted(pairs.items()):
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f"Incomplete condition pair for {image_name}")
        grayscale = conditions["grayscale_only"]
        production = conditions["production_preprocessing"]
        paired_rows.append(
            {
                "category": category,
                "board_id": board_id,
                "image": image_name,
                "best_gt_iou_delta": (
                    production["mean_best_ground_truth_iou"]
                    - grayscale["mean_best_ground_truth_iou"]
                ),
                "predicted_boxes_delta": (
                    production["predicted_boxes"] - grayscale["predicted_boxes"]
                ),
                "true_positives_delta": production["tp"] - grayscale["tp"],
                "false_positives_delta": production["fp"] - grayscale["fp"],
                "total_seconds_delta": (
                    production["total_seconds"] - grayscale["total_seconds"]
                ),
            }
        )

    tolerance = 1e-12
    iou_deltas = [row["best_gt_iou_delta"] for row in paired_rows]
    paired_summary = {
        "images": len(paired_rows),
        "mean_best_gt_iou_delta": statistics.fmean(iou_deltas),
        "median_best_gt_iou_delta": statistics.median(iou_deltas),
        "images_with_higher_best_gt_iou": sum(delta > tolerance for delta in iou_deltas),
        "images_with_lower_best_gt_iou": sum(delta < -tolerance for delta in iou_deltas),
        "images_with_equal_best_gt_iou": sum(abs(delta) <= tolerance for delta in iou_deltas),
        "mean_predicted_boxes_delta": statistics.fmean(
            row["predicted_boxes_delta"] for row in paired_rows
        ),
        "mean_total_seconds_delta": statistics.fmean(
            row["total_seconds_delta"] for row in paired_rows
        ),
    }
    return paired_rows, paired_summary


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_finite(rows: Iterable[Dict]) -> None:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(value):
                raise ValueError(f"Non-finite metric in {key}: {row}")


def _plot_summary(summary: Sequence[Dict], output_path: Path) -> None:
    overall = {
        row["condition"]: row for row in summary if row["scope"] == "OVERALL"
    }
    labels = ["Grayscale only", "Production preprocessing"]
    colours = ["#7a90b0", "#3a86ff"]
    metric_names = (
        "precision",
        "recall",
        "f1",
        "mean_best_ground_truth_iou",
        "image_detection_rate",
    )
    display_names = (
        "Precision",
        "Recall",
        "F1",
        "Mean best GT IoU",
        "Detection rate",
    )
    x_positions = np.arange(len(metric_names))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index, condition in enumerate(CONDITIONS):
        axes[0].bar(
            x_positions + (index - 0.5) * width,
            [overall[condition][name] for name in metric_names],
            width,
            label=labels[index],
            color=colours[index],
        )
    axes[0].set_xticks(x_positions, display_names, rotation=18, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Pascal VOC bounding-box performance (IoU >= 0.5)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        labels,
        [overall[condition]["mean_total_seconds"] for condition in CONDITIONS],
        color=colours,
    )
    axes[1].set_title("Mean end-to-end experiment time per image")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Effect of current preprocessing on real PCB defect segmentation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _print_summary(summary: Sequence[Dict], iou_threshold: float) -> None:
    print(f"PCB-DATASET comparison (bounding-box IoU threshold={iou_threshold:.2f})")
    print(
        "Scope | Condition | Images | TP | FP | FN | Precision | Recall | F1 | "
        "Mean IoU | Mean best GT IoU | Detection | Mean total sec"
    )
    for row in summary:
        print(
            f"{row['scope']} | {row['condition']} | {row['images']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['mean_matched_iou']:.4f} | "
            f"{row['mean_best_ground_truth_iou']:.4f} | "
            f"{row['image_detection_rate']:.4f} | "
            f"{row['mean_total_seconds']:.4f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="*", help="optional category subset")
    parser.add_argument(
        "--variants-per-board",
        type=int,
        default=DEFAULT_VARIANTS_PER_BOARD,
        help="first sorted variants selected per board and category",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="optional deterministic cap after balanced selection (0 means all)",
    )
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if args.variants_per_board <= 0:
        parser.error("--variants-per-board must be positive")
    if args.max_images < 0:
        parser.error("--max-images cannot be negative")
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")

    try:
        image_paths = select_balanced_images(args.categories, args.variants_per_board)
    except ValueError as error:
        parser.error(str(error))
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print("No matching dataset images were found.", file=sys.stderr)
        return 1

    rows: List[Dict] = []
    skipped: List[Dict[str, str]] = []
    for image_index, image_path in enumerate(image_paths):
        annotation_path = find_annotation(image_path)
        reference_path = find_reference_image(image_path)
        if annotation_path is None or reference_path is None:
            skipped.append(
                {
                    "image": str(image_path),
                    "reason": "missing annotation or matching clean reference",
                }
            )
            continue

        source_image = load_image(image_path)
        actual_size = (source_image.shape[1], source_image.shape[0])
        declared_size = _annotation_size(annotation_path)
        if declared_size is not None and declared_size != actual_size:
            skipped.append(
                {
                    "image": str(image_path),
                    "reason": (
                        f"annotation size {declared_size} does not match image {actual_size}"
                    ),
                }
            )
            continue

        ground_truth = load_pascal_voc_boxes(annotation_path)
        if not ground_truth:
            skipped.append(
                {"image": str(image_path), "reason": "annotation has no valid boxes"}
            )
            continue

        condition_order = CONDITIONS if image_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in condition_order:
            result = _run_condition(image_path, reference_path, condition)
            if result["prepared_shape"] != (actual_size[1], actual_size[0]):
                raise AssertionError("Preprocessing changed annotation coordinates.")
            tp, fp, fn, matched_ious = match_boxes(
                ground_truth, result["predictions"], args.iou_threshold
            )
            best_ground_truth_ious = [
                max(
                    (
                        intersection_over_union(ground_truth_box, prediction)
                        for prediction in result["predictions"]
                    ),
                    default=0.0,
                )
                for ground_truth_box in ground_truth
            ]
            rows.append(
                {
                    "category": image_path.parent.name,
                    "board_id": board_id_from_name(image_path),
                    "image": image_path.name,
                    "reference": reference_path.name,
                    "annotation": annotation_path.name,
                    "condition": condition,
                    "iou_threshold": args.iou_threshold,
                    "ground_truth_boxes": len(ground_truth),
                    "predicted_boxes": len(result["predictions"]),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "matched_iou_sum": sum(matched_ious),
                    "mean_matched_iou": (
                        statistics.fmean(matched_ious) if matched_ious else 0.0
                    ),
                    "best_ground_truth_iou_sum": sum(best_ground_truth_ious),
                    "mean_best_ground_truth_iou": statistics.fmean(
                        best_ground_truth_ious
                    ),
                    "image_detected": int(tp > 0),
                    "otsu_threshold": result["otsu_threshold"],
                    "preprocessing_seconds": result["preprocessing_seconds"],
                    "segmentation_seconds": result["segmentation_seconds"],
                    "feature_seconds": result["feature_seconds"],
                    "total_seconds": result["total_seconds"],
                }
            )

    if not rows:
        print("No valid image/annotation/reference triplets were evaluated.", file=sys.stderr)
        return 1
    summary = _summarize(rows)
    paired_rows, paired_summary = _paired_differences(rows)
    _validate_finite(rows)
    _validate_finite(summary)
    _validate_finite(paired_rows)
    _validate_finite((paired_summary,))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "segmentation_comparison_per_image.csv", rows)
    _write_csv(output_dir / "segmentation_comparison_summary.csv", summary)
    _write_csv(output_dir / "segmentation_comparison_paired.csv", paired_rows)
    _write_csv(
        output_dir / "segmentation_comparison_paired_summary.csv",
        [paired_summary],
    )
    _plot_summary(summary, output_dir / "segmentation_comparison.png")

    manifest = {
        "purpose": "Real-data with-vs-without full preprocessing comparison.",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "selection_rule": (
            "First sorted image variant for each category/board combination; "
            "condition order alternates per image."
        ),
        "variants_per_board": args.variants_per_board,
        "selected_images": [str(path.relative_to(PROJECT_ROOT)) for path in image_paths],
        "evaluated_image_count": len(rows) // len(CONDITIONS),
        "categories": sorted({row["category"] for row in rows}),
        "conditions": {
            "grayscale_only": "load -> no resize -> grayscale -> unchanged segmentation",
            "production_preprocessing": (
                "load -> no resize -> grayscale -> 5x5 median -> CLAHE "
                "(clip 2.0, grid 8x8) -> unchanged segmentation"
            ),
        },
        "production_parameters": {
            "median_kernel": MEDIAN_KERNEL_SIZE,
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid": list(CLAHE_TILE_GRID_SIZE),
        },
        "ground_truth": "Pascal VOC object bounding boxes",
        "iou_threshold": args.iou_threshold,
        "metrics": [
            "bounding-box precision",
            "bounding-box recall",
            "bounding-box F1",
            "mean IoU of one-to-one matched boxes",
            "mean best predicted IoU per ground-truth box (continuous diagnostic)",
            "image detection rate",
            "processing time",
        ],
        "pixel_metrics_excluded": (
            "Pascal VOC boxes do not provide pixel masks, so pixel IoU and Dice are invalid."
        ),
        "skipped": skipped,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _print_summary(summary, args.iou_threshold)
    print(
        "\nPaired production-minus-grayscale best-GT-IoU delta: "
        f"mean={paired_summary['mean_best_gt_iou_delta']:.6f}, "
        f"median={paired_summary['median_best_gt_iou_delta']:.6f}; "
        f"higher/lower/equal images="
        f"{paired_summary['images_with_higher_best_gt_iou']}/"
        f"{paired_summary['images_with_lower_best_gt_iou']}/"
        f"{paired_summary['images_with_equal_best_gt_iou']}."
    )
    print(f"\nEvaluated {len(rows) // 2} images; skipped {len(skipped)}.")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

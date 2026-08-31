"""Reproducible evidence for the production median-filter and CLAHE settings.

Synthetic impulse noise is used only in the controlled restoration experiment.
It is never passed to the production PCB inspection or segmentation workflow.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
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

from modules.dataset_paths import discover_reference_images
from modules.preprocessing import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MEDIAN_KERNEL_SIZE,
    apply_clahe,
    apply_median_filter,
    convert_grayscale,
    load_image,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "preprocessing"
DEFAULT_SEED = 2133
DEFAULT_NOISE_DENSITY = 0.02
DEFAULT_TIMING_REPEATS = 3
KERNEL_SIZES = (3, 5, 7)
CLAHE_CONFIGURATIONS = (
    (1.0, (8, 8)),
    (2.0, (8, 8)),
    (3.0, (8, 8)),
    (2.0, (4, 4)),
    (2.0, (16, 16)),
)


def add_salt_and_pepper_noise(
    clean: np.ndarray,
    density: float,
    generator: np.random.Generator,
) -> Tuple[np.ndarray, int, int]:
    """Corrupt exactly ``density`` of pixel locations with salt or pepper.

    Half of the selected locations are assigned 0 and the rest 255. A selected
    location can already contain the assigned value; the documented density is
    therefore the injected-location density, not a claim about changed pixels.
    """
    if clean.ndim != 2 or clean.dtype != np.uint8 or clean.size == 0:
        raise ValueError("Controlled noise requires a non-empty uint8 grayscale image.")
    if not 0 < density < 1:
        raise ValueError("Noise density must be between 0 and 1.")

    selected_count = max(2, int(round(clean.size * density)))
    flat_indices = generator.choice(clean.size, size=selected_count, replace=False)
    pepper_count = selected_count // 2
    salt_count = selected_count - pepper_count
    noisy = clean.copy().reshape(-1)
    noisy[flat_indices[:pepper_count]] = 0
    noisy[flat_indices[pepper_count:]] = 255
    return noisy.reshape(clean.shape), salt_count, pepper_count


def mean_squared_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return full-reference pixel MSE."""
    difference = reference.astype(np.float32) - candidate.astype(np.float32)
    return float(np.mean(difference * difference, dtype=np.float64))


def peak_signal_to_noise_ratio(mse: float) -> Optional[float]:
    """Return finite 8-bit PSNR, or ``None`` for an identical image."""
    if mse < 0 or not math.isfinite(mse):
        raise ValueError("MSE must be finite and non-negative.")
    if mse == 0:
        return None
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Return 3x3 Sobel gradient magnitudes for an edge/detail comparison."""
    horizontal = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(horizontal, vertical)


def gradient_cosine_similarity(
    reference_gradient: np.ndarray, candidate: np.ndarray
) -> Optional[float]:
    """Compare edge-gradient layout with the known clean/reference input.

    This is a structural indicator, not a perceptual quality score. Values near
    one mean that gradient magnitudes have a similar spatial pattern.
    """
    candidate_gradient = gradient_magnitude(candidate)
    numerator = float(
        np.sum(reference_gradient * candidate_gradient, dtype=np.float64)
    )
    reference_energy = float(
        np.sum(reference_gradient * reference_gradient, dtype=np.float64)
    )
    candidate_energy = float(
        np.sum(candidate_gradient * candidate_gradient, dtype=np.float64)
    )
    denominator = math.sqrt(reference_energy * candidate_energy)
    if denominator == 0:
        return None
    return numerator / denominator


def _median_timed(
    image: np.ndarray, kernel_size: int, repeats: int
) -> Tuple[np.ndarray, float]:
    timings: List[float] = []
    output = image
    for _ in range(repeats):
        started = time.perf_counter()
        output = apply_median_filter(image, kernel_size=kernel_size)
        timings.append(time.perf_counter() - started)
    return output, statistics.median(timings)


def _clahe_timed(
    image: np.ndarray,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
    repeats: int,
) -> Tuple[np.ndarray, float]:
    timings: List[float] = []
    output = image
    for _ in range(repeats):
        started = time.perf_counter()
        output = apply_clahe(
            image,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
        timings.append(time.perf_counter() - started)
    return output, statistics.median(timings)


def _finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None]
    return statistics.fmean(finite_values) if finite_values else None


def _write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _kernel_experiment(
    references: Sequence[Path],
    seed: int,
    noise_density: float,
    timing_repeats: int,
    output_dir: Path,
) -> Tuple[List[Dict], List[Dict]]:
    per_image: List[Dict] = []
    example_images: Optional[List[Tuple[str, np.ndarray]]] = None

    for image_index, reference_path in enumerate(references):
        clean = convert_grayscale(load_image(reference_path))
        clean_gradient = gradient_magnitude(clean)
        image_seed = seed + image_index
        noisy, salt_count, pepper_count = add_salt_and_pepper_noise(
            clean,
            noise_density,
            np.random.default_rng(image_seed),
        )

        noisy_mse = mean_squared_error(clean, noisy)
        per_image.append(
            {
                "reference": reference_path.name,
                "seed": image_seed,
                "noise_density": noise_density,
                "salt_locations": salt_count,
                "pepper_locations": pepper_count,
                "method": "Noisy input",
                "kernel_size": "",
                "mse": noisy_mse,
                "psnr_db": peak_signal_to_noise_ratio(noisy_mse),
                "gradient_cosine_similarity": gradient_cosine_similarity(
                    clean_gradient, noisy
                ),
                "processing_seconds": 0.0,
            }
        )

        current_example = [("Known clean", clean), ("2% impulse noise", noisy)]
        for kernel_size in KERNEL_SIZES:
            restored, elapsed = _median_timed(noisy, kernel_size, timing_repeats)
            mse = mean_squared_error(clean, restored)
            per_image.append(
                {
                    "reference": reference_path.name,
                    "seed": image_seed,
                    "noise_density": noise_density,
                    "salt_locations": salt_count,
                    "pepper_locations": pepper_count,
                    "method": f"Median {kernel_size}x{kernel_size}",
                    "kernel_size": kernel_size,
                    "mse": mse,
                    "psnr_db": peak_signal_to_noise_ratio(mse),
                    "gradient_cosine_similarity": gradient_cosine_similarity(
                        clean_gradient, restored
                    ),
                    "processing_seconds": elapsed,
                }
            )
            current_example.append((f"Median {kernel_size}x{kernel_size}", restored))

        if example_images is None:
            example_images = current_example

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in per_image:
        grouped[row["method"]].append(row)

    method_order = ["Noisy input"] + [f"Median {size}x{size}" for size in KERNEL_SIZES]
    summary: List[Dict] = []
    for method in method_order:
        rows = grouped[method]
        summary.append(
            {
                "method": method,
                "kernel_size": rows[0]["kernel_size"],
                "images": len(rows),
                "mean_mse": _finite_mean(row["mse"] for row in rows),
                "mean_psnr_db": _finite_mean(row["psnr_db"] for row in rows),
                "mean_gradient_cosine_similarity": _finite_mean(
                    row["gradient_cosine_similarity"] for row in rows
                ),
                "mean_processing_seconds": _finite_mean(
                    row["processing_seconds"] for row in rows
                ),
            }
        )

    _write_csv(
        output_dir / "kernel_comparison_per_image.csv",
        per_image,
        tuple(per_image[0]),
    )
    _write_csv(
        output_dir / "kernel_comparison_summary.csv",
        summary,
        tuple(summary[0]),
    )
    _plot_kernel_summary(summary, output_dir / "kernel_comparison.png")
    if example_images:
        _plot_image_grid(
            example_images,
            output_dir / "kernel_comparison_example.png",
            "Controlled impulse-noise restoration (display only)",
        )
    return per_image, summary


def _clahe_experiment(
    references: Sequence[Path], timing_repeats: int, output_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    per_image: List[Dict] = []
    example_images: Optional[List[Tuple[str, np.ndarray]]] = None

    for reference_path in references:
        grayscale = convert_grayscale(load_image(reference_path))
        filtered = apply_median_filter(grayscale, kernel_size=MEDIAN_KERNEL_SIZE)
        filtered_gradient = gradient_magnitude(filtered)
        input_contrast = float(np.std(filtered))
        input_dynamic_range = int(filtered.max()) - int(filtered.min())
        current_example = [("Median-filtered input", filtered)]

        for clip_limit, tile_grid in CLAHE_CONFIGURATIONS:
            enhanced, elapsed = _clahe_timed(
                filtered, clip_limit, tile_grid, timing_repeats
            )
            output_contrast = float(np.std(enhanced))
            contrast_gain = (
                ((output_contrast - input_contrast) / input_contrast) * 100.0
                if input_contrast > 0
                else 0.0
            )
            saturation_percentage = (
                np.count_nonzero((enhanced <= 1) | (enhanced >= 254))
                / enhanced.size
                * 100.0
            )
            is_production = (
                clip_limit == CLAHE_CLIP_LIMIT
                and tile_grid == CLAHE_TILE_GRID_SIZE
            )
            label = f"clip={clip_limit:g}, grid={tile_grid[0]}x{tile_grid[1]}"
            per_image.append(
                {
                    "reference": reference_path.name,
                    "setting": label,
                    "production_default": is_production,
                    "clip_limit": clip_limit,
                    "tile_grid": f"{tile_grid[0]}x{tile_grid[1]}",
                    "input_contrast_std": input_contrast,
                    "output_contrast_std": output_contrast,
                    "contrast_gain_percent": contrast_gain,
                    "input_dynamic_range": input_dynamic_range,
                    "output_dynamic_range": int(enhanced.max()) - int(enhanced.min()),
                    "extreme_intensity_percent": float(saturation_percentage),
                    "gradient_cosine_similarity": gradient_cosine_similarity(
                        filtered_gradient, enhanced
                    ),
                    "processing_seconds": elapsed,
                }
            )
            current_example.append((label, enhanced))

        if example_images is None:
            example_images = current_example

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in per_image:
        grouped[row["setting"]].append(row)

    summary: List[Dict] = []
    for clip_limit, tile_grid in CLAHE_CONFIGURATIONS:
        label = f"clip={clip_limit:g}, grid={tile_grid[0]}x{tile_grid[1]}"
        rows = grouped[label]
        summary.append(
            {
                "setting": label,
                "production_default": rows[0]["production_default"],
                "images": len(rows),
                "mean_output_contrast_std": _finite_mean(
                    row["output_contrast_std"] for row in rows
                ),
                "mean_contrast_gain_percent": _finite_mean(
                    row["contrast_gain_percent"] for row in rows
                ),
                "mean_output_dynamic_range": _finite_mean(
                    row["output_dynamic_range"] for row in rows
                ),
                "mean_extreme_intensity_percent": _finite_mean(
                    row["extreme_intensity_percent"] for row in rows
                ),
                "mean_gradient_cosine_similarity": _finite_mean(
                    row["gradient_cosine_similarity"] for row in rows
                ),
                "mean_processing_seconds": _finite_mean(
                    row["processing_seconds"] for row in rows
                ),
            }
        )

    _write_csv(
        output_dir / "clahe_comparison_per_image.csv",
        per_image,
        tuple(per_image[0]),
    )
    _write_csv(
        output_dir / "clahe_comparison_summary.csv",
        summary,
        tuple(summary[0]),
    )
    _plot_clahe_summary(summary, output_dir / "clahe_comparison.png")
    if example_images:
        _plot_image_grid(
            example_images,
            output_dir / "clahe_comparison_example.png",
            "Small CLAHE parameter comparison (display only)",
        )
    return per_image, summary


def _plot_kernel_summary(rows: Sequence[Dict], output_path: Path) -> None:
    labels = [row["method"].replace("Median ", "") for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    values = (
        ("mean_psnr_db", "PSNR (dB)", "higher is better"),
        ("mean_mse", "MSE", "lower is better"),
        (
            "mean_gradient_cosine_similarity",
            "Gradient cosine similarity",
            "higher is better",
        ),
        ("mean_processing_seconds", "Median-filter time (s)", "lower is faster"),
    )
    for axis, (key, title, note) in zip(axes.flat, values):
        axis.bar(labels, [row[key] for row in rows], color="#3a86ff")
        axis.set_title(f"{title} ({note})")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Controlled salt-and-pepper experiment: aggregate over references")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_clahe_summary(rows: Sequence[Dict], output_path: Path) -> None:
    labels = [row["setting"].replace(", ", "\n") for row in rows]
    colors = ["#22c55e" if row["production_default"] else "#3a86ff" for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    values = (
        ("mean_output_contrast_std", "Output contrast (intensity std.)"),
        ("mean_extreme_intensity_percent", "Extreme intensities <=1 or >=254 (%)"),
        ("mean_gradient_cosine_similarity", "Gradient cosine similarity"),
        ("mean_processing_seconds", "CLAHE time (s)"),
    )
    for axis, (key, title) in zip(axes.flat, values):
        axis.bar(labels, [row[key] for row in rows], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=0, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("CLAHE comparison (green = production default)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_image_grid(
    images: Sequence[Tuple[str, np.ndarray]], output_path: Path, title: str
) -> None:
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for axis, (label, image) in zip(np.atleast_1d(axes), images):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _validate_finite(rows: Sequence[Dict], allowed_blank: Sequence[str] = ()) -> None:
    """Reject unexpected NaN/Infinity before results are accepted."""
    for row in rows:
        for key, value in row.items():
            if value is None:
                if key not in allowed_blank:
                    raise ValueError(f"Unexpected blank metric in {key}: {row}")
            elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
                raise ValueError(f"Non-finite metric in {key}: {row}")


def _print_table(title: str, rows: Sequence[Dict], columns: Sequence[str]) -> None:
    print(f"\n{title}")
    print(" | ".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        print(" | ".join(values))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-density", type=float, default=DEFAULT_NOISE_DENSITY)
    parser.add_argument("--timing-repeats", type=int, default=DEFAULT_TIMING_REPEATS)
    parser.add_argument(
        "--reference-limit",
        type=int,
        default=0,
        help="maximum sorted clean references (0 evaluates all)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if not 0 < args.noise_density < 1:
        parser.error("--noise-density must be between 0 and 1")
    if args.timing_repeats <= 0:
        parser.error("--timing-repeats must be positive")
    if args.reference_limit < 0:
        parser.error("--reference-limit cannot be negative")

    references = discover_reference_images()
    if args.reference_limit:
        references = references[: args.reference_limit]
    if not references:
        print("No clean PCB_USED references were found.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_rows, kernel_summary = _kernel_experiment(
        references,
        args.seed,
        args.noise_density,
        args.timing_repeats,
        output_dir,
    )
    clahe_rows, clahe_summary = _clahe_experiment(
        references,
        args.timing_repeats,
        output_dir,
    )
    _validate_finite(kernel_rows, allowed_blank=("psnr_db",))
    _validate_finite(kernel_summary, allowed_blank=("mean_psnr_db",))
    _validate_finite(clahe_rows)
    _validate_finite(clahe_summary)

    first_filtered = apply_median_filter(
        convert_grayscale(load_image(references[0])), MEDIAN_KERNEL_SIZE
    )
    if not np.array_equal(
        apply_clahe(first_filtered),
        apply_clahe(
            first_filtered,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
        ),
    ):
        raise AssertionError("Explicit production CLAHE settings changed default output.")

    manifest = {
        "purpose": "Experimental evidence only; synthetic noise is not used in production.",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "noise_type": "salt-and-pepper impulse noise",
        "injected_location_density": args.noise_density,
        "kernel_sizes": list(KERNEL_SIZES),
        "production_median_kernel": MEDIAN_KERNEL_SIZE,
        "clahe_configurations": [
            {"clip_limit": clip, "tile_grid_size": list(grid)}
            for clip, grid in CLAHE_CONFIGURATIONS
        ],
        "production_clahe": {
            "clip_limit": CLAHE_CLIP_LIMIT,
            "tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
        },
        "timing_repeats_per_operation": args.timing_repeats,
        "reference_images": [str(path.relative_to(PROJECT_ROOT)) for path in references],
        "edge_detail_indicator": (
            "Cosine similarity of 3x3 Sobel gradient magnitudes to the known clean "
            "image (kernel test) or median-filtered input (CLAHE test)."
        ),
        "ssim_included": False,
        "ssim_reason": "Not required; MSE, PSNR, and a declared gradient indicator suffice.",
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _print_table(
        "Median kernel summary",
        kernel_summary,
        (
            "method",
            "mean_mse",
            "mean_psnr_db",
            "mean_gradient_cosine_similarity",
            "mean_processing_seconds",
        ),
    )
    _print_table(
        "CLAHE summary",
        clahe_summary,
        (
            "setting",
            "production_default",
            "mean_output_contrast_std",
            "mean_output_dynamic_range",
            "mean_extreme_intensity_percent",
            "mean_gradient_cosine_similarity",
            "mean_processing_seconds",
        ),
    )
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

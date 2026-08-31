
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

from modules.calibration import calibrate_to_reference
from modules.dataset_paths import PROJECT_ROOT, discover_dataset_images, find_reference_image
from modules.feature_analysis import analyse_features
from modules.inspection_evaluation import assess_defect_severity
from modules.preprocessing import load_image, preprocess_image_array
from modules.reporting import generate_inspection_summary
from modules.segmentation import create_defect_overlay, segment_image


GUI_PATH = PROJECT_ROOT / "gui" / "interface.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _save_image(image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Unable to save output image: {output_path}")
    return output_path


def _resolve_cli_inputs(
    image_path: Optional[str], template_path: Optional[str]
) -> tuple[Path, Path]:
    if image_path:
        test_image = Path(image_path).expanduser().resolve()
    else:
        available_images = discover_dataset_images()
        if not available_images:
            raise FileNotFoundError(
                "No PCB images were found under dataset/images/. Supply one with --image."
            )
        test_image = available_images[0]

    if not test_image.is_file():
        raise FileNotFoundError(f"Test image does not exist: {test_image}")

    if template_path:
        template_image = Path(template_path).expanduser().resolve()
    else:
        template_image = find_reference_image(test_image)
        if template_image is None:
            raise FileNotFoundError(
                "No matching PCB_USED reference was found. Supply one with --template."
            )

    if not template_image.is_file():
        raise FileNotFoundError(f"Template image does not exist: {template_image}")
    return test_image, template_image


def run_cli_inspection(
    image_path: Optional[str] = None,
    template_path: Optional[str] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    test_path, reference_path = _resolve_cli_inputs(image_path, template_path)
    started = time.perf_counter()

    test_image = load_image(test_path)
    reference_image = load_image(reference_path)
    calibration_result = calibrate_to_reference(test_image, reference_image)
    if not calibration_result.is_verified:
        metadata = calibration_result.metadata
        raise ValueError(
            f"Calibration {metadata['status']}: {metadata['message']}"
        )
    processed_test = preprocess_image_array(calibration_result.calibrated_test)
    processed_reference = preprocess_image_array(reference_image)
    _, _, _, cleaned, contours, otsu_value = segment_image(
        processed_test, processed_reference
    )
    defects, analysis_metrics = analyse_features(cleaned, contours)
    processing_time = time.perf_counter() - started
    evaluation = assess_defect_severity(defects, cleaned.shape, processing_time)
    report = generate_inspection_summary(
        test_filename=test_path.name,
        template_filename=reference_path.name,
        processing_time=processing_time,
        defects=defects,
        analysis_metrics=analysis_metrics,
        evaluation_result=evaluation,
    )

    overlay = create_defect_overlay(processed_test, contours)
    _save_image(processed_test, output_dir / "preprocessing" / "preprocessed_result.png")
    _save_image(cleaned, output_dir / "segmentation" / "segmented_mask.png")
    _save_image(overlay, output_dir / "segmentation" / "defect_overlay.png")

    print("PCB Inspection System - CLI Result")
    print(f"Test image: {test_path}")
    print(f"Reference: {reference_path}")
    print(f"Calibration: {calibration_result.metadata['status']}")
    print(f"Calibration method: {calibration_result.metadata['method']}")
    print(f"Warp applied: {calibration_result.metadata['warp_applied']}")
    print(f"Otsu threshold: {otsu_value:.2f}")
    print(f"Status: {evaluation['status_label']}")
    print(f"Total defects: {evaluation['total_defect_count']}")
    print(f"Total defect area: {evaluation['total_defect_area']} pixels")
    print(f"Largest defect area: {evaluation['largest_defect_area']} pixels")
    print(f"Defect coverage: {evaluation['defect_coverage_percentage']:.4f}%")
    print(f"Highest severity: {evaluation['highest_severity_level']}")
    print(f"Highest-priority defect: {evaluation['highest_priority_defect_id']}")
    print(f"Most concentrated region: {evaluation['most_concentrated_region']}")
    print(f"Processing time: {processing_time:.3f} seconds")
    print(report["summary_text"])
    print(f"Outputs: {output_dir.resolve()}")
    return 0


def launch_gui(headless: bool = False, port: Optional[int] = None) -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(GUI_PATH)]
    if headless:
        command.append("--server.headless=true")
    if port is not None:
        command.append(f"--server.port={port}")
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical image-processing PCB inspection")
    parser.add_argument("--cli", action="store_true", help="run once without the Streamlit GUI")
    parser.add_argument("--image", help="defective/test PCB image for CLI mode")
    parser.add_argument("--template", help="matching clean reference image for CLI mode")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated CLI images",
    )
    parser.add_argument("--headless", action="store_true", help="launch Streamlit headlessly")
    parser.add_argument("--port", type=int, help="optional Streamlit server port")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.cli:
            return run_cli_inspection(args.image, args.template, args.output_dir)
        return launch_gui(headless=args.headless, port=args.port)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

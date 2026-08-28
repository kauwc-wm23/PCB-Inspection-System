"""
=============================================================================
File        : main.py
Project     : PCB Inspection System

Description :
    Central controller / entry point for the PCB Inspection System.

    This file orchestrates all processing modules in sequence.
    Currently, only Module 1 (Image Pre-processing and Calibration)
    is implemented. Placeholders are provided for future modules so
    that teammates can integrate their work without restructuring this
    file.

    Module Integration Plan:
        Module 1 - Image Pre-processing and Calibration  [ACTIVE]
            modules/preprocessing.py
            Function: preprocess_image()

        Module 2 - Segmentation                          [PENDING]
            modules/segmentation.py
            Function: segment_image()   (to be implemented)

        Module 3 - Feature Analysis                      [PENDING]
            modules/feature_analysis.py
            Function: analyse_features()  (to be implemented)

        Module 4 - Reporting                             [PENDING]
            modules/reporting.py
            Function: generate_report()   (to be implemented)

Usage:
    python main.py

Author:
    PCB Inspection Team
=============================================================================
"""

# =============================================================================
# Standard library imports
# =============================================================================

import os
import sys

# =============================================================================
# Third-party imports
# =============================================================================

import cv2
import matplotlib.pyplot as plt

# =============================================================================
# Project root setup
# =============================================================================

# Ensure the project root is on sys.path so that all module imports work
# correctly regardless of where this script is launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Module imports
# =============================================================================

# --- Module 1: Image Pre-processing and Calibration (ACTIVE) ---
from modules.preprocessing import preprocess_image

# --- Module 2: Segmentation (PENDING - uncomment when implemented) ---
from modules.segmentation import segment_image

# --- Module 3: Feature Analysis (PENDING - uncomment when implemented) ---
# from modules.feature_analysis import analyse_features

# --- Module 4: Reporting (PENDING - uncomment when implemented) ---
# from modules.reporting import generate_report

# =============================================================================
# Configuration
# =============================================================================

# Input: one PCB image from the DeepPCB training set
IMAGE_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "DeepPCB", "train", "images", "01_PCB__1.jpg"
)

# Output directory for preprocessed results
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")


# =============================================================================
# Helper: display banner
# =============================================================================

def print_banner():
    """Print the system startup banner to the console."""
    print("=" * 50)
    print("  PCB Inspection System")
    print("=" * 50)


# =============================================================================
# Helper: save processed image
# =============================================================================

def save_result(image, filename: str, output_dir: str) -> str:
    """
    Save a processed image to the output directory.

    Parameters
    ----------
    image : np.ndarray
        Image array to save.
    filename : str
        Output filename (e.g. 'preprocessed_result.png').
    output_dir : str
        Directory to save the file into (created if absent).

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    return output_path


# =============================================================================
# Helper: visualise results
# =============================================================================

def display_results(original_bgr, processed):
    """
    Display the original and preprocessed PCB images side-by-side
    using matplotlib.

    Parameters
    ----------
    original_bgr : np.ndarray
        Raw BGR image as loaded by OpenCV.
    processed : np.ndarray
        Final preprocessed grayscale image.
    """

    # Convert BGR -> RGB for correct matplotlib colour display
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle(
        "PCB Inspection System - Module 1: Image Pre-processing Result",
        fontsize=13,
        fontweight="bold",
    )

    # Left panel - Original
    axes[0].imshow(original_rgb)
    axes[0].set_title(
        f"Original PCB Image\n"
        f"{original_bgr.shape[1]} x {original_bgr.shape[0]} px  |  BGR colour",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].axis("off")

    # Right panel - Preprocessed
    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title(
        f"Preprocessed PCB Image\n"
        f"{processed.shape[1]} x {processed.shape[0]} px  |  Grayscale  |  CLAHE enhanced",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main pipeline controller
# =============================================================================

def main():
    """
    Central controller for the PCB Inspection System.

    Runs all active processing modules in sequence and saves results.
    Inactive modules are stubbed out with clear TODO markers so
    teammates can integrate their work directly into this function.
    """

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    print_banner()
    print(f"\n  Image  : {os.path.basename(IMAGE_PATH)}")
    print(f"  Source : {IMAGE_PATH}")
    print()

    # ------------------------------------------------------------------
    # Load the original image for display purposes
    # (preprocessing.py handles the full pipeline internally)
    # ------------------------------------------------------------------
    print("Loading image...")
    original_bgr = cv2.imread(IMAGE_PATH)

    if original_bgr is None:
        print(f"\n[ERROR] Could not load image: {IMAGE_PATH}")
        print("        Check that the dataset folder exists and the path is correct.")
        sys.exit(1)

    print(f"  -> Loaded  : shape={original_bgr.shape}, dtype={original_bgr.dtype}")

    # ------------------------------------------------------------------
    # Module 1: Image Pre-processing and Calibration
    # ------------------------------------------------------------------
    print("\nRunning Module 1 - Image Pre-processing and Calibration...")

    try:
        processed = preprocess_image(IMAGE_PATH)

    except Exception as error:
        print("\n[ERROR] Module 1 failed.")
        print(error)
        sys.exit(1)

    print(f"  -> Output  : shape={processed.shape}, dtype={processed.dtype}")
    print(f"  -> Range   : min={processed.min()}  max={processed.max()}")

    # Save preprocessed result to outputs/preprocessing/
    output_path = save_result(
        processed,
        filename="preprocessed_result.png",
        output_dir=OUTPUT_DIR,
    )
    print(f"  -> Saved   : {output_path}")

    print("\nPreprocessing completed.")

# ------------------------------------------------------------------
    # Module 2: Segmentation [ACTIVE]
    # ------------------------------------------------------------------
    print("\nRunning Module 2 - Segmentation...")
    try:
        binary_mask, contours = segment_image(processed)
        print(f"  -> Output Mask Shape : {binary_mask.shape}")
        print(f"  -> Defect Contours   : {len(contours)} candidate regions detected")
        
        # 保存 Module 2 的输出掩膜图片
        seg_output_dir = os.path.join(PROJECT_ROOT, "outputs", "segmentation")
        seg_output_path = save_result(
            binary_mask,
            filename="segmented_mask.png",
            output_dir=seg_output_dir,
        )
        print(f"  -> Saved Segmented Mask : {seg_output_path}")
        print("Segmentation completed.")

    except Exception as error:
        print("\n[ERROR] Module 2 failed.")
        print(error)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Module 3: Feature Analysis  [PENDING]
    # ------------------------------------------------------------------
    # TODO: Integrate when modules/feature_analysis.py is implemented.
    #
    # print("\nRunning Module 3 - Feature Analysis...")
    # features = analyse_features(segmented)
    # print("Feature analysis completed.")

    # ------------------------------------------------------------------
    # Module 4: Reporting  [PENDING]
    # ------------------------------------------------------------------
    # TODO: Integrate when modules/reporting.py is implemented.
    #
    # print("\nRunning Module 4 - Reporting...")
    # generate_report(features, output_dir=os.path.join(PROJECT_ROOT, "outputs", "reports"))
    # print("Report generated.")

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print("\nDisplaying results...")
    display_results(original_bgr, processed)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  System run complete.")
    print("=" * 50)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()

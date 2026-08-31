"""
=============================================================================
Script      : preprocessing_test.py
Project     : PCB Inspection System

Description :
    Experiment / testing script for Module 1:
    Image Pre-processing and Calibration.

    This script verifies that preprocessing.py works correctly by:
        1. Discovering one PCB image from PCB-DATASET
        2. Running each pipeline step individually
        3. Visualising all intermediate and final results

    Figures produced:
        Figure 1 - Original PCB image (BGR colour)
        Figure 2 - Full pipeline comparison (5-panel)
                   Panel 1 : Original (colour)
                   Panel 2 : Grayscale
                   Panel 3 : Median filtered
                   Panel 4 : CLAHE enhanced (final)
                   Panel 5 : Pixel intensity histogram comparison

    These figures are intended for Chapter 4 of the project report.

Author:
    PCB Inspection Team - Image Pre-processing Module
=============================================================================
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that
# "from modules.preprocessing import ..." works regardless of where
# this script is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the main pipeline entry point (used for the quick overview figure)
from modules.preprocessing import preprocess_image

# Import individual step functions to build intermediate visualisations
from modules.preprocessing import (
    load_image,
    resize_image,
    convert_grayscale,
    apply_median_filter,
    apply_clahe,
)
from modules.dataset_paths import discover_dataset_images

# =============================================================================
# Configuration
# =============================================================================

# Discover a real sample recursively instead of hardcoding a category/filename.
_dataset_images = discover_dataset_images()
if not _dataset_images:
    raise FileNotFoundError("No PCB images found under dataset/images/.")
IMAGE_PATH = str(_dataset_images[0])

# Directory where result figures will be saved (outputs/)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Step 1  -  Run the pipeline step-by-step (intermediate results)
# =============================================================================

print("=" * 60)
print("  PCB Inspection System - Preprocessing Test")
print("=" * 60)
print(f"\n[test] Image path : {IMAGE_PATH}")

# --- Step 1: Load raw BGR image ---
original_bgr = load_image(IMAGE_PATH)
print(f"[test] Loaded      : shape={original_bgr.shape}, dtype={original_bgr.dtype}")

# Convert BGR -> RGB for correct matplotlib display
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

# --- Step 2: Optional resize (disabled - target_size=None preserves coords) ---
image_resized = resize_image(original_bgr, target_size=None)
print(
    f"[test] After resize: shape={image_resized.shape}  (unchanged - resize disabled)"
)

# --- Step 3: Grayscale conversion ---
gray = convert_grayscale(image_resized)
print(f"[test] Grayscale   : shape={gray.shape}, dtype={gray.dtype}")

# --- Step 4: Median filtering (potential impulse-noise suppression) ---
filtered = apply_median_filter(gray)
print(f"[test] Median      : shape={filtered.shape}, dtype={filtered.dtype}")

# --- Step 5: CLAHE contrast enhancement ---
enhanced = apply_clahe(filtered)
print(f"[test] CLAHE       : shape={enhanced.shape}, dtype={enhanced.dtype}")

print("\n[test] All pipeline steps completed successfully.\n")


# =============================================================================
# Step 2  -  Run the full pipeline via preprocess_image() (sanity check)
# =============================================================================

processed = preprocess_image(IMAGE_PATH)
print(f"[test] preprocess_image() output shape : {processed.shape}")
print(f"[test] Output dtype                    : {processed.dtype}")
print(f"[test] Pixel range  min={processed.min()}  max={processed.max()}\n")


# =============================================================================
# Figure 1  -  Original PCB Image
# =============================================================================

fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.imshow(original_rgb)
ax1.set_title(
    "Figure 1 - Original PCB Image\n"
    f"Resolution: {original_bgr.shape[1]} x {original_bgr.shape[0]} px",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax1.axis("off")

fig1.suptitle(
    "PCB Inspection System - Module 1: Image Pre-processing",
    fontsize=11,
    color="grey",
    y=0.02,
)
fig1.tight_layout()

# Save Figure 1
fig1_path = os.path.join(OUTPUT_DIR, "fig1_original_pcb.png")
fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
print(f"[test] Figure 1 saved -> {fig1_path}")


# =============================================================================
# Figure 2  -  Full Pipeline Comparison (5 panels)
# =============================================================================

fig2 = plt.figure(figsize=(18, 10))
fig2.suptitle(
    "Figure 2 - Image Pre-processing Pipeline\n" "PCB Inspection System - Module 1",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# Use GridSpec for flexible layout:
#   Top row  : 4 image panels (original, grayscale, median, CLAHE)
#   Bottom   : full-width histogram comparison
gs = gridspec.GridSpec(
    2, 4, figure=fig2, hspace=0.45, wspace=0.3, height_ratios=[3, 1.8]
)

# ---- Panel 1: Original (colour) ----
ax_orig = fig2.add_subplot(gs[0, 0])
ax_orig.imshow(original_rgb)
ax_orig.set_title("Step 1\nOriginal (BGR)", fontsize=10, fontweight="bold")
ax_orig.axis("off")
ax_orig.text(
    0.5,
    -0.06,
    f"{original_bgr.shape[1]}x{original_bgr.shape[0]} px  |  3 channels",
    transform=ax_orig.transAxes,
    ha="center",
    fontsize=7.5,
    color="grey",
)

# ---- Panel 2: Grayscale ----
ax_gray = fig2.add_subplot(gs[0, 1])
ax_gray.imshow(gray, cmap="gray")
ax_gray.set_title("Step 3\nGrayscale Conversion", fontsize=10, fontweight="bold")
ax_gray.axis("off")
ax_gray.text(
    0.5,
    -0.06,
    f"Mean pixel: {gray.mean():.1f}  |  1 channel",
    transform=ax_gray.transAxes,
    ha="center",
    fontsize=7.5,
    color="grey",
)

# ---- Panel 3: Median filtered ----
ax_med = fig2.add_subplot(gs[0, 2])
ax_med.imshow(filtered, cmap="gray")
ax_med.set_title("Step 4\nMedian Filter (5x5)", fontsize=10, fontweight="bold")
ax_med.axis("off")
ax_med.text(
    0.5,
    -0.06,
    f"Mean pixel: {filtered.mean():.1f}  |  5x5 median output",
    transform=ax_med.transAxes,
    ha="center",
    fontsize=7.5,
    color="grey",
)

# ---- Panel 4: CLAHE enhanced (final output) ----
ax_clahe = fig2.add_subplot(gs[0, 3])
ax_clahe.imshow(enhanced, cmap="gray")
ax_clahe.set_title("Step 5\nCLAHE Enhancement (final)", fontsize=10, fontweight="bold")
ax_clahe.axis("off")
ax_clahe.text(
    0.5,
    -0.06,
    f"Mean pixel: {enhanced.mean():.1f}  |  contrast enhanced",
    transform=ax_clahe.transAxes,
    ha="center",
    fontsize=7.5,
    color="grey",
)

# ---- Bottom: Pixel Intensity Histogram Comparison ----
ax_hist = fig2.add_subplot(gs[1, :])  # Span all 4 columns

ax_hist.hist(
    gray.ravel(),
    bins=256,
    range=(0, 255),
    color="#4C72B0",
    alpha=0.6,
    label="Grayscale",
    density=True,
)
ax_hist.hist(
    filtered.ravel(),
    bins=256,
    range=(0, 255),
    color="#DD8452",
    alpha=0.6,
    label="After Median Filter",
    density=True,
)
ax_hist.hist(
    enhanced.ravel(),
    bins=256,
    range=(0, 255),
    color="#55A868",
    alpha=0.7,
    label="After CLAHE (final)",
    density=True,
)

ax_hist.set_title(
    "Pixel Intensity Distribution — Grayscale vs Median Filtered vs CLAHE",
    fontsize=10,
    fontweight="bold",
)
ax_hist.set_xlabel("Pixel Intensity (0 = black, 255 = white)", fontsize=9)
ax_hist.set_ylabel("Normalised Frequency", fontsize=9)
ax_hist.set_xlim([0, 255])
ax_hist.legend(fontsize=9)
ax_hist.grid(axis="y", linestyle="--", alpha=0.4)

# Save Figure 2
fig2_path = os.path.join(OUTPUT_DIR, "fig2_pipeline_comparison.png")
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"[test] Figure 2 saved -> {fig2_path}")


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("  Test Summary")
print("=" * 60)
print(f"  Image tested   : {os.path.basename(IMAGE_PATH)}")
print(f"  Original shape : {original_bgr.shape}")
print(f"  Final shape    : {processed.shape}")
print(f"  Output figures : outputs/fig1_original_pcb.png")
print(f"                   outputs/fig2_pipeline_comparison.png")
print("=" * 60)
if "agg" not in plt.get_backend().lower():
    print("\n[test] Displaying figures... (close windows to exit)\n")
    plt.show()
else:
    print("\n[test] Non-interactive backend detected; figures were saved without opening windows.\n")

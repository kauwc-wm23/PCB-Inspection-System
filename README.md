# PCB Defect Inspection System

BMDS2133 Mode B prototype using classical image processing only.

## Processing flow

1. Module 1 — grayscale conversion, 5×5 median filtering, and CLAHE histogram equalisation.
2. Module 2 — clean-reference difference, low-level JPEG noise suppression, Otsu thresholding, morphological opening/closing, and contour detection.
3. Module 3 — connected-component area, dimensions, bounding box, and centroid extraction.
4. Module 5 — per-defect severity scoring, priority ranking, and quadrant-based spatial analysis.
5. Module 4 — Streamlit inspection interface and report presentation.

The project uses `dataset/images/<category>/` for test images,
`dataset/PCB_USED/` for clean references, and Pascal VOC XML files in
`dataset/Annotations/<category>/`. A test image's leading board ID (for
example, `01` in `01_missing_hole_01.jpg`) selects `PCB_USED/01.JPG`.

## Setup and run

```powershell
pip install -r requirement.txt
python main.py
```

`python main.py` launches the GUI. The CLI uses the first discovered dataset
image by default and can also accept explicit paths:

```powershell
python main.py --cli
python main.py --cli --image dataset/images/Short/01_short_01.jpg
```

## Experiments

```powershell
python experiments/preprocessing_test.py
python experiments/segmentation_test.py --category Missing_hole
python experiments/module5_test.py
python experiments/gui_integration_test.py
python experiments/evaluation.py --limit 12 --iou-threshold 0.5
python experiments/preprocessing_evaluation.py
python experiments/segmentation_preprocessing_comparison.py
```

The evaluation script matches Module 3 boxes to the dataset's real Pascal VOC
boxes and reports precision, recall, F1, mean matched IoU, detection rate, and
average processing time. Dataset category names are grouping metadata only;
the application does not perform defect-type classification.

`preprocessing_evaluation.py` uses all clean `PCB_USED` references by default.
It injects exactly 2% salt-and-pepper locations with seed `2133`, compares
3×3, 5×5, and 7×7 production median-filter calls against the known clean
images, and evaluates five small CLAHE configurations around the production
default. The synthetic corruption is experimental only and is never passed to
the inspection workflow. MSE, PSNR, timing, and a declared Sobel-gradient
similarity indicator are calculated from the images; SSIM is intentionally not
required.

`segmentation_preprocessing_comparison.py` deterministically selects the first
sorted image for each board/category combination (60 real defect images with
the current dataset). It compares grayscale-only input with the complete
production preprocessing path while keeping reference matching, segmentation,
feature extraction, IoU threshold (`0.5`), and box matching identical. Pascal
VOC supplies bounding boxes rather than pixel masks, so the experiment reports
box precision, recall, F1, matched IoU, best-overlap IoU, image detection rate,
and timing; it does not report pixel Dice or pixel IoU.

Both scripts save per-image CSV files, aggregate CSV summaries, concise plots,
example panels, and reproducibility manifests under `outputs/experiments/`.
Generated files do not affect application runtime and are ignored by Git.

The GUI's filtering change map is the display-normalized absolute difference
between grayscale and median-filtered images. It shows where filtering changed
intensities, not a noise ground-truth map: bright areas can include potential
noise, compression artefacts, PCB edges, and fine structural detail. Likewise,
Laplacian variance is presented only as a high-frequency estimate.

## Prototype severity scoring

PCB-DATASET provides defect categories and locations but no official severity
labels. Module 5 therefore assesses each detected region relative to the image:

`severity score = 0.60 × area ratio + 0.20 × width ratio + 0.20 × height ratio`

The default score thresholds are LOW at or below `0.0025`, MEDIUM above
`0.0025` and at or below `0.0050`, and HIGH above `0.0050`. The weights and
thresholds are configurable. They are transparent prototype assumptions based
on a small cross-category detection audit, not scientifically validated
manufacturing limits.

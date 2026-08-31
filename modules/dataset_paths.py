
from pathlib import Path
from typing import Iterable, List, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGE_ROOT = DATASET_ROOT / "images"
ANNOTATION_ROOT = DATASET_ROOT / "Annotations"
PCB_USED_ROOT = DATASET_ROOT / "PCB_USED"
ROTATION_ROOT = DATASET_ROOT / "rotation"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def discover_dataset_images(category: Optional[str] = None) -> List[Path]:
    search_root = IMAGE_ROOT / category if category else IMAGE_ROOT
    if not search_root.is_dir():
        return []
    return sorted(path for path in search_root.rglob("*") if _is_supported_image(path))


def discover_reference_images() -> List[Path]:
    if not PCB_USED_ROOT.is_dir():
        return []
    return sorted(path for path in PCB_USED_ROOT.iterdir() if _is_supported_image(path))


def board_id_from_name(image: Union[str, Path]) -> str:
    stem = Path(image).stem
    return stem.split("_", 1)[0]


def find_reference_image(image: Union[str, Path]) -> Optional[Path]:
    board_id = board_id_from_name(image).casefold()
    for reference in discover_reference_images():
        if reference.stem.casefold() == board_id:
            return reference
    return None


def find_annotation(image: Union[str, Path]) -> Optional[Path]:
    image_path = Path(image)
    stem = image_path.stem

    category_candidates: Iterable[Path]
    direct_category = ANNOTATION_ROOT / image_path.parent.name
    if direct_category.is_dir():
        category_candidates = [direct_category]
    elif ANNOTATION_ROOT.is_dir():
        category_candidates = sorted(path for path in ANNOTATION_ROOT.iterdir() if path.is_dir())
    else:
        return None

    for category_root in category_candidates:
        annotation = category_root / f"{stem}.xml"
        if annotation.is_file():
            return annotation
    return None

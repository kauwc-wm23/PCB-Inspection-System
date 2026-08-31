"""Focused regression checks for Module 5 severity and spatial assessment."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.inspection_evaluation import assess_defect_severity


def _defect(defect_id, area, width, height, x, y):
    return {
        "id": defect_id,
        "area": area,
        "width": width,
        "height": height,
        "bounding_box": {"x": 0, "y": 0, "width": width, "height": height},
        "centroid": {"x": x, "y": y},
    }


def main() -> int:
    defects = [
        _defect(2, 1000, 10, 10, 100, 100),
        _defect(1, 2000, 15, 15, 900, 900),
        _defect(4, 1000, 10, 10, 900, 100),
        _defect(3, 100, 1, 1, 100, 900),
    ]
    result = assess_defect_severity(defects, (1000, 1000))
    by_id = {defect["id"]: defect for defect in result["defects"]}

    assert abs(by_id[2]["severity_score"] - 0.0046) < 1e-12
    assert by_id[2]["severity_level"] == "MEDIUM"
    assert by_id[1]["severity_level"] == "HIGH"
    assert by_id[3]["severity_level"] == "LOW"
    assert [by_id[index]["priority_rank"] for index in (1, 2, 4, 3)] == [1, 2, 3, 4]
    assert result["spatial_distribution"] == {
        "TOP_LEFT": 1,
        "TOP_RIGHT": 1,
        "BOTTOM_LEFT": 1,
        "BOTTOM_RIGHT": 1,
    }
    assert result["most_concentrated_region"] == "TOP_LEFT"
    assert result["highest_priority_defect_region"] == "BOTTOM_RIGHT"
    assert result["largest_defect_region"] == "BOTTOM_RIGHT"

    boundary = assess_defect_severity(
        [_defect(1, 1, 1, 1, 500, 500)], (1000, 1000)
    )
    assert boundary["defects"][0]["spatial_region"] == "BOTTOM_RIGHT"

    empty = assess_defect_severity([], (1000, 1000))
    assert empty["total_defects"] == 0
    assert empty["highest_severity_level"] == "N/A"
    assert empty["highest_priority_defect_id"] == "N/A"
    assert empty["most_concentrated_region"] == "N/A"

    invalid = assess_defect_severity([{"id": 1}], (1000, 1000))
    assert invalid["total_defects"] == 0
    assert invalid["invalid_defect_count"] == 1

    for invalid_shape in ((0, 1000), (1000, 0), ("bad", 1000)):
        try:
            assess_defect_severity([], invalid_shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for image shape {invalid_shape!r}")

    try:
        assess_defect_severity([], (1000, 1000), area_weight=0.7)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when severity weights do not sum to 1.0")

    print("Module 5 severity, ranking, spatial, validation, and zero-defect checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

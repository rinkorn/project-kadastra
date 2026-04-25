"""Tests for filter_inside_polygon — postfilter NSPD frames spatially.

The NSPD attrib-search is text-based (we filter by ``readable_address ⊂ Казань``)
which captures both the agglomeration we want and a long tail of objects with
"Казань" in the address for unrelated reasons (e.g. "ул. Казанская" in another
city). Postfiltering by the agglomeration polygon trims that tail.
"""

from __future__ import annotations

import polars as pl
from shapely.geometry import Polygon

from kadastra.etl.filter_inside_polygon import filter_inside_polygon

# A small square polygon roughly around Kazan center for testing.
_KAZAN_BOX = Polygon(
    [
        (49.05, 55.75),
        (49.20, 55.75),
        (49.20, 55.85),
        (49.05, 55.85),
        (49.05, 55.75),
    ]
)


def _frame(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "lat": [r[1] for r in rows],
            "lon": [r[2] for r in rows],
        }
    )


def test_point_inside_polygon_is_kept() -> None:
    df = _frame([("kazan-center", 55.7887, 49.1221)])

    result = filter_inside_polygon(df, _KAZAN_BOX)

    assert result.height == 1
    assert result["id"][0] == "kazan-center"


def test_point_outside_polygon_is_dropped() -> None:
    df = _frame([("moscow", 55.7558, 37.6173)])  # Moscow

    result = filter_inside_polygon(df, _KAZAN_BOX)

    assert result.height == 0


def test_mixed_points_filters_correctly() -> None:
    df = _frame(
        [
            ("kazan-center", 55.7887, 49.1221),
            ("moscow", 55.7558, 37.6173),
            ("kazan-edge", 55.80, 49.10),
            ("st-petersburg", 59.9343, 30.3351),
        ]
    )

    result = filter_inside_polygon(df, _KAZAN_BOX)

    assert sorted(result["id"].to_list()) == ["kazan-center", "kazan-edge"]


def test_empty_input_returns_empty_with_same_schema() -> None:
    df = _frame([])

    result = filter_inside_polygon(df, _KAZAN_BOX)

    assert result.height == 0
    assert result.columns == ["id", "lat", "lon"]


def test_preserves_all_columns_unchanged() -> None:
    df = pl.DataFrame(
        {
            "id": ["x"],
            "lat": [55.79],
            "lon": [49.12],
            "extra_col": ["preserved"],
            "another": [42],
        }
    )

    result = filter_inside_polygon(df, _KAZAN_BOX)

    assert result.columns == ["id", "lat", "lon", "extra_col", "another"]
    assert result.height == 1
    assert result["extra_col"][0] == "preserved"
    assert result["another"][0] == 42


def test_handles_null_lat_lon_by_dropping() -> None:
    df = pl.DataFrame(
        {
            "id": ["null-lat", "null-lon", "kazan"],
            "lat": [None, 55.79, 55.79],
            "lon": [49.12, None, 49.12],
        }
    )

    result = filter_inside_polygon(df, _KAZAN_BOX)

    assert result["id"].to_list() == ["kazan"]

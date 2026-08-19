"""Unit tests for ADR-0027 §12 — overlap-weighted object→cell weights."""

from __future__ import annotations

import h3
import polars as pl
import shapely
from pyproj import Transformer

from kadastra.etl.cell_overlap_weights import (
    compute_overlap_weights,
    object_cell_weights,
)

_KAZAN_LAT = 55.79
_KAZAN_LON = 49.12
_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _kazan_centre_mercator() -> tuple[float, float]:
    return _4326_TO_3857.transform(_KAZAN_LON, _KAZAN_LAT)


def _mercator_box(cx: float, cy: float, half_m: float) -> str:
    return shapely.geometry.box(cx - half_m, cy - half_m, cx + half_m, cy + half_m).wkt


def test_null_geometry_returns_empty() -> None:
    assert object_cell_weights(None, 10) == []
    assert object_cell_weights("", 10) == []


def test_sub_cell_polygon_yields_empty_and_centroid_fallback() -> None:
    """h3 polyfill (4.4.2) is center-based: a polygon smaller than a
    cell contains no cell centre, so ``object_cell_weights`` returns
    []. The centroid fallback lives in ``compute_overlap_weights`` —
    verified separately."""
    cx, cy = _kazan_centre_mercator()
    wkt = _mercator_box(cx, cy, 10.0)  # 20 m square — well inside one res-10 cell

    assert object_cell_weights(wkt, 10) == []

    # And compute_overlap_weights falls back to the centroid cell.
    df = pl.DataFrame(
        {
            "object_id": ["x"],
            "polygon_wkt_3857": [wkt],
            "lat": [_KAZAN_LAT],
            "lon": [_KAZAN_LON],
        }
    )
    out = compute_overlap_weights(df, resolution=10)
    assert out.height == 1
    assert out["h3_index"][0] == h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    assert float(out["weight"][0]) == 1.0


def test_large_polygon_spanning_cells_weights_sum_to_one() -> None:
    cx, cy = _kazan_centre_mercator()
    # 300 m box spans ~2-3 res-10 cells (edge 75 m).
    wkt = _mercator_box(cx, cy, 150.0)

    weights = object_cell_weights(wkt, 10)

    assert len(weights) >= 2
    total = sum(w for _, w in weights)
    assert abs(total - 1.0) < 1e-9
    assert all(w > 0 for _, w in weights)


def test_compute_overlap_weights_returns_long_frame() -> None:
    cx, cy = _kazan_centre_mercator()
    df = pl.DataFrame(
        {
            "object_id": ["a", "b"],
            "polygon_wkt_3857": [_mercator_box(cx, cy, 10.0), _mercator_box(cx, cy, 150.0)],
            "lat": [_KAZAN_LAT, _KAZAN_LAT],
            "lon": [_KAZAN_LON, _KAZAN_LON],
        }
    )

    out = compute_overlap_weights(df, resolution=10)

    assert set(out.columns) == {"object_id", "h3_index", "weight"}
    a_rows = out.filter(pl.col("object_id") == "a")
    # Object "a" is sub-cell → centroid fallback, single row weight 1.0.
    assert a_rows.height == 1
    assert float(a_rows["weight"][0]) == 1.0
    b_rows = out.filter(pl.col("object_id") == "b")
    assert b_rows.height >= 2
    assert abs(float(b_rows["weight"].sum()) - 1.0) < 1e-9


def test_compute_overlap_weights_null_geometry_falls_back_to_centroid() -> None:
    df = pl.DataFrame(
        {
            "object_id": ["x"],
            "polygon_wkt_3857": [None],
            "lat": [_KAZAN_LAT],
            "lon": [_KAZAN_LON],
        }
    )

    out = compute_overlap_weights(df, resolution=10)

    assert out.height == 1
    assert out["h3_index"][0] == h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    assert float(out["weight"][0]) == 1.0


def test_compute_overlap_weights_empty_frame() -> None:
    df = pl.DataFrame(
        {"object_id": [], "polygon_wkt_3857": [], "lat": [], "lon": []},
        schema={
            "object_id": pl.Utf8,
            "polygon_wkt_3857": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
        },
    )

    out = compute_overlap_weights(df, resolution=10)

    assert out.is_empty()
    assert set(out.columns) == {"object_id", "h3_index", "weight"}

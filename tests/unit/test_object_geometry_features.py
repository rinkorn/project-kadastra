"""Tests for compute_object_geometry_features (ADR-0018).

Reads ``polygon_wkt_3857`` (EPSG:3857 WKT) from a per-object DataFrame
and appends 7 derived geometric features:

- ``polygon_area_m2``        — shapely.area
- ``polygon_perimeter_m``    — shapely.length
- ``polygon_compactness``    — Polsby–Popper 4·π·A / P²
- ``polygon_convexity``      — A / area(convex_hull)
- ``bbox_aspect_ratio``      — long side / short side of min rotated bbox
- ``polygon_orientation_deg``— long-axis angle of min rotated bbox, [0°, 180°)
- ``polygon_n_vertices``     — number of unique vertices (excludes the
                                closing repeat in shapely's exterior ring)

Where WKT is null → all 7 columns are null on that row.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from kadastra.etl.object_geometry_features import compute_object_geometry_features


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "polygon_wkt_3857": pl.Utf8,
        },
    )


# --------------------------------------------------------------------------
# Schema / passthrough behavior
# --------------------------------------------------------------------------


def test_appends_seven_new_columns() -> None:
    df = _objects([{"object_id": "a1", "polygon_wkt_3857": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"}])
    out = compute_object_geometry_features(df)
    expected_new = {
        "polygon_area_m2",
        "polygon_perimeter_m",
        "polygon_compactness",
        "polygon_convexity",
        "bbox_aspect_ratio",
        "polygon_orientation_deg",
        "polygon_n_vertices",
    }
    assert expected_new.issubset(set(out.columns))


def test_preserves_input_columns() -> None:
    df = _objects([{"object_id": "a1", "polygon_wkt_3857": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"}])
    out = compute_object_geometry_features(df)
    # Original columns must survive in the output (function is additive).
    assert "object_id" in out.columns
    assert "polygon_wkt_3857" in out.columns
    assert out["object_id"].to_list() == ["a1"]


# --------------------------------------------------------------------------
# Square — golden values for compactness / convexity / aspect / vertices
# --------------------------------------------------------------------------


def test_square_has_canonical_geometry_values() -> None:
    """A 10×10 square at origin: area=100, perimeter=40, compactness=π/4,
    convexity=1.0 (already convex), aspect_ratio=1.0, n_vertices=4."""
    df = _objects([{"object_id": "sq", "polygon_wkt_3857": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"}])
    out = compute_object_geometry_features(df)
    row = out.row(0, named=True)

    assert row["polygon_area_m2"] == pytest.approx(100.0)
    assert row["polygon_perimeter_m"] == pytest.approx(40.0)
    assert row["polygon_compactness"] == pytest.approx(math.pi / 4, rel=1e-6)
    assert row["polygon_convexity"] == pytest.approx(1.0, rel=1e-6)
    assert row["bbox_aspect_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert row["polygon_n_vertices"] == 4


# --------------------------------------------------------------------------
# Long thin rectangle — compactness collapses, aspect ratio explodes
# --------------------------------------------------------------------------


def test_long_thin_rectangle_collapses_compactness() -> None:
    """10×100 rectangle: A=1000, P=220, compactness = 4π·1000/220² ≈ 0.260,
    aspect_ratio = 100/10 = 10, still convex."""
    df = _objects([{"object_id": "r1", "polygon_wkt_3857": "POLYGON ((0 0, 100 0, 100 10, 0 10, 0 0))"}])
    out = compute_object_geometry_features(df)
    row = out.row(0, named=True)

    assert row["polygon_area_m2"] == pytest.approx(1000.0)
    assert row["polygon_perimeter_m"] == pytest.approx(220.0)
    assert row["polygon_compactness"] == pytest.approx(4 * math.pi * 1000 / (220 * 220), rel=1e-6)
    assert row["polygon_convexity"] == pytest.approx(1.0, rel=1e-6)
    assert row["bbox_aspect_ratio"] == pytest.approx(10.0, rel=1e-6)


# --------------------------------------------------------------------------
# L-shape — convexity drops below 1
# --------------------------------------------------------------------------


def test_l_shape_has_convexity_below_one() -> None:
    """L-shape carved out of a 10×10 square (remove 5×5 corner).
    Resulting polygon has area = 75, convex hull is the original 10×10
    square (area 100), so convexity = 0.75."""
    l_wkt = "POLYGON ((0 0, 10 0, 10 5, 5 5, 5 10, 0 10, 0 0))"
    df = _objects([{"object_id": "l1", "polygon_wkt_3857": l_wkt}])
    out = compute_object_geometry_features(df)
    row = out.row(0, named=True)

    assert row["polygon_area_m2"] == pytest.approx(75.0)
    assert row["polygon_convexity"] == pytest.approx(0.75, rel=1e-6)
    assert row["polygon_n_vertices"] == 6


# --------------------------------------------------------------------------
# Orientation — rectangle rotated 45° must report ~45° (or equivalent)
# --------------------------------------------------------------------------


def test_orientation_aligned_rectangle_reports_zero_or_ninety() -> None:
    """Axis-aligned 10×100 rectangle: orientation must be 0° or 90°
    depending on which side shapely picks as "long". We accept both
    since min-rotated-rectangle's vertex order is arbitrary."""
    df = _objects([{"object_id": "r1", "polygon_wkt_3857": "POLYGON ((0 0, 100 0, 100 10, 0 10, 0 0))"}])
    out = compute_object_geometry_features(df)
    angle = out.row(0, named=True)["polygon_orientation_deg"]
    # Long axis is horizontal → angle ≈ 0° (mod 180°).
    assert angle == pytest.approx(0.0, abs=0.5) or angle == pytest.approx(180.0, abs=0.5)


def test_orientation_rotated_rectangle_reports_45() -> None:
    """10×100 rectangle rotated 45° around origin. Long axis aligns with
    the y=x diagonal → orientation must report ~45°."""
    # Pre-rotated coords: corners of 100-long axis at angle 45°.
    # Half-diagonal of long side: 50. dx = dy = 50/√2·... easier to
    # just write rotated coords directly:
    # original corners (0,0), (100,0), (100,10), (0,10) rotated by 45°
    # around origin: (x', y') = (x cos45 - y sin45, x sin45 + y cos45)
    s = math.sqrt(2) / 2
    corners = [(0, 0), (100, 0), (100, 10), (0, 10)]
    rotated = [(x * s - y * s, x * s + y * s) for x, y in corners]
    coords_str = ", ".join(f"{x} {y}" for x, y in rotated + rotated[:1])
    df = _objects([{"object_id": "r45", "polygon_wkt_3857": f"POLYGON (({coords_str}))"}])
    out = compute_object_geometry_features(df)
    angle = out.row(0, named=True)["polygon_orientation_deg"]
    # Long axis at 45° → orientation ≈ 45°. (mod 180° still 45°.)
    assert angle == pytest.approx(45.0, abs=0.5)


# --------------------------------------------------------------------------
# Triangle — sanity for compactness < square, n_vertices=3
# --------------------------------------------------------------------------


def test_triangle_has_three_vertices_and_lower_compactness_than_square() -> None:
    """Right triangle legs 10, 10. A = 50, P = 20 + 10·√2 ≈ 34.14,
    compactness = 4πA/P² ≈ 0.539, strictly less than a square's π/4."""
    df = _objects([{"object_id": "t1", "polygon_wkt_3857": "POLYGON ((0 0, 10 0, 0 10, 0 0))"}])
    out = compute_object_geometry_features(df)
    row = out.row(0, named=True)

    assert row["polygon_n_vertices"] == 3
    assert row["polygon_area_m2"] == pytest.approx(50.0)
    p = 10.0 + 10.0 + math.sqrt(200.0)
    assert row["polygon_perimeter_m"] == pytest.approx(p)
    assert row["polygon_compactness"] == pytest.approx(4 * math.pi * 50.0 / (p * p), rel=1e-6)
    # Triangle compactness must be strictly less than square compactness.
    assert row["polygon_compactness"] < math.pi / 4


# --------------------------------------------------------------------------
# Null geometry — feature columns null
# --------------------------------------------------------------------------


def test_null_wkt_yields_null_feature_columns() -> None:
    df = _objects([{"object_id": "n1", "polygon_wkt_3857": None}])
    out = compute_object_geometry_features(df)
    row = out.row(0, named=True)
    assert row["polygon_area_m2"] is None
    assert row["polygon_perimeter_m"] is None
    assert row["polygon_compactness"] is None
    assert row["polygon_convexity"] is None
    assert row["bbox_aspect_ratio"] is None
    assert row["polygon_orientation_deg"] is None
    assert row["polygon_n_vertices"] is None


def test_handles_mixed_null_and_valid_rows() -> None:
    df = _objects(
        [
            {"object_id": "ok", "polygon_wkt_3857": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"},
            {"object_id": "null", "polygon_wkt_3857": None},
        ]
    )
    out = compute_object_geometry_features(df).sort("object_id")
    assert out.filter(pl.col("object_id") == "ok").row(0, named=True)["polygon_area_m2"] == pytest.approx(100.0)
    assert out.filter(pl.col("object_id") == "null").row(0, named=True)["polygon_area_m2"] is None


# --------------------------------------------------------------------------
# Polygon column missing entirely → degrades gracefully (all-null)
# --------------------------------------------------------------------------


def test_missing_polygon_column_raises_keyerror() -> None:
    """If the upstream gold doesn't have polygon_wkt_3857 at all, this
    is a contract violation — surface it as KeyError, not silently
    null-filled. Catches schema drift early."""
    df = pl.DataFrame({"object_id": ["a1"]}, schema={"object_id": pl.Utf8})
    with pytest.raises(KeyError, match="polygon_wkt_3857"):
        compute_object_geometry_features(df)

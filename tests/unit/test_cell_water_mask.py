"""Tests for the per-cell water-land share (ADR-0029 addendum)."""

from __future__ import annotations

import h3
import numpy as np
import pytest
from shapely.geometry import Polygon

from kadastra.etl.cell_water_mask import (
    ON_WATER_SHARE_THRESHOLD,
    compute_cell_water_share,
)


def _water_polygon_covering(cells: list[str], *, fraction: float) -> Polygon:
    """A polygon covering `fraction` of the first cell's bounding box."""
    cell = h3.cell_to_boundary(cells[0])
    lats = [lat for lat, _ in cell]
    lngs = [lng for _, lng in cell]
    lat0, lat1 = min(lats), max(lats)
    lng0, lng1 = min(lngs), max(lngs)
    # Interpolate a box inside the cell's bbox from the south edge up:
    # covers ~`fraction` of the cell area for the interior test cell.
    lat_cut = lat0 + (lat1 - lat0) * fraction
    return Polygon(
        [
            (lng0 - 1, lat0 - 1),
            (lng1 + 1, lat0 - 1),
            (lng1 + 1, lat_cut),
            (lng0 - 1, lat_cut),
        ]
    )


def _two_cells() -> list[str]:
    # First cell anchors the water polygon (sized against its bbox);
    # the second sits a few km away — the polygon cannot reach it.
    return [h3.latlng_to_cell(55.7, 49.1, 10), h3.latlng_to_cell(55.72, 49.15, 10)]


def test_full_water_coverage() -> None:
    cells = _two_cells()
    big = _water_polygon_covering(cells, fraction=2.0)
    shares = compute_cell_water_share(cells, [big])
    assert shares[0] == pytest.approx(1.0)
    assert shares[0] >= ON_WATER_SHARE_THRESHOLD


def test_partial_coverage_stays_below_threshold() -> None:
    cells = _two_cells()
    partial = _water_polygon_covering(cells, fraction=0.3)
    shares = compute_cell_water_share(cells, [partial])
    assert 0 < shares[0] < ON_WATER_SHARE_THRESHOLD


def test_cell_far_from_water_is_zero() -> None:
    cells = _two_cells()
    partial = _water_polygon_covering(cells, fraction=0.3)
    shares = compute_cell_water_share(cells, [partial])
    # The polygon is bounded by the first cell's bbox; the neighbour
    # (grid_disk ring 1) shares no area with it.
    assert shares[1] == pytest.approx(0.0)


def test_no_water_geometries_returns_zeros() -> None:
    cells = _two_cells()
    shares = compute_cell_water_share(cells, [])
    assert np.allclose(shares, 0.0)


def test_overlapping_water_polygons_do_not_inflate() -> None:
    cells = _two_cells()
    big = _water_polygon_covering(cells, fraction=2.0)
    # The same body of water stored twice must not push the share past 1.
    shares = compute_cell_water_share(cells, [big, big])
    assert shares[0] == pytest.approx(1.0)

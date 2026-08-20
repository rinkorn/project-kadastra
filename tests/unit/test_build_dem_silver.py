"""Unit tests for BuildDemSilver (ADR-0023).

Two adjacent synthetic 1-arc-second EPSG:4326 tiles are merged,
reprojected to a metric UTM grid, and expanded into three silver
layers (elevation / slope_deg / relative_relief_500m). The synthetic
surface is a plane rising eastward at tan(10°) per metre, so the
expected slope is ~10° and the relief over ±500 m east-west is
~2·500·tan(10°) ≈ 176 m.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from kadastra.usecases.build_dem_silver import BuildDemSilver

_RES_DEG = 1 / 3600  # 1 arc-second, as in GLO-30/SRTM
_ROWS = 100
_COLS_PER_TILE = 100
_LAT_TOP = 56.0
_LON_WEST = 48.0
# Eastward rise per metre → slope ≈ 10°.
_GRADE = math.tan(math.radians(10.0))
# Metres per degree of longitude at the fixture latitude.
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(_LAT_TOP - _ROWS * _RES_DEG / 2))
_DZ_PER_COL = _GRADE * _M_PER_DEG_LON * _RES_DEG


def _write_tile(path: Path, *, lon_west: float, col_offset: int) -> None:
    cols = np.arange(_COLS_PER_TILE, dtype=np.float32) + col_offset
    values = 100.0 + _DZ_PER_COL * cols[None, :] * np.ones((_ROWS, 1), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        width=_COLS_PER_TILE,
        height=_ROWS,
        count=1,
        crs="EPSG:4326",
        transform=Affine(_RES_DEG, 0, lon_west, 0, -_RES_DEG, _LAT_TOP),
    ) as ds:
        ds.write(values, 1)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_tile(raw / "tile_west.tif", lon_west=_LON_WEST, col_offset=0)
    _write_tile(raw / "tile_east.tif", lon_west=_LON_WEST + _COLS_PER_TILE * _RES_DEG, col_offset=_COLS_PER_TILE)
    return raw


def test_empty_raw_dir_raises(tmp_path: Path) -> None:
    usecase = BuildDemSilver(dem_raw_dir=tmp_path / "nope", output_base_path=tmp_path / "silver")
    with pytest.raises(FileNotFoundError):
        usecase.execute("RU-TEST")


def test_writes_three_metric_layers(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "silver"
    BuildDemSilver(dem_raw_dir=raw_dir, output_base_path=out, relief_radius_m=500.0).execute("RU-TEST")

    base = out / "region=RU-TEST"
    elevation_path = base / "elevation.tif"
    slope_path = base / "slope_deg.tif"
    relief_path = base / "relative_relief_500m.tif"
    for path in (elevation_path, slope_path, relief_path):
        assert path.is_file(), path

    with rasterio.open(elevation_path) as ds:
        assert ds.crs.is_projected  # metric grid, not degrees
        elevation = ds.read(1)
        assert ds.res[0] == pytest.approx(ds.res[1], rel=0.05)
        assert 20.0 < ds.res[0] < 40.0  # ~1 arc-second in metres
    # Merge worked: elevation keeps rising eastward across both tiles.
    assert float(np.nanmean(elevation[:, -10:])) > float(np.nanmean(elevation[:, :10])) + 500.0

    with rasterio.open(slope_path) as ds:
        slope = ds.read(1)
    with rasterio.open(relief_path) as ds:
        relief = ds.read(1)

    assert slope.shape == elevation.shape == relief.shape
    # Interior medians (edges lose gradient/window context).
    interior = np.s_[20:-20, 20:-20]
    assert float(np.nanmedian(slope[interior])) == pytest.approx(10.0, abs=1.0)
    assert float(np.nanmedian(relief[interior])) == pytest.approx(176.0, abs=30.0)

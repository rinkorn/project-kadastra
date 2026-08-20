"""Unit tests for the rasterio DEM sampler adapter (ADR-0023).

Synthetic 10×10 GeoTIFFs in EPSG:32639 (metric CRS — the silver DEM
layers are stored in UTM so slope/relief derivatives are computed in
metres). Sampling happens at WGS84 coordinates; the adapter must
reproject query points into the raster CRS, return the pixel value at
pixel centres, and yield None outside the extent / on nodata pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer

from kadastra.adapters.rasterio_dem_sampler import RasterioDemSampler

_NODATA = -9999.0
_ORIGIN_X = 500_000.0
_ORIGIN_Y = 6_200_000.0
_PIXEL_M = 30.0
_SIZE = 10


def _write_raster(path: Path, values: np.ndarray, *, nodata: float | None = None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        width=_SIZE,
        height=_SIZE,
        count=1,
        crs="EPSG:32639",
        transform=Affine(_PIXEL_M, 0, _ORIGIN_X, 0, -_PIXEL_M, _ORIGIN_Y),
        nodata=nodata,
    ) as ds:
        ds.write(values.astype(np.float32), 1)


def _wgs84_at_pixel(row: int, col: int) -> tuple[float, float]:
    """WGS84 (lat, lon) of the centre of pixel (row, col)."""
    x = _ORIGIN_X + (col + 0.5) * _PIXEL_M
    y = _ORIGIN_Y - (row + 0.5) * _PIXEL_M
    transformer = Transformer.from_crs("EPSG:32639", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


@pytest.fixture
def sampler(tmp_path: Path) -> RasterioDemSampler:
    # elevation: 100 + 10*row + col → unique per pixel, exact float32.
    rows = np.arange(_SIZE, dtype=np.float32)
    cols = np.arange(_SIZE, dtype=np.float32)
    elevation = 100.0 + 10.0 * rows[:, None] + cols[None, :]
    elevation[0, 0] = _NODATA
    slope = np.full((_SIZE, _SIZE), 5.0, dtype=np.float32)
    relief = 200.0 + rows[:, None] + cols[None, :]
    _write_raster(tmp_path / "elevation.tif", elevation, nodata=_NODATA)
    _write_raster(tmp_path / "slope_deg.tif", slope)
    _write_raster(tmp_path / "relative_relief_500m.tif", relief)
    return RasterioDemSampler(
        elevation_path=tmp_path / "elevation.tif",
        slope_path=tmp_path / "slope_deg.tif",
        relief_path=tmp_path / "relative_relief_500m.tif",
    )


def test_samples_elevation_at_pixel_centre(sampler: RasterioDemSampler) -> None:
    lat, lon = _wgs84_at_pixel(2, 3)
    assert sampler.sample_elevation(lat=lat, lon=lon) == pytest.approx(100 + 10 * 2 + 3)


def test_samples_slope_and_relief(sampler: RasterioDemSampler) -> None:
    lat, lon = _wgs84_at_pixel(4, 5)
    assert sampler.sample_slope_deg(lat=lat, lon=lon) == pytest.approx(5.0)
    assert sampler.sample_relative_relief(lat=lat, lon=lon) == pytest.approx(200 + 4 + 5)


def test_nodata_pixel_returns_none(sampler: RasterioDemSampler) -> None:
    lat, lon = _wgs84_at_pixel(0, 0)
    assert sampler.sample_elevation(lat=lat, lon=lon) is None
    # Other layers are sampled independently — no nodata there.
    assert sampler.sample_slope_deg(lat=lat, lon=lon) == pytest.approx(5.0)


def test_outside_extent_returns_none(sampler: RasterioDemSampler) -> None:
    # ~10 km north-west of the raster origin.
    lat, lon = _wgs84_at_pixel(0, 0)
    assert sampler.sample_elevation(lat=lat + 0.1, lon=lon - 0.1) is None
    assert sampler.sample_slope_deg(lat=lat + 0.1, lon=lon - 0.1) is None
    assert sampler.sample_relative_relief(lat=lat + 0.1, lon=lon - 0.1) is None


def test_close_releases_datasets(sampler: RasterioDemSampler) -> None:
    sampler.close()

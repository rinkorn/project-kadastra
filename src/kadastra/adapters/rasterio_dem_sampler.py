"""Rasterio adapter for DemSamplerPort (ADR-0023).

Samples the three silver DEM rasters (elevation / slope / relative
relief) at WGS84 points. Stub — implementation follows the failing
tests in ``tests/unit/test_rasterio_dem_sampler.py``.
"""

from __future__ import annotations

from pathlib import Path


class RasterioDemSampler:
    def __init__(self, *, elevation_path: Path, slope_path: Path, relief_path: Path) -> None:
        self._elevation_path = elevation_path
        self._slope_path = slope_path
        self._relief_path = relief_path

    def sample_elevation(self, *, lat: float, lon: float) -> float | None:
        raise NotImplementedError

    def sample_slope_deg(self, *, lat: float, lon: float) -> float | None:
        raise NotImplementedError

    def sample_relative_relief(self, *, lat: float, lon: float) -> float | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

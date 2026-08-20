"""Rasterio adapter for DemSamplerPort (ADR-0023).

Samples the three silver DEM rasters (elevation / slope / relative
relief) at WGS84 points. The silver layers are stored in a metric CRS
(UTM zone picked by ``BuildDemSilver``), so every query point is
reprojected into each raster's CRS before the pixel lookup. A point
outside the raster extent or on a nodata/NaN pixel yields None — a
legitimate state for objects near the region boundary.
"""

from __future__ import annotations

import math
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.io import DatasetReader


class RasterioDemSampler:
    def __init__(self, *, elevation_path: Path, slope_path: Path, relief_path: Path) -> None:
        self._elevation = rasterio.open(elevation_path)
        self._slope = rasterio.open(slope_path)
        self._relief = rasterio.open(relief_path)
        # One WGS84→raster-CRS transformer per dataset (all three share
        # the CRS in practice, but the adapter does not assume it).
        self._transformers = {
            id(ds): Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            for ds in (self._elevation, self._slope, self._relief)
        }

    def sample_elevation(self, *, lat: float, lon: float) -> float | None:
        return self._sample(self._elevation, lat=lat, lon=lon)

    def sample_slope_deg(self, *, lat: float, lon: float) -> float | None:
        return self._sample(self._slope, lat=lat, lon=lon)

    def sample_relative_relief(self, *, lat: float, lon: float) -> float | None:
        return self._sample(self._relief, lat=lat, lon=lon)

    def close(self) -> None:
        for ds in (self._elevation, self._slope, self._relief):
            ds.close()

    def _sample(self, ds: DatasetReader, *, lat: float, lon: float) -> float | None:
        x, y = self._transformers[id(ds)].transform(lon, lat)
        row, col = ds.index(x, y)
        if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
            return None
        # Window as a plain ((row0, row1), (col0, col1)) tuple — avoids
        # rasterio's attrs-based Window class, which pyright cannot
        # type (its __init__ is synthesized at runtime by attr.s).
        value = float(ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
        nodata = ds.nodata
        if nodata is not None and value == nodata:
            return None
        if math.isnan(value):
            return None
        return value

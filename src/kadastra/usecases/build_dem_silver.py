"""Use case: raw DEM tiles → silver topographic layers (ADR-0023).

Reads every ``*.tif`` in the raw DEM directory (Copernicus GLO-30
tiles, EPSG:4326, 1 arc-second), merges them into a single mosaic,
reprojects to the UTM zone of the mosaic centre (metre grid — slope
and relief need metric pixels), then writes three aligned layers:

``{output_base_path}/region={code}/``
  - ``elevation.tif``             — absolute elevation, m
  - ``slope_deg.tif``             — local slope, degrees
  - ``relative_relief_500m.tif``  — max − min elevation within
    ``relief_radius_m``, m

All layers are float32 GeoTIFFs with NaN nodata. The use case is
idempotent: outputs are overwritten on rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from kadastra.etl.dem_derivatives import (
    compute_relative_relief,
    compute_slope_deg,
    utm_epsg_for_lonlat,
)


class BuildDemSilver:
    def __init__(self, *, dem_raw_dir: Path, output_base_path: Path, relief_radius_m: float = 500.0) -> None:
        self._dem_raw_dir = dem_raw_dir
        self._output_base_path = output_base_path
        self._relief_radius_m = relief_radius_m

    def execute(self, region_code: str) -> None:
        tile_paths = sorted(self._dem_raw_dir.glob("*.tif"))
        if not tile_paths:
            raise FileNotFoundError(f"No DEM tiles (*.tif) in {self._dem_raw_dir}")

        elevation, dst_transform, dst_crs = self._merge_and_reproject(tile_paths)
        pixel_width_m = float(dst_transform.a)
        pixel_height_m = float(-dst_transform.e)

        slope = compute_slope_deg(elevation, pixel_width_m=pixel_width_m, pixel_height_m=pixel_height_m)
        relief = compute_relative_relief(
            elevation,
            radius_m=self._relief_radius_m,
            pixel_width_m=pixel_width_m,
            pixel_height_m=pixel_height_m,
        )

        out_dir = self._output_base_path / f"region={region_code}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, values in (
            ("elevation", elevation.astype(np.float32)),
            ("slope_deg", slope),
            ("relative_relief_500m", relief),
        ):
            self._write(out_dir / f"{name}.tif", values, transform=dst_transform, crs=dst_crs)

    def _merge_and_reproject(self, tile_paths: list[Path]) -> tuple[np.ndarray, Affine, Any]:
        """Merge raw tiles and reproject the mosaic onto a UTM metre grid."""
        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            src_crs = datasets[0].crs
            src_nodata = datasets[0].nodata
            mosaic, mosaic_transform = merge(datasets, nodata=src_nodata)
            src = mosaic[0].astype(np.float32)
            if src_nodata is not None:
                src = np.where(src == src_nodata, np.nan, src)

            # UTM zone of the mosaic centre: the pilot region sits well
            # inside one zone, and the metric grid is what the
            # derivatives need (dev-rules: UTM for areal operations).
            center_lon = mosaic_transform.c + mosaic_transform.a * src.shape[1] / 2
            center_lat = mosaic_transform.f + mosaic_transform.e * src.shape[0] / 2
            dst_crs = CRS.from_epsg(utm_epsg_for_lonlat(center_lon, center_lat))

            left, bottom, right, top = array_bounds(src.shape[0], src.shape[1], mosaic_transform)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src.shape[1], src.shape[0], left, bottom, right, top
            )
            dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
            reproject(
                source=src,
                destination=dst,
                src_transform=mosaic_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            return dst, dst_transform, dst_crs
        finally:
            for ds in datasets:
                ds.close()

    @staticmethod
    def _write(path: Path, values: np.ndarray, *, transform: Affine, crs: Any) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype="float32",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
        ) as ds:
            ds.write(values.astype(np.float32), 1)

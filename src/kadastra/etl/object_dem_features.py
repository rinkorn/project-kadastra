"""Per-object topographic features (ADR-0023).

Attaches three DEM-derived columns to a per-object DataFrame by
sampling the silver DEM rasters at each object's (lat, lon):

- ``elevation_m``             (Float64) — absolute elevation.
- ``slope_deg_local``         (Float64) — local slope in degrees.
- ``relative_relief_500m_m``  (Float64) — max − min elevation within
  the configured relief radius (``dem_relief_radius_m``, 500 m default).

Objects outside the raster extent (or on nodata pixels) get null
features — a no-coverage state is legitimate, not an error. Rows with
null coordinates are not sampled at all and get nulls too.

Idempotent: pre-existing output columns are dropped first (the store
is read-write; a rerun reads its own enriched output).
"""

from __future__ import annotations

import polars as pl

from kadastra.ports.dem_sampler import DemSamplerPort

DEM_FEATURE_COLUMNS: tuple[str, ...] = (
    "elevation_m",
    "slope_deg_local",
    "relative_relief_500m_m",
)


def compute_object_dem_features(objects: pl.DataFrame, *, dem_sampler: DemSamplerPort) -> pl.DataFrame:
    drop_existing = [c for c in DEM_FEATURE_COLUMNS if c in objects.columns]
    if drop_existing:
        objects = objects.drop(drop_existing)

    if objects.height == 0:
        return objects.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in DEM_FEATURE_COLUMNS])

    elevations: list[float | None] = []
    slopes: list[float | None] = []
    reliefs: list[float | None] = []
    for lat, lon in objects.select(["lat", "lon"]).iter_rows():
        if lat is None or lon is None:
            elevations.append(None)
            slopes.append(None)
            reliefs.append(None)
            continue
        elevations.append(dem_sampler.sample_elevation(lat=lat, lon=lon))
        slopes.append(dem_sampler.sample_slope_deg(lat=lat, lon=lon))
        reliefs.append(dem_sampler.sample_relative_relief(lat=lat, lon=lon))

    return objects.with_columns(
        pl.Series("elevation_m", elevations, dtype=pl.Float64),
        pl.Series("slope_deg_local", slopes, dtype=pl.Float64),
        pl.Series("relative_relief_500m_m", reliefs, dtype=pl.Float64),
    )

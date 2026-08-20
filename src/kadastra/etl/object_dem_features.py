"""Per-object topographic features (ADR-0023).

Stub — implementation follows the failing tests in
``tests/unit/test_object_dem_features.py``.
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
    raise NotImplementedError

"""Unit tests for ADR-0023 — per-object topographic (DEM) features.

``compute_object_dem_features`` attaches three columns sampled from the
silver DEM rasters: ``elevation_m``, ``slope_deg_local``,
``relative_relief_500m_m``. The sampler is a port — tests use a fake
returning known values; the rasterio adapter is covered separately in
``test_rasterio_dem_sampler.py``.
"""

from __future__ import annotations

import polars as pl

from kadastra.etl.object_dem_features import (
    DEM_FEATURE_COLUMNS,
    compute_object_dem_features,
)

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class _FakeDemSampler:
    """Constant-value fake with a per-call log to assert coordinates."""

    def __init__(self, *, elevation: float, slope: float, relief: float) -> None:
        self._elevation = elevation
        self._slope = slope
        self._relief = relief
        self.calls: list[tuple[float, float]] = []

    def sample_elevation(self, *, lat: float, lon: float) -> float | None:
        self.calls.append((lat, lon))
        return self._elevation

    def sample_slope_deg(self, *, lat: float, lon: float) -> float | None:
        return self._slope

    def sample_relative_relief(self, *, lat: float, lon: float) -> float | None:
        return self._relief


class _OutOfCoverageDemSampler:
    """Everything samples outside the raster → all None."""

    def sample_elevation(self, *, lat: float, lon: float) -> float | None:
        return None

    def sample_slope_deg(self, *, lat: float, lon: float) -> float | None:
        return None

    def sample_relative_relief(self, *, lat: float, lon: float) -> float | None:
        return None


def _objects(lats: list[float | None], lons: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": [f"way/{i}" for i in range(len(lats))],
            "lat": pl.Series(lats, dtype=pl.Float64),
            "lon": pl.Series(lons, dtype=pl.Float64),
        }
    )


def test_attaches_three_dem_columns() -> None:
    objects = _objects([KAZAN_LAT, KAZAN_LAT + 0.01], [KAZAN_LON, KAZAN_LON])
    sampler = _FakeDemSampler(elevation=83.0, slope=4.2, relief=35.0)

    df = compute_object_dem_features(objects, dem_sampler=sampler)

    assert set(DEM_FEATURE_COLUMNS).issubset(df.columns)
    for col in DEM_FEATURE_COLUMNS:
        assert df.schema[col] == pl.Float64
    row = df.row(0, named=True)
    assert row["elevation_m"] == 83.0
    assert row["slope_deg_local"] == 4.2
    assert row["relative_relief_500m_m"] == 35.0


def test_samples_at_object_coordinates() -> None:
    objects = _objects([KAZAN_LAT], [KAZAN_LON])
    sampler = _FakeDemSampler(elevation=83.0, slope=4.2, relief=35.0)

    compute_object_dem_features(objects, dem_sampler=sampler)

    assert sampler.calls == [(KAZAN_LAT, KAZAN_LON)]


def test_sampler_none_yields_null_features() -> None:
    objects = _objects([KAZAN_LAT], [KAZAN_LON])

    df = compute_object_dem_features(objects, dem_sampler=_OutOfCoverageDemSampler())

    row = df.row(0, named=True)
    for col in DEM_FEATURE_COLUMNS:
        assert row[col] is None


def test_null_coordinates_yield_null_features_without_sampling() -> None:
    objects = _objects([None, KAZAN_LAT], [KAZAN_LON, None])
    sampler = _FakeDemSampler(elevation=83.0, slope=4.2, relief=35.0)

    df = compute_object_dem_features(objects, dem_sampler=sampler)

    assert sampler.calls == []
    for row in df.to_dicts():
        for col in DEM_FEATURE_COLUMNS:
            assert row[col] is None


def test_idempotent_rerun_replaces_columns() -> None:
    """Re-running on an already-enriched frame replaces the DEM columns
    instead of duplicating them (store is read-write)."""
    objects = _objects([KAZAN_LAT], [KAZAN_LON]).with_columns(
        pl.lit(1.0).alias("elevation_m"),
        pl.lit(1.0).alias("slope_deg_local"),
        pl.lit(1.0).alias("relative_relief_500m_m"),
    )
    sampler = _FakeDemSampler(elevation=83.0, slope=4.2, relief=35.0)

    df = compute_object_dem_features(objects, dem_sampler=sampler)

    assert not any(c.endswith("_right") for c in df.columns)
    assert df.row(0, named=True)["elevation_m"] == 83.0


def test_empty_frame_gets_null_columns() -> None:
    objects = pl.DataFrame(schema={"object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64})
    sampler = _FakeDemSampler(elevation=83.0, slope=4.2, relief=35.0)

    df = compute_object_dem_features(objects, dem_sampler=sampler)

    assert df.height == 0
    assert set(DEM_FEATURE_COLUMNS).issubset(df.columns)
    assert df.schema["elevation_m"] == pl.Float64

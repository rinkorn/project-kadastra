"""Unit tests for the CBD distance feature (ADR-0025, п. 1).

Fixtures use real Kazan coordinates: the CBD constant is the Kremlin /
пл. Свободы (55.7975, 49.1066), so a Kremlin point must be ~0 m and
the outskirts (Новое Караваево side) in the 20–30 km band.
"""

from __future__ import annotations

import polars as pl
import pytest

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.object_cbd_distance import compute_cbd_distance

CBD_LAT, CBD_LON = 55.7975, 49.1066

# Kremlin walls — a few hundred metres from the CBD anchor at most.
KREMLIN_LAT, KREMLIN_LON = 55.7987, 49.1064
# South-east outskirts of the agglomeration.
OUTSKIRTS_LAT, OUTSKIRTS_LON = 55.65, 49.35


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64},
    )


def test_cbd_point_itself_is_zero() -> None:
    df = compute_cbd_distance(
        _objects([{"object_id": "a", "lat": CBD_LAT, "lon": CBD_LON}]),
        cbd_lat=CBD_LAT,
        cbd_lon=CBD_LON,
    )
    assert df["dist_to_cbd_m"][0] == pytest.approx(0.0, abs=1.0)


def test_kremlin_is_close() -> None:
    df = compute_cbd_distance(
        _objects([{"object_id": "a", "lat": KREMLIN_LAT, "lon": KREMLIN_LON}]),
        cbd_lat=CBD_LAT,
        cbd_lon=CBD_LON,
    )
    dist = df["dist_to_cbd_m"][0]
    assert 0.0 <= dist < 500.0
    # Cross-check against the project's haversine utility.
    assert dist == pytest.approx(
        haversine_meters(KREMLIN_LAT, KREMLIN_LON, CBD_LAT, CBD_LON),
        rel=1e-6,
    )


def test_outskirts_in_20_30km_band() -> None:
    df = compute_cbd_distance(
        _objects([{"object_id": "a", "lat": OUTSKIRTS_LAT, "lon": OUTSKIRTS_LON}]),
        cbd_lat=CBD_LAT,
        cbd_lon=CBD_LON,
    )
    assert 20_000.0 < df["dist_to_cbd_m"][0] < 30_000.0


def test_null_coords_yield_null_distance() -> None:
    df = compute_cbd_distance(
        _objects(
            [
                {"object_id": "a", "lat": None, "lon": None},
                {"object_id": "b", "lat": KREMLIN_LAT, "lon": KREMLIN_LON},
            ]
        ),
        cbd_lat=CBD_LAT,
        cbd_lon=CBD_LON,
    )
    assert df["dist_to_cbd_m"][0] is None
    assert df["dist_to_cbd_m"][1] is not None


def test_empty_frame_gets_empty_column() -> None:
    df = compute_cbd_distance(
        _objects([]),
        cbd_lat=CBD_LAT,
        cbd_lon=CBD_LON,
    )
    assert df.height == 0
    assert df.schema["dist_to_cbd_m"] == pl.Float64


def test_original_columns_preserved() -> None:
    df = compute_cbd_distance(
        _objects([{"object_id": "a", "lat": KREMLIN_LAT, "lon": KREMLIN_LON}]),
        cbd_lat=CBD_LAT,
        cbd_lon=CBD_LON,
    )
    assert {"object_id", "lat", "lon", "dist_to_cbd_m"}.issubset(set(df.columns))

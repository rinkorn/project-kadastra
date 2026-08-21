"""Tests for ``DissolvedLayers`` — the per-run dissolve cache shared by
the polygon-share and geom-distance feature blocks.

Profiling ``BuildObjectFeatures.execute`` showed ``unary_union`` over the
OSM polygonal layers (water/park/forest/industrial/cemetery) dominating
the runtime: each layer was dissolved once per feature block (and again
per asset-class slice) although the dissolved geometry depends only on
the layer. The cache must:

1. compute ``project(EPSG:4326→32639) → unary_union`` exactly once per
   layer geometries list, however many consumers ask for it;
2. return results bit-identical to the previous inline pipeline.
"""

from __future__ import annotations

import polars as pl
import pytest
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

import kadastra.etl.dissolved_layers as dissolved_layers_mod
from kadastra.etl.dissolved_layers import DissolvedLayers
from kadastra.etl.object_geom_distance_features import (
    compute_object_geom_distance_features,
)
from kadastra.etl.object_polygon_features import compute_object_polygon_features

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    """The pre-refactor inline projection, kept here as the equivalence anchor."""
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _water_layer() -> list[BaseGeometry]:
    return [
        box(_KAZAN_LON - 0.002, _KAZAN_LAT - 0.002, _KAZAN_LON, _KAZAN_LAT),
        box(_KAZAN_LON - 0.001, _KAZAN_LAT - 0.001, _KAZAN_LON + 0.001, _KAZAN_LAT + 0.001),
    ]


def _park_layer() -> list[BaseGeometry]:
    return [box(_KAZAN_LON + 0.005, _KAZAN_LAT + 0.005, _KAZAN_LON + 0.007, _KAZAN_LAT + 0.007)]


def _objects(coords: list[tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": [f"way/{i}" for i in range(len(coords))],
            "lat": [lat for lat, _ in coords],
            "lon": [lon for _, lon in coords],
        }
    )


@pytest.fixture
def union_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count real ``unary_union`` calls inside the dissolved-layers module."""
    calls = [0]
    real = unary_union

    def _counting(geoms: object, *args: object, **kwargs: object) -> BaseGeometry:
        calls[0] += 1
        return real(geoms)  # type: ignore[arg-type]

    monkeypatch.setattr(dissolved_layers_mod, "unary_union", _counting, raising=False)
    return calls


def test_dissolve_matches_inline_project_union() -> None:
    """The cached dissolve must be bit-identical to the previous inline
    ``unary_union([project(g) for g in layer])`` pipeline."""
    layer = _water_layer()
    expected = unary_union([_project_lonlat(g) for g in layer])
    merged = DissolvedLayers().dissolved(layer)
    assert merged.wkb == expected.wkb


def test_dissolve_runs_union_once_per_layer_list(union_counter: list[int]) -> None:
    cache = DissolvedLayers()
    layer = _water_layer()
    first = cache.dissolved(layer)
    second = cache.dissolved(layer)
    assert first is second
    assert union_counter[0] == 1


def test_dissolve_distinguishes_different_list_objects(union_counter: list[int]) -> None:
    """Two distinct list objects (e.g. parsed twice from the same file)
    are different cache keys — identity keying must never share a
    dissolve across lists the caller handed in separately."""
    cache = DissolvedLayers()
    cache.dissolved(_water_layer())
    cache.dissolved(_water_layer())
    assert union_counter[0] == 2


def test_polygon_features_with_shared_cache_bit_exact(union_counter: list[int]) -> None:
    """Same layer dict, two 'asset-class' calls sharing one cache:
    results must equal the no-cache baseline exactly, and each layer is
    dissolved once across both calls."""
    layers = {"water": _water_layer(), "park": _park_layer()}
    objects_a = _objects([(_KAZAN_LAT, _KAZAN_LON), (_KAZAN_LAT + 0.01, _KAZAN_LON)])
    objects_b = _objects([(_KAZAN_LAT - 0.005, _KAZAN_LON + 0.003)])

    baseline_a = compute_object_polygon_features(objects_a, polygons_by_layer=layers, radii_m=[100, 800])
    baseline_b = compute_object_polygon_features(objects_b, polygons_by_layer=layers, radii_m=[100, 800])

    cache = DissolvedLayers()
    out_a = compute_object_polygon_features(objects_a, polygons_by_layer=layers, radii_m=[100, 800], dissolved=cache)
    out_b = compute_object_polygon_features(objects_b, polygons_by_layer=layers, radii_m=[100, 800], dissolved=cache)

    assert out_a.equals(baseline_a)
    assert out_b.equals(baseline_b)
    assert union_counter[0] == len(layers)


def test_geom_distance_features_with_shared_cache_bit_exact(union_counter: list[int]) -> None:
    layers: dict[str, list[BaseGeometry]] = {"water": _water_layer(), "railway": [Point(_KAZAN_LON + 0.01, _KAZAN_LAT)]}
    objects_a = _objects([(_KAZAN_LAT, _KAZAN_LON)])
    objects_b = _objects([(_KAZAN_LAT + 0.02, _KAZAN_LON - 0.02)])

    baseline_a = compute_object_geom_distance_features(objects_a, geometries_by_layer=layers)
    baseline_b = compute_object_geom_distance_features(objects_b, geometries_by_layer=layers)

    cache = DissolvedLayers()
    out_a = compute_object_geom_distance_features(objects_a, geometries_by_layer=layers, dissolved=cache)
    out_b = compute_object_geom_distance_features(objects_b, geometries_by_layer=layers, dissolved=cache)

    assert out_a.equals(baseline_a)
    assert out_b.equals(baseline_b)
    assert union_counter[0] == len(layers)


def test_share_and_distance_blocks_share_one_dissolve(union_counter: list[int]) -> None:
    """The production wiring: one cache serves both feature blocks in a
    single execute, so a layer present in both is dissolved once total."""
    layers = {"water": _water_layer(), "park": _park_layer()}
    objects = _objects([(_KAZAN_LAT, _KAZAN_LON)])

    cache = DissolvedLayers()
    share = compute_object_polygon_features(objects, polygons_by_layer=layers, radii_m=[500], dissolved=cache)
    dist = compute_object_geom_distance_features(objects, geometries_by_layer=layers, dissolved=cache)

    assert union_counter[0] == len(layers)
    assert "water_share_500m" in share.columns
    assert "dist_to_water_m" in dist.columns

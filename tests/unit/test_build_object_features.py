import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h3
import numpy as np
import polars as pl
import pytest
from shapely.geometry.base import BaseGeometry

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.haversine import haversine_meters
from kadastra.etl.load_geometries import load_geojsonseq_points
from kadastra.etl.object_zonal_features import compute_object_zonal_features
from kadastra.ports.dem_sampler import DemSamplerPort
from kadastra.ports.feature_reader import FeatureReaderPort
from kadastra.ports.road_graph import RoadGraphPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort
from kadastra.usecases.build_object_features import BuildObjectFeatures

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class _HaversineRoadGraph(RoadGraphPort):
    """Test fake: returns euclidean haversine, ignores topology.

    Lets BuildObjectFeatures tests focus on routing/wiring without
    needing a real graph. Real graph behavior is covered in
    test_object_metro_features and test_networkx_road_graph.
    """

    def distance_matrix_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        out = np.empty((len(from_coords), len(to_coords)), dtype=np.float64)
        for i, (la1, lo1) in enumerate(from_coords):
            for j, (la2, lo2) in enumerate(to_coords):
                out[i, j] = haversine_meters(la1, lo1, la2, lo2)
        return out

    def nearest_distance_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        return self.distance_matrix_m(from_coords, to_coords).min(axis=1)

    # ADR-0024 topology accessors — this fake has no graph, so there is
    # nothing to snap to or reach. Not exercised by these tests.
    def snap_node(self, coord: tuple[float, float]) -> tuple[int, float]:
        raise ValueError("haversine fake has no nodes")

    def node_coord(self, node_id: int) -> tuple[float, float]:
        raise ValueError("haversine fake has no nodes")

    def reachable_nodes_within_m(
        self,
        from_coord: tuple[float, float],
        cutoff_m: float,
    ) -> dict[int, float]:
        return {}


_FAKE_GRAPH = _HaversineRoadGraph()


def _objects_for(ac: AssetClass) -> pl.DataFrame:
    # ADR-0017 + ADR-0018: a polygon_wkt_3857 column is part of the
    # gold contract. Use a 10×10 square in mercator metres so the
    # geometry compute step has a deterministic input — the exact
    # numbers are checked by the geometry-features unit tests; here
    # the fixture just guarantees the column is present and parseable.
    sq = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    return pl.DataFrame(
        [
            {
                "object_id": f"way/{ac.value}-1",
                "asset_class": ac.value,
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
                "year_built": 2015,
                "polygon_wkt_3857": sq,
            },
            {
                "object_id": f"way/{ac.value}-2",
                "asset_class": ac.value,
                "lat": KAZAN_LAT + 0.0009,
                "lon": KAZAN_LON,
                "levels": 5,
                "flats": 30,
                "year_built": 1972,
                "polygon_wkt_3857": sq,
            },
        ],
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
            "year_built": pl.Int64,
            "polygon_wkt_3857": pl.Utf8,
        },
    )


@dataclass
class _StoreCall:
    region_code: str
    asset_class: AssetClass
    df: pl.DataFrame


class _FakeStore:
    def __init__(self, initial: dict[AssetClass, pl.DataFrame] | None = None) -> None:
        self._initial = dict(initial or {})
        self.calls: list[_StoreCall] = []

    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None:
        self.calls.append(_StoreCall(region_code, asset_class, df))

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        assert region_code == "RU-KAZAN-AGG"
        return self._initial[asset_class]


class _ReaderStore(ValuationObjectReaderPort, ValuationObjectStorePort, Protocol):
    """Contract BuildObjectFeatures actually depends on: one object
    wired into both ``reader=`` and ``store=``. Mirrors the real
    ParquetValuationObjectStore, which is read-write-same-path — the
    design the idempotency test exercises. Fakes satisfy this
    structurally; no inheritance needed."""


def _stations_csv(rows: list[tuple[float, float]]) -> bytes:
    header = "name,lat,lon\n"
    body = "".join(f"s,{lat},{lon}\n" for lat, lon in rows)
    return (header + body).encode()


def _roads_json(ways: list[list[tuple[float, float]]]) -> bytes:
    elements = [
        {
            "type": "way",
            "geometry": [{"lat": lat, "lon": lon} for lat, lon in coords],
        }
        for coords in ways
    ]
    return json.dumps({"elements": elements}).encode()


class _FakeRawData:
    def __init__(
        self,
        stations: bytes = b"",
        entrances: bytes = b"",
        roads: bytes = b"",
    ) -> None:
        self._payloads = {
            "stations.csv": stations,
            "entrances.csv": entrances,
            "roads.json": roads,
        }

    def read_bytes(self, key: str) -> bytes:
        return self._payloads[key]

    def list_keys(self, prefix: str) -> list[str]:
        return [k for k in self._payloads if k.startswith(prefix)]


class _FakeCellDistReader:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def load(self, region_code: str, resolution: int, feature_set: str) -> pl.DataFrame:
        return self._df


def _usecase(
    store: _ReaderStore,
    raw: _FakeRawData,
    *,
    relative_feature_parent_resolutions: list[int] | None = None,
    relative_feature_columns: list[str] | None = None,
    zonal_radii_m: list[int] | None = None,
    zonal_layer_names: list[str] | None = None,
    poly_area_radii_m: list[int] | None = None,
    poly_area_layer_paths: dict[str, str] | None = None,
    geom_distance_layer_paths: dict[str, str] | None = None,
    current_year_for_age_features: int = 2026,
    cell_geom_distance_reader: FeatureReaderPort | None = None,
    cell_polygon_reader: FeatureReaderPort | None = None,
    cell_zonal_reader: FeatureReaderPort | None = None,
    cell_road_reader: FeatureReaderPort | None = None,
    cell_metro_reader: FeatureReaderPort | None = None,
    cell_walk_dist_reader: FeatureReaderPort | None = None,
    cell_tsorf_resolution: int = 10,
    cell_tsorf_overlap_weighted: bool = False,
    macro_oktmo_features_path: Path | None = None,
    cadastre_target_year: int = 2024,
    dem_sampler: DemSamplerPort | None = None,
    road_class_features_path: Path | None = None,
    isochrone_cache_path: Path | None = None,
    isochrone_cache_resolution: int = 11,
    cbd_coords: dict[str, tuple[float, float]] | None = None,
    heritage_silver_path: Path | None = None,
    zouit_features_path: Path | None = None,
) -> BuildObjectFeatures:
    return BuildObjectFeatures(
        reader=store,
        store=store,
        raw_data=raw,
        stations_key="stations.csv",
        entrances_key="entrances.csv",
        roads_key="roads.json",
        neighbor_radius_m=500.0,
        road_radius_m=500.0,
        road_graph=_FAKE_GRAPH,
        relative_feature_parent_resolutions=(
            relative_feature_parent_resolutions if relative_feature_parent_resolutions is not None else [7]
        ),
        relative_feature_columns=(
            relative_feature_columns if relative_feature_columns is not None else ["dist_metro_m"]
        ),
        zonal_radii_m=(zonal_radii_m if zonal_radii_m is not None else [100, 300, 500, 800]),
        zonal_layer_names=(
            zonal_layer_names
            if zonal_layer_names is not None
            else ["stations", "entrances", "apartments", "houses", "commercial"]
        ),
        poly_area_radii_m=(poly_area_radii_m if poly_area_radii_m is not None else [100, 800]),
        poly_area_layer_paths=(poly_area_layer_paths if poly_area_layer_paths is not None else {}),
        geom_distance_layer_paths=(geom_distance_layer_paths if geom_distance_layer_paths is not None else {}),
        current_year_for_age_features=current_year_for_age_features,
        cell_geom_distance_reader=cell_geom_distance_reader,
        cell_polygon_reader=cell_polygon_reader,
        cell_zonal_reader=cell_zonal_reader,
        cell_road_reader=cell_road_reader,
        cell_metro_reader=cell_metro_reader,
        cell_walk_dist_reader=cell_walk_dist_reader,
        cell_tsorf_resolution=cell_tsorf_resolution,
        cell_tsorf_overlap_weighted=cell_tsorf_overlap_weighted,
        macro_oktmo_features_path=macro_oktmo_features_path,
        cadastre_target_year=cadastre_target_year,
        dem_sampler=dem_sampler,
        road_class_features_path=road_class_features_path,
        isochrone_cache_path=isochrone_cache_path,
        isochrone_cache_resolution=isochrone_cache_resolution,
        cbd_coords=cbd_coords,
        heritage_silver_path=heritage_silver_path,
        zouit_features_path=zouit_features_path,
    )


def test_emits_one_save_per_requested_class() -> None:
    initial = {
        AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT),
        AssetClass.HOUSE: _objects_for(AssetClass.HOUSE),
        AssetClass.COMMERCIAL: _objects_for(AssetClass.COMMERCIAL),
    }
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE, AssetClass.COMMERCIAL]
    )

    saved_classes = sorted(c.asset_class.value for c in store.calls)
    assert saved_classes == ["apartment", "commercial", "house"]


def test_appends_feature_columns_to_each_partition() -> None:
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    expected = {
        "object_id",
        "asset_class",
        "lat",
        "lon",
        "levels",
        "flats",
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
        "count_apartments_500m",
        "count_houses_500m",
        "count_commercial_500m",
        "road_length_500m",
    }
    assert expected.issubset(set(df.columns))


def test_appends_object_geometry_feature_columns() -> None:
    """ADR-0018: BuildObjectFeatures must call compute_object_geometry_features
    so the 7 geometry-derived columns appear in the saved partition."""
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    geometry_cols = {
        "polygon_area_m2",
        "polygon_perimeter_m",
        "polygon_compactness",
        "polygon_convexity",
        "bbox_aspect_ratio",
        "polygon_orientation_deg",
        "polygon_n_vertices",
    }
    assert geometry_cols.issubset(set(df.columns))
    # Fixture is a 10×10 square in EPSG:3857 metres → area = 100, n_verts = 4.
    row = df.row(0, named=True)
    assert row["polygon_area_m2"] == 100.0
    assert row["polygon_n_vertices"] == 4


def test_appends_object_age_feature_columns() -> None:
    """ADR-0020: BuildObjectFeatures must call compute_object_age_features
    so the 4 derived age/era columns appear in the saved partition with
    a deterministic current_year (2026)."""
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, current_year_for_age_features=2026).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    df = store.calls[0].df
    age_cols = {
        "age_years",
        "age_years_sq",
        "era_category",
        "is_new_construction",
    }
    assert age_cols.issubset(set(df.columns))
    # Fixture has year_built ∈ {2015, 1972} → ages 11 and 54 (2026−).
    rows = sorted(df.iter_rows(named=True), key=lambda r: r["year_built"])
    assert rows[0]["year_built"] == 1972  # brezhnev (1969–1980)
    assert rows[0]["age_years"] == 54
    assert rows[0]["era_category"] == "brezhnev"
    assert rows[1]["year_built"] == 2015
    assert rows[1]["age_years"] == 11
    assert rows[1]["era_category"] == "2010s"


def test_neighbor_counts_see_all_classes() -> None:
    # Apartment near a house and a commercial — neighbor counts should reflect that
    sq = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    apt = pl.DataFrame(
        [
            {
                "object_id": "way/apt-1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
                "year_built": 2010,
                "polygon_wkt_3857": sq,
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
            "year_built": pl.Int64,
            "polygon_wkt_3857": pl.Utf8,
        },
    )
    house = pl.DataFrame(
        [
            {
                "object_id": "way/h-1",
                "asset_class": "house",
                "lat": KAZAN_LAT + 0.0009,
                "lon": KAZAN_LON,
                "levels": 1,
                "flats": None,
                "year_built": 1985,
                "polygon_wkt_3857": sq,
            }
        ],
        schema=apt.schema,
    )
    commercial = pl.DataFrame(
        [
            {
                "object_id": "way/c-1",
                "asset_class": "commercial",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON + 0.0009,
                "levels": 1,
                "flats": None,
                "year_built": 2005,
                "polygon_wkt_3857": sq,
            }
        ],
        schema=apt.schema,
    )

    store = _FakeStore(
        {
            AssetClass.APARTMENT: apt,
            AssetClass.HOUSE: house,
            AssetClass.COMMERCIAL: commercial,
        }
    )
    raw = _FakeRawData(
        stations=_stations_csv([]),
        entrances=_stations_csv([]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE, AssetClass.COMMERCIAL]
    )

    saved_apt = next(c for c in store.calls if c.asset_class is AssetClass.APARTMENT).df
    apt_row = saved_apt.filter(pl.col("object_id") == "way/apt-1")
    assert apt_row["count_houses_500m"][0] == 1
    assert apt_row["count_commercial_500m"][0] == 1
    assert apt_row["count_apartments_500m"][0] == 0  # only one apartment, self excluded


def test_appends_relative_feature_columns_for_configured_features() -> None:
    """BuildObjectFeatures must run compute_relative_features after the
    other ETL steps and surface the derived `__rel_p{R}_*` columns
    plus the `parent_h3_p{R}` / `count_p{R}` book-keeping columns
    in each saved partition. Without this the methodological-block-2
    win (ADR-0012) doesn't reach the model.
    """
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        relative_feature_parent_resolutions=[7, 8],
        relative_feature_columns=["dist_metro_m", "levels"],
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    expected_relative = {
        "parent_h3_p7",
        "count_p7",
        "parent_h3_p8",
        "count_p8",
        "dist_metro_m__rel_p7_diff_med",
        "dist_metro_m__rel_p7_ratio_med",
        "dist_metro_m__rel_p7_z_iqr",
        "dist_metro_m__rel_p8_diff_med",
        "dist_metro_m__rel_p8_ratio_med",
        "dist_metro_m__rel_p8_z_iqr",
        "levels__rel_p7_diff_med",
        "levels__rel_p7_ratio_med",
        "levels__rel_p7_z_iqr",
        "levels__rel_p8_diff_med",
        "levels__rel_p8_ratio_med",
        "levels__rel_p8_z_iqr",
    }
    assert expected_relative.issubset(set(df.columns))


def test_appends_zonal_density_columns_per_layer_and_radius() -> None:
    """BuildObjectFeatures must run compute_object_zonal_features and
    surface `{layer}_within_{R}m` for each configured (layer, radius)
    in each saved partition. Without this the methodological-block-3
    win (ADR-0013) doesn't reach the model.
    """
    initial = {
        AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT),
        AssetClass.HOUSE: _objects_for(AssetClass.HOUSE),
        AssetClass.COMMERCIAL: _objects_for(AssetClass.COMMERCIAL),
    }
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        zonal_radii_m=[100, 800],
        zonal_layer_names=["stations", "entrances", "apartments", "houses", "commercial"],
    ).execute(
        "RU-KAZAN-AGG",
        asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE, AssetClass.COMMERCIAL],
    )

    df = next(c for c in store.calls if c.asset_class is AssetClass.APARTMENT).df
    expected = {
        "stations_within_100m",
        "stations_within_800m",
        "entrances_within_100m",
        "entrances_within_800m",
        "apartments_within_100m",
        "apartments_within_800m",
        "houses_within_100m",
        "houses_within_800m",
        "commercial_within_100m",
        "commercial_within_800m",
    }
    assert expected.issubset(set(df.columns)), f"missing columns: {expected - set(df.columns)}"


def test_appends_poly_area_share_columns_for_each_layer_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """BuildObjectFeatures must read each poly-area layer GeoJSON-seq
    file from disk and surface `{layer}_share_{R}m` columns in each
    saved partition. Without this the methodological-block-3b win
    (ADR-0014) doesn't reach the model.
    """
    # Synthetic GeoJSON-seq with one polygon covering Kazan center.
    geojson_path = tmp_path / "water.geojsonseq"
    geojson_path.write_text(
        '{"type":"Feature","properties":{},"geometry":{"type":"Polygon",'
        '"coordinates":[[[49.10,55.78],[49.14,55.78],[49.14,55.80],[49.10,55.80],[49.10,55.78]]]}}\n'
    )

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        poly_area_radii_m=[100, 800],
        poly_area_layer_paths={"water": str(geojson_path)},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "water_share_100m" in df.columns
    assert "water_share_800m" in df.columns
    # KAZAN_LAT/KAZAN_LON is inside the polygon; share should be 1.0 at both radii.
    assert df["water_share_100m"][0] > 0.99
    assert df["water_share_800m"][0] > 0.99


def test_appends_geom_distance_columns_for_each_layer_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0019: BuildObjectFeatures must read each geom-distance layer
    GeoJSON-seq and surface `dist_to_<layer>_m` columns in each saved
    partition. Distance is independent of share — both blocks coexist."""
    # Polygon enveloping KAZAN_LAT/KAZAN_LON (55.7887, 49.1221) so the
    # first object lands inside it and reports distance 0.
    geojson_path = tmp_path / "park.geojsonseq"
    geojson_path.write_text(
        '{"type":"Feature","properties":{},"geometry":{"type":"Polygon",'
        '"coordinates":[[[49.121,55.788],[49.123,55.788],'
        "[49.123,55.789],[49.121,55.789],[49.121,55.788]]]}}\n"
    )

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        geom_distance_layer_paths={"park": str(geojson_path)},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "dist_to_park_m" in df.columns
    # Object at (55.7887, 49.1221) is inside the polygon → distance 0.
    d = float(df["dist_to_park_m"][0])
    assert d == 0.0


def test_missing_geom_distance_layer_path_yields_null_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """If a configured geom-distance layer's file does not exist, the
    pipeline must not crash — it emits null distance column. Keeps
    Settings.geom_distance_layer_paths configurable as a superset."""
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        geom_distance_layer_paths={"landfill": str(tmp_path / "missing.geojsonseq")},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "dist_to_landfill_m" in df.columns
    assert df["dist_to_landfill_m"][0] is None


def test_zonal_poi_layer_loaded_from_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0019 part 4: zonal counts for OSM-extracted point POIs.

    When a layer name is in zonal_layer_names but is not stations /
    entrances / a self-class slice, the pipeline should look it up in
    geom_distance_layer_paths and load the GeoJSON-seq as a point layer
    (Point features as-is, polygon features centroided) for the count
    helper. Verifies that `school_within_500m` materialises with at
    least one count when a school point is placed near the test object.
    """
    geojson_path = tmp_path / "school.geojsonseq"
    # Place a Point right on KAZAN_LAT/LON — should fall within every
    # radius (100/300/500/800 m) for the test object at the same location.
    geojson_path.write_text(
        '{"type":"Feature","properties":{"name":"Школа №1"},'
        f'"geometry":{{"type":"Point","coordinates":[{KAZAN_LON},{KAZAN_LAT}]}}}}\n'
    )

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        zonal_layer_names=["school"],
        zonal_radii_m=[500],
        geom_distance_layer_paths={"school": str(geojson_path)},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "school_within_500m" in df.columns
    assert int(df["school_within_500m"][0]) == 1


def test_missing_zonal_poi_layer_file_yields_zero_counts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A POI layer name in zonal_layer_names whose file is missing must
    not crash; columns appear with zero counts so downstream schema is
    stable across regions where the OSM extract hasn't been run yet."""
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        zonal_layer_names=["bus_stop"],
        zonal_radii_m=[500],
        geom_distance_layer_paths={"bus_stop": str(tmp_path / "missing.geojsonseq")},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "bus_stop_within_500m" in df.columns
    assert int(df["bus_stop_within_500m"][0]) == 0


def test_missing_poly_area_layer_path_is_skipped_gracefully(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """If a configured poly-area layer's file does not exist, the
    pipeline must not crash — it just emits zero-share columns. This
    keeps `Settings.poly_area_layer_paths` configurable as a superset
    even when some extractions haven't been run yet.
    """
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        poly_area_radii_m=[100],
        poly_area_layer_paths={"water": str(tmp_path / "missing.geojsonseq")},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "water_share_100m" in df.columns
    assert df["water_share_100m"][0] == 0.0


def test_handles_empty_partition_gracefully() -> None:
    empty_schema = {
        "object_id": pl.Utf8,
        "asset_class": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "levels": pl.Int64,
        "flats": pl.Int64,
        "year_built": pl.Int64,
    }
    store = _FakeStore(
        {
            AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT),
            AssetClass.HOUSE: pl.DataFrame(schema=empty_schema),
            AssetClass.COMMERCIAL: pl.DataFrame(schema=empty_schema),
        }
    )
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE, AssetClass.COMMERCIAL]
    )

    saved = {c.asset_class: c.df for c in store.calls}
    assert saved[AssetClass.HOUSE].is_empty()
    assert saved[AssetClass.APARTMENT].height == 2


def test_joins_cell_geom_distance_from_grid_store() -> None:
    """ADR-0027: when a cell_geom_distance_reader is wired, dist_to_* comes
    from the cell grid store via join, not per-object computation."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    reader = _FakeCellDistReader(pl.DataFrame({"h3_index": [cell], "resolution": [10], "dist_to_water_m": [123.0]}))

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        cell_geom_distance_reader=reader,
        cell_tsorf_resolution=10,
        geom_distance_layer_paths={},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "dist_to_water_m" in df.columns
    assert float(df["dist_to_water_m"][0]) == 123.0
    assert "h3_index" not in df.columns  # join key dropped
    assert "resolution" not in df.columns  # store bookkeeping column dropped


def test_joins_cell_polygon_share_from_grid_store() -> None:
    """ADR-0027: when a cell_polygon_reader is wired, share comes from the
    cell grid store via join, not per-object computation."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    reader = _FakeCellDistReader(pl.DataFrame({"h3_index": [cell], "resolution": [10], "water_share_500m": [0.42]}))

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        cell_polygon_reader=reader,
        cell_tsorf_resolution=10,
        poly_area_layer_paths={},
        poly_area_radii_m=[500],
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "water_share_500m" in df.columns
    assert float(df["water_share_500m"][0]) == 0.42
    assert "h3_index" not in df.columns


def test_joins_cell_zonal_density_from_grid_store() -> None:
    """ADR-0027: when a cell_zonal_reader is wired, within-counts come from
    the cell grid store via join, not per-object computation."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    reader = _FakeCellDistReader(pl.DataFrame({"h3_index": [cell], "resolution": [10], "stations_within_500m": [7]}))

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        cell_zonal_reader=reader,
        cell_tsorf_resolution=10,
        zonal_radii_m=[500],
        zonal_layer_names=["stations"],
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "stations_within_500m" in df.columns
    assert int(df["stations_within_500m"][0]) == 7
    assert "h3_index" not in df.columns


def test_joins_cell_road_density_from_grid_store() -> None:
    """ADR-0027: when a cell_road_reader is wired, road_length comes from
    the cell grid store via join, not per-object computation."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    reader = _FakeCellDistReader(pl.DataFrame({"h3_index": [cell], "resolution": [10], "road_length_500m": [950.0]}))

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        cell_road_reader=reader,
        cell_tsorf_resolution=10,
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "road_length_500m" in df.columns
    assert float(df["road_length_500m"][0]) == 950.0
    assert "h3_index" not in df.columns


def test_joins_cell_walk_dist_from_grid_store() -> None:
    """ADR-0027: when a cell_walk_dist_reader is wired, walk_dist_to_* comes
    from the cell grid store via join. Grid-only — no per-object fallback."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    reader = _FakeCellDistReader(
        pl.DataFrame({"h3_index": [cell], "resolution": [10], "walk_dist_to_school_m": [450.0]})
    )

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        cell_walk_dist_reader=reader,
        cell_tsorf_resolution=10,
        geom_distance_layer_paths={},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "walk_dist_to_school_m" in df.columns
    assert float(df["walk_dist_to_school_m"][0]) == 450.0
    assert "h3_index" not in df.columns
    assert "resolution" not in df.columns


def test_joins_cell_metro_from_grid_store() -> None:
    """ADR-0027: when a cell_metro_reader is wired, metro columns come from
    the cell grid store via join, not per-object graph computation."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    reader = _FakeCellDistReader(
        pl.DataFrame(
            {
                "h3_index": [cell],
                "resolution": [10],
                "dist_metro_m": [500.0],
                "dist_entrance_m": [400.0],
                "count_stations_1km": [2],
                "count_entrances_500m": [1],
            }
        )
    )

    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        cell_metro_reader=reader,
        cell_tsorf_resolution=10,
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "dist_metro_m" in df.columns
    assert float(df["dist_metro_m"][0]) == 500.0
    assert int(df["count_stations_1km"][0]) == 2
    assert "h3_index" not in df.columns


class _WriteThroughStore:
    """Read-write store that persists saves — so a second execute()
    reads the enriched output of the first. Mirrors
    ParquetValuationObjectStore's read-write-same-path design, the root
    of the ``_right`` contamination bug the idempotency test guards."""

    def __init__(self, initial: dict[AssetClass, pl.DataFrame]) -> None:
        self._data = dict(initial)
        self.calls: list[_StoreCall] = []

    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None:
        self.calls.append(_StoreCall(region_code, asset_class, df))
        self._data[asset_class] = df

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._data[asset_class]


def test_rerun_is_idempotent_no_right_duplicates() -> None:
    """ADR-0027 A/B guard: build_object_features reads and writes the
    same store. A rerun must not read its own enriched output and
    duplicate locational features as ``*_right`` via the grid join.
    Reduces to the raw schema before recomputing."""
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
    grid_metro = _FakeCellDistReader(pl.DataFrame({"h3_index": [cell], "resolution": [10], "dist_metro_m": [500.0]}))
    initial = {AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)}
    store = _WriteThroughStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    usecase = _usecase(
        store,
        raw,
        cell_metro_reader=grid_metro,
        cell_tsorf_resolution=10,
        geom_distance_layer_paths={},
    )
    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])
    first = store.calls[0].df
    assert "dist_metro_m" in first.columns

    # Second run reads the enriched partition (store is write-through).
    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])
    second = store.calls[1].df

    right_cols = [c for c in second.columns if c.endswith("_right")]
    assert right_cols == [], f"rerot produced _right duplicates: {right_cols}"
    assert "dist_metro_m" in second.columns
    assert float(second["dist_metro_m"][0]) == 500.0
    # Feature count must not grow across reruns.
    assert second.width == first.width


def test_joins_cell_tsorf_overlap_weighted() -> None:
    """ADR-0027 §12: with overlap_weighted enabled, an object whose
    footprint spans multiple res cells blends their ЦОФ by area share,
    not inherits the centroid cell's value. Verify the weighted-mean
    math against the overlap-weights helper directly."""
    import shapely
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    cx, cy = t.transform(KAZAN_LON, KAZAN_LAT)
    # 300 m box — spans 2 res-10 cells (edge 75 m).
    wkt = shapely.geometry.box(cx - 150, cy - 150, cx + 150, cy + 150).wkt

    from kadastra.etl.cell_overlap_weights import compute_overlap_weights

    obj = pl.DataFrame(
        {
            "object_id": ["way/apt-1"],
            "asset_class": ["apartment"],
            "lat": [KAZAN_LAT],
            "lon": [KAZAN_LON],
            "levels": [9],
            "flats": [72],
            "year_built": [2015],
            "polygon_wkt_3857": [wkt],
        },
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
            "year_built": pl.Int64,
            "polygon_wkt_3857": pl.Utf8,
        },
    )
    # Cell reader: the cells the box overlaps get distinct values.
    weights = compute_overlap_weights(obj, resolution=10)
    assert weights.height >= 2, "fixture should span ≥2 cells"
    cell_rows = []
    weight_list = weights["weight"].to_list()
    for i, cell in enumerate(weights["h3_index"].to_list()):
        cell_rows.append({"h3_index": cell, "resolution": 10, "dist_metro_m": 100.0 + 100.0 * i})
    reader = _FakeCellDistReader(pl.DataFrame(cell_rows))

    store = _FakeStore({AssetClass.APARTMENT: obj})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )
    _usecase(
        store,
        raw,
        cell_metro_reader=reader,
        cell_tsorf_resolution=10,
        cell_tsorf_overlap_weighted=True,
        geom_distance_layer_paths={},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    # Expected = Σ weight_i · value_i over the overlapping cells.
    expected = sum(w * (100.0 + 100.0 * i) for i, w in enumerate(weight_list))
    assert "dist_metro_m" in df.columns
    assert float(df["dist_metro_m"][0]) == pytest.approx(expected, rel=1e-6)
    # Must lie strictly between the two extremes (proves it's a blend).
    assert 100.0 < float(df["dist_metro_m"][0]) < 200.0
    assert "h3_index" not in df.columns


def _macro_wide_row(oktmo: str, year: int, salary: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "oktmo": [oktmo],
            "year": [year],
            "oktmo_avg_salary_rub": [salary],
            "oktmo_population": [5000.0],
            "oktmo_population_density": [50.0],
            "oktmo_housing_volume_5y_m2": [12000.0],
            "oktmo_unemployment_pct": [3.5],
            "oktmo_retail_turnover_per_capita": [200000.0],
        }
    )


def test_appends_macro_emiss_columns_when_table_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0022: with macro_oktmo_features_path wired and a wide table on
    disk, the 6 oktmo_* columns land on the saved partition. Test objects
    carry no oktmo_full (no GAR lookups wired), so values are null — the
    value-level join contract is covered by test_object_macro_features."""
    part_dir = tmp_path / "region=RU-KAZAN-AGG" / "year=2024"
    part_dir.mkdir(parents=True)
    _macro_wide_row("92601000", 2024, 70000.0).write_parquet(part_dir / "data.parquet")

    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, macro_oktmo_features_path=tmp_path).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    df = store.calls[0].df
    for col in (
        "oktmo_avg_salary_rub",
        "oktmo_population",
        "oktmo_population_density",
        "oktmo_housing_volume_5y_m2",
        "oktmo_unemployment_pct",
        "oktmo_retail_turnover_per_capita",
    ):
        assert col in df.columns
        assert df[col].is_null().all()


def test_macro_emiss_columns_absent_when_path_not_wired() -> None:
    """ADR-0022: without macro_oktmo_features_path the step is a no-op —
    no oktmo_* columns appear (feature-flagged via composition root)."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "oktmo_avg_salary_rub" not in df.columns


def test_macro_emiss_skipped_when_region_partition_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Path wired but no ``region=…/year=*`` partition on disk → the join
    is skipped entirely (no columns), matching the gar-lookup opt-in."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, macro_oktmo_features_path=tmp_path).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    df = store.calls[0].df
    assert "oktmo_avg_salary_rub" not in df.columns


class _FakeDemSampler:
    """Constant-value DemSamplerPort fake for wiring tests."""

    def sample_elevation(self, *, lat: float, lon: float) -> float | None:
        return 83.0

    def sample_slope_deg(self, *, lat: float, lon: float) -> float | None:
        return 4.2

    def sample_relative_relief(self, *, lat: float, lon: float) -> float | None:
        return 35.0


def test_appends_dem_columns_when_sampler_wired() -> None:
    """ADR-0023: with a DemSamplerPort wired, the three topographic
    columns land on the saved partition. The DEM columns are derived
    inside the usecase (after the RAW_OBJECT_SCHEMA reset), so they do
    NOT belong to the raw schema — a rerun recomputes them from the
    silver rasters."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, dem_sampler=_FakeDemSampler()).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    for col in ("elevation_m", "slope_deg_local", "relative_relief_500m_m"):
        assert col in df.columns
    row = df.row(0, named=True)
    assert row["elevation_m"] == 83.0
    assert row["slope_deg_local"] == 4.2
    assert row["relative_relief_500m_m"] == 35.0


def test_dem_columns_absent_when_sampler_not_wired() -> None:
    """ADR-0023: without a dem_sampler the step is a no-op — no DEM
    columns appear (feature-flagged via composition root)."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "elevation_m" not in df.columns
    assert "slope_deg_local" not in df.columns
    assert "relative_relief_500m_m" not in df.columns


def _road_class_silver_rows() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": "way/apartment-1",
                "nearest_road_class": "residential",
                "dist_to_motorway_m": 2100.0,
                "dist_to_primary_m": 900.0,
                "dist_to_secondary_m": 250.0,
                "dist_to_residential_m": 15.0,
                "dist_to_pedestrian_m": 60.0,
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "nearest_road_class": pl.Utf8,
            "dist_to_motorway_m": pl.Float64,
            "dist_to_primary_m": pl.Float64,
            "dist_to_secondary_m": pl.Float64,
            "dist_to_residential_m": pl.Float64,
            "dist_to_pedestrian_m": pl.Float64,
        },
    )


def test_appends_road_class_columns_when_silver_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0024 group 1: with road_class_features_path wired and the
    silver table on disk, the 6 road-class columns land on the saved
    partition, joined by object_id; objects missing from the table get
    nulls. The join runs after the RAW_OBJECT_SCHEMA reset, so the
    columns are recomputed from silver on every rerun."""
    part_dir = tmp_path / "region=RU-KAZAN-AGG"
    part_dir.mkdir(parents=True)
    _road_class_silver_rows().write_parquet(part_dir / "data.parquet")

    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, road_class_features_path=tmp_path).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    df = store.calls[0].df
    for col in (
        "nearest_road_class",
        "dist_to_motorway_m",
        "dist_to_primary_m",
        "dist_to_secondary_m",
        "dist_to_residential_m",
        "dist_to_pedestrian_m",
    ):
        assert col in df.columns
    row_1 = df.filter(pl.col("object_id") == "way/apartment-1").row(0, named=True)
    assert row_1["nearest_road_class"] == "residential"
    assert row_1["dist_to_residential_m"] == 15.0
    row_2 = df.filter(pl.col("object_id") == "way/apartment-2").row(0, named=True)
    assert row_2["nearest_road_class"] is None


def test_road_class_columns_absent_when_partition_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Path wired but no ``region=…`` partition on disk → the join is
    skipped entirely (no columns), matching the macro-oktmo opt-in."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, road_class_features_path=tmp_path).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    df = store.calls[0].df
    assert "nearest_road_class" not in df.columns
    assert "dist_to_motorway_m" not in df.columns


def _isochrone_cache_for_objects(objects: pl.DataFrame, resolution: int = 11) -> pl.DataFrame:
    rows = []
    for i, (lat, lon) in enumerate(objects.select(["lat", "lon"]).iter_rows()):
        cell = h3.latlng_to_cell(lat, lon, resolution)
        rows.append((cell, 1000.0 * (i + 1), 10 + i, i % 2))
    return pl.DataFrame(
        rows,
        schema={
            "h3_index": pl.Utf8,
            "iso15_pop_count": pl.Float64,
            "iso15_amenity_count": pl.Int64,
            "iso15_metro_reach": pl.Int64,
        },
        orient="row",
    ).unique(subset=["h3_index"], keep="first")


def test_appends_isochrone_columns_when_cache_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0024 group 2: with isochrone_cache_path wired and the per-hex
    cache on disk, the 3 iso15_* columns land on the saved partition via
    the object's res-11 cell."""
    objects = _objects_for(AssetClass.APARTMENT)
    part_dir = tmp_path / "region=RU-KAZAN-AGG" / "h3_p=11"
    part_dir.mkdir(parents=True)
    _isochrone_cache_for_objects(objects).write_parquet(part_dir / "data.parquet")

    store = _FakeStore({AssetClass.APARTMENT: objects})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, isochrone_cache_path=tmp_path).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    for col in ("iso15_pop_count", "iso15_amenity_count", "iso15_metro_reach"):
        assert col in df.columns
    # Both test objects sit ~100 m apart — same res-11 cell or not, each
    # must inherit exactly its own cell's cached values (no nulls here:
    # the cache covers both cells).
    assert df["iso15_pop_count"].null_count() == 0


def test_isochrone_columns_absent_when_partition_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Path wired but no cache partition on disk → the join is skipped
    entirely (no columns)."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, isochrone_cache_path=tmp_path).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "iso15_pop_count" not in df.columns
    assert "iso15_amenity_count" not in df.columns
    assert "iso15_metro_reach" not in df.columns


def test_appends_cbd_distance_when_coords_configured() -> None:
    """ADR-0025 п. 1: with the region present in cbd_coords, the saved
    partition carries dist_to_cbd_m (haversine to the CBD anchor). The
    fixture objects sit ~1.1 km from the Kazan CBD (55.7975, 49.1066)."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, cbd_coords={"RU-KAZAN-AGG": (55.7975, 49.1066)}).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    df = store.calls[0].df
    assert "dist_to_cbd_m" in df.columns
    assert df.schema["dist_to_cbd_m"] == pl.Float64
    dist = df["dist_to_cbd_m"][0]
    assert dist == pytest.approx(haversine_meters(KAZAN_LAT, KAZAN_LON, 55.7975, 49.1066), rel=1e-6)


def test_cbd_distance_absent_for_unknown_region() -> None:
    """No CBD configured for the region → the column is skipped entirely
    (per-region constant, ADR-0025 «CBD для не-Казани»)."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, cbd_coords={"RU-IRKUTSK-AGG": (52.2864, 104.2807)}).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    assert "dist_to_cbd_m" not in store.calls[0].df.columns


def _heritage_silver_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "osm_id": "n1",
                "ref_egrokn": "161510400850006",
                "heritage_level": "2",
                "name": "Памятник",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "polygon_wkt": None,
            }
        ],
        schema={
            "osm_id": pl.Utf8,
            "ref_egrokn": pl.Utf8,
            "heritage_level": pl.Utf8,
            "name": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "polygon_wkt": pl.Utf8,
        },
    )


def test_appends_heritage_columns_when_silver_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0025 п. 2: with heritage_silver_path wired and the silver ОКН
    layer on disk, the 4 heritage columns land on the saved partition.
    The fixture ОКН sits exactly at object 1's coords → dist 0, is_heritage 1."""
    part_dir = tmp_path / "region=RU-KAZAN-AGG"
    part_dir.mkdir(parents=True)
    _heritage_silver_frame().write_parquet(part_dir / "data.parquet")

    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, heritage_silver_path=tmp_path).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    for col in ("is_heritage_object", "dist_to_nearest_heritage_m", "count_heritage_500m", "inside_heritage_zone"):
        assert col in df.columns
    row_1 = df.row(0, named=True)
    assert row_1["dist_to_nearest_heritage_m"] == pytest.approx(0.0, abs=1.0)
    assert row_1["is_heritage_object"] == 1
    # Point-only layer → fallback: object 2 is ~100 m north → boundary;
    # assert only non-null here, exact fallback covered by unit tests.
    assert df["inside_heritage_zone"].null_count() == 0


def test_heritage_columns_absent_when_partition_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Path wired but no silver partition on disk → block skipped."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, heritage_silver_path=tmp_path).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    assert "dist_to_nearest_heritage_m" not in df.columns


def _zouit_silver_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": "way/apartment-1",
                "inside_zouit": 1,
                "zouit_types": "power_line;water_protection",
                "inside_water_protection": 1,
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "inside_zouit": pl.Int64,
            "zouit_types": pl.Utf8,
            "inside_water_protection": pl.Int64,
        },
    )


def test_appends_zouit_columns_when_silver_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0025 п. 3: with zouit_features_path wired and the per-object
    silver table on disk, the 3 ЗОУИТ columns land on the saved
    partition, joined by object_id; objects missing from the table get
    nulls. The join runs after the RAW_OBJECT_SCHEMA reset (ADR-0022/
    0023/0024 pattern), so the columns are recomputed from silver on
    every rerun."""
    part_dir = tmp_path / "region=RU-KAZAN-AGG"
    part_dir.mkdir(parents=True)
    _zouit_silver_frame().write_parquet(part_dir / "data.parquet")

    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, zouit_features_path=tmp_path).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    for col in ("inside_zouit", "zouit_types", "inside_water_protection"):
        assert col in df.columns
    row_1 = df.row(0, named=True)
    assert row_1["inside_zouit"] == 1
    assert row_1["zouit_types"] == "power_line;water_protection"
    assert df.row(1, named=True)["inside_zouit"] is None


def test_zouit_columns_absent_when_partition_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Path wired but no silver partition on disk → the join is skipped
    entirely (no columns)."""
    store = _FakeStore({AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT)})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(store, raw, zouit_features_path=tmp_path).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    df = store.calls[0].df
    for col in ("inside_zouit", "zouit_types", "inside_water_protection"):
        assert col not in df.columns


def test_shared_osm_layers_read_and_dissolved_once_per_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Perf: the polygonal OSM layers are referenced by both
    ``poly_area_layer_paths`` and ``geom_distance_layer_paths``. Within
    one execute the pipeline must parse each GeoJSON-seq file once and
    dissolve each layer once, however many feature blocks consume it —
    profiling showed the per-block redissolve dominating runtime.
    Values must stay identical to the per-block computation."""
    import kadastra.etl.dissolved_layers as dissolved_layers_mod
    import kadastra.etl.load_geometries as load_geometries_mod

    water_path = tmp_path / "water.geojsonseq"
    water_path.write_text(
        '{"type":"Feature","properties":{},"geometry":{"type":"Polygon",'
        '"coordinates":[[[49.10,55.78],[49.14,55.78],[49.14,55.80],[49.10,55.80],[49.10,55.78]]]}}\n'
    )
    school_path = tmp_path / "school.geojsonseq"
    school_path.write_text(
        '{"type":"Feature","properties":{},'
        f'"geometry":{{"type":"Point","coordinates":[{KAZAN_LON},{KAZAN_LAT}]}}}}\n'
    )

    union_calls = [0]
    real_union = dissolved_layers_mod.unary_union

    def _counting_union(geoms: object) -> object:
        union_calls[0] += 1
        return real_union(geoms)  # type: ignore[arg-type]

    monkeypatch.setattr(dissolved_layers_mod, "unary_union", _counting_union)

    loaded_paths: list[str] = []
    real_load = load_geometries_mod.load_geojsonseq_geometries

    def _counting_load(paths: dict[str, str]) -> dict[str, list[BaseGeometry]]:
        loaded_paths.extend(paths.values())
        return real_load(paths)

    monkeypatch.setattr(load_geometries_mod, "load_geojsonseq_geometries", _counting_load)

    initial = {
        AssetClass.APARTMENT: _objects_for(AssetClass.APARTMENT),
        AssetClass.HOUSE: _objects_for(AssetClass.HOUSE),
    }
    store = _FakeStore(initial)
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        poly_area_radii_m=[100, 800],
        poly_area_layer_paths={"water": str(water_path)},
        geom_distance_layer_paths={"water": str(water_path), "school": str(school_path)},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE])

    # water feeds both the share and the distance blocks; school feeds
    # distance only. Each layer dissolved exactly once for the whole run.
    assert union_calls[0] == 2
    # Each file parsed exactly once (no per-block reread of the water file).
    assert sorted(loaded_paths) == sorted([str(water_path), str(school_path)])

    df = next(c for c in store.calls if c.asset_class is AssetClass.APARTMENT).df
    # KAZAN_LAT/KAZAN_LON is inside the water polygon.
    assert df["water_share_100m"][0] > 0.99
    assert df["water_share_800m"][0] > 0.99
    assert float(df["dist_to_water_m"][0]) == 0.0
    # School point sits exactly on the first object.
    assert float(df["dist_to_school_m"][0]) < 1.0


def test_zonal_poi_layer_shares_parse_with_geom_distance_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Perf: a POI layer used both as a zonal count layer (ADR-0019
    part 4) and as a geom-distance layer must be read from disk exactly
    once per execute. The zonal block applies its point transform
    (Point as-is, else centroid, empty skipped) to the parsed geometries
    already cached for the distance block instead of re-reading the
    GeoJSON-seq. Values must stay bit-identical to the standalone
    ``load_geojsonseq_points`` computation."""
    school_path = tmp_path / "school.geojsonseq"
    d = 0.0001
    school_path.write_text(
        # Point exactly on object 1; a symmetric polygon whose centroid is
        # the same spot; an empty and a null geometry that must be skipped.
        '{"type":"Feature","properties":{},'
        f'"geometry":{{"type":"Point","coordinates":[{KAZAN_LON},{KAZAN_LAT}]}}}}\n'
        '{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[['
        f"{KAZAN_LON - d},{KAZAN_LAT - d}],[{KAZAN_LON + d},{KAZAN_LAT - d}],"
        f"[{KAZAN_LON + d},{KAZAN_LAT + d}],[{KAZAN_LON - d},{KAZAN_LAT + d}],"
        f"[{KAZAN_LON - d},{KAZAN_LAT - d}]]]}}}}\n"
        '{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[]}}\n'
        '{"type":"Feature","properties":{},"geometry":null}\n'
    )

    open_counts: dict[str, int] = {}
    real_open = Path.open

    def _counting_open(self: Path, *args: object, **kwargs: object) -> object:
        open_counts[str(self)] = open_counts.get(str(self), 0) + 1
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", _counting_open)

    objects = _objects_for(AssetClass.APARTMENT)
    store = _FakeStore({AssetClass.APARTMENT: objects})
    raw = _FakeRawData(
        stations=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        entrances=_stations_csv([(KAZAN_LAT, KAZAN_LON)]),
        roads=_roads_json([]),
    )

    _usecase(
        store,
        raw,
        zonal_layer_names=["school"],
        zonal_radii_m=[50],
        geom_distance_layer_paths={"school": str(school_path)},
    ).execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    # One disk read for the whole execute, shared by both blocks.
    assert open_counts.get(str(school_path), 0) == 1

    df = store.calls[0].df
    # Bit-equivalence with the baseline point-layer computation (asserted
    # before the baseline read so the counter above sees only execute()).
    baseline = compute_object_zonal_features(
        objects,
        layers={"school": load_geojsonseq_points(str(school_path))},
        radii_m=[50],
    )
    saved_col = df.select(["object_id", "school_within_50m"]).sort("object_id")
    base_col = baseline.select(["object_id", "school_within_50m"]).sort("object_id")
    assert saved_col.equals(base_col)
    # Concrete expectation: point + polygon centroid sit on object 1
    # (2 hits), ~100 m from object 2 (0 hits at 50 m); empty/null skipped.
    assert df.filter(pl.col("object_id") == "way/apartment-1")["school_within_50m"][0] == 2
    assert df.filter(pl.col("object_id") == "way/apartment-2")["school_within_50m"][0] == 0
    # The distance block still consumes the same shared layer.
    assert float(df.filter(pl.col("object_id") == "way/apartment-1")["dist_to_school_m"][0]) < 1.0

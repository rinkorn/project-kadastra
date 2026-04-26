import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.haversine import haversine_meters
from kadastra.ports.road_graph import RoadGraphPort
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
                "polygon_wkt_3857": sq,
            },
            {
                "object_id": f"way/{ac.value}-2",
                "asset_class": ac.value,
                "lat": KAZAN_LAT + 0.0009,
                "lon": KAZAN_LON,
                "levels": 5,
                "flats": 30,
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
            "polygon_wkt_3857": pl.Utf8,
        },
    )


@dataclass
class _StoreCall:
    region_code: str
    asset_class: AssetClass
    df: pl.DataFrame


class _FakeStore:
    def __init__(
        self, initial: dict[AssetClass, pl.DataFrame] | None = None
    ) -> None:
        self._initial = dict(initial or {})
        self.calls: list[_StoreCall] = []

    def save(
        self, region_code: str, asset_class: AssetClass, df: pl.DataFrame
    ) -> None:
        self.calls.append(_StoreCall(region_code, asset_class, df))

    def load(
        self, region_code: str, asset_class: AssetClass
    ) -> pl.DataFrame:
        assert region_code == "RU-KAZAN-AGG"
        return self._initial[asset_class]


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


def _usecase(
    store: _FakeStore,
    raw: _FakeRawData,
    *,
    relative_feature_parent_resolutions: list[int] | None = None,
    relative_feature_columns: list[str] | None = None,
    zonal_radii_m: list[int] | None = None,
    zonal_layer_names: list[str] | None = None,
    poly_area_radii_m: list[int] | None = None,
    poly_area_layer_paths: dict[str, str] | None = None,
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
            relative_feature_parent_resolutions
            if relative_feature_parent_resolutions is not None
            else [7]
        ),
        relative_feature_columns=(
            relative_feature_columns
            if relative_feature_columns is not None
            else ["dist_metro_m"]
        ),
        zonal_radii_m=(
            zonal_radii_m if zonal_radii_m is not None else [100, 300, 500, 800]
        ),
        zonal_layer_names=(
            zonal_layer_names
            if zonal_layer_names is not None
            else ["stations", "entrances", "apartments", "houses", "commercial"]
        ),
        poly_area_radii_m=(
            poly_area_radii_m if poly_area_radii_m is not None else [100, 800]
        ),
        poly_area_layer_paths=(
            poly_area_layer_paths if poly_area_layer_paths is not None else {}
        ),
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

    _usecase(store, raw).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

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
        "parent_h3_p7", "count_p7",
        "parent_h3_p8", "count_p8",
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
        "stations_within_100m", "stations_within_800m",
        "entrances_within_100m", "entrances_within_800m",
        "apartments_within_100m", "apartments_within_800m",
        "houses_within_100m", "houses_within_800m",
        "commercial_within_100m", "commercial_within_800m",
    }
    assert expected.issubset(set(df.columns)), (
        f"missing columns: {expected - set(df.columns)}"
    )


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

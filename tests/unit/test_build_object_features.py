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
    return pl.DataFrame(
        [
            {
                "object_id": f"way/{ac.value}-1",
                "asset_class": ac.value,
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
            },
            {
                "object_id": f"way/{ac.value}-2",
                "asset_class": ac.value,
                "lat": KAZAN_LAT + 0.0009,
                "lon": KAZAN_LON,
                "levels": 5,
                "flats": 30,
            },
        ],
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
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


def _usecase(
    store: _FakeStore, raw: _FakeRawData
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


def test_neighbor_counts_see_all_classes() -> None:
    # Apartment near a house and a commercial — neighbor counts should reflect that
    apt = pl.DataFrame(
        [
            {
                "object_id": "way/apt-1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
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

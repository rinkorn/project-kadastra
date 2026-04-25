import json

import h3
import polars as pl

from kadastra.usecases.build_road_features import BuildRoadFeatures

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _roads_json() -> bytes:
    payload = {
        "version": 0.6,
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"highway": "primary"},
                "geometry": [
                    {"lat": KAZAN_LAT, "lon": KAZAN_LON},
                    {"lat": KAZAN_LAT + 0.0001, "lon": KAZAN_LON},
                ],
            },
            {
                "type": "node",
                "id": 99,
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
            },
        ],
    }
    return json.dumps(payload).encode("utf-8")


class FakeRawData:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.reads: list[str] = []

    def read_bytes(self, key: str) -> bytes:
        self.reads.append(key)
        return self._objects[key]

    def list_keys(self, prefix: str) -> list[str]:
        return [k for k in self._objects if k.startswith(prefix)]


class FakeCoverageReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return self._by_resolution[resolution]


class FakeFeatureStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, str, pl.DataFrame]] = []

    def save(self, region_code: str, resolution: int, feature_set: str, df: pl.DataFrame) -> None:
        self.saved.append((region_code, resolution, feature_set, df))


def _coverage_for_kazan(resolution: int) -> pl.DataFrame:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, resolution)
    return pl.DataFrame({"h3_index": [cell], "resolution": [resolution]})


def _make_usecase(
    raw_data: FakeRawData,
    coverage_reader: FakeCoverageReader,
    feature_store: FakeFeatureStore,
) -> BuildRoadFeatures:
    return BuildRoadFeatures(
        coverage_reader=coverage_reader,
        raw_data=raw_data,
        feature_store=feature_store,
        roads_key="roads/tatarstan.json",
    )


def test_execute_reads_roads_from_raw_data() -> None:
    raw_data = FakeRawData({"roads/tatarstan.json": _roads_json()})
    coverage_reader = FakeCoverageReader({8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[8])

    assert "roads/tatarstan.json" in raw_data.reads


def test_execute_saves_roads_feature_set_per_resolution() -> None:
    raw_data = FakeRawData({"roads/tatarstan.json": _roads_json()})
    coverage_reader = FakeCoverageReader(
        {7: _coverage_for_kazan(7), 8: _coverage_for_kazan(8)}
    )
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[7, 8])

    saved_resolutions = sorted(r for _, r, _, _ in feature_store.saved)
    assert saved_resolutions == [7, 8]
    for region, _, feature_set, _ in feature_store.saved:
        assert region == "RU-TA"
        assert feature_set == "roads"


def test_execute_produces_road_length_column() -> None:
    raw_data = FakeRawData({"roads/tatarstan.json": _roads_json()})
    coverage_reader = FakeCoverageReader({8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[8])

    _, _, _, df = feature_store.saved[0]
    assert {"h3_index", "resolution", "road_length_m"} <= set(df.columns)
    assert df["road_length_m"][0] > 0


def test_execute_does_not_re_read_json_per_resolution() -> None:
    raw_data = FakeRawData({"roads/tatarstan.json": _roads_json()})
    coverage_reader = FakeCoverageReader(
        {7: _coverage_for_kazan(7), 8: _coverage_for_kazan(8)}
    )
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[7, 8])

    assert raw_data.reads.count("roads/tatarstan.json") == 1


def test_execute_ignores_non_way_elements() -> None:
    """JSON often mixes ways with nodes/relations — non-ways must be filtered out, not crash."""
    raw_data = FakeRawData({"roads/tatarstan.json": _roads_json()})
    coverage_reader = FakeCoverageReader({8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[8])

    _, _, _, df = feature_store.saved[0]
    assert df["road_length_m"][0] > 0  # the way contributes; the node is skipped

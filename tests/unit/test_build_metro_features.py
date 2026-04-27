import h3
import polars as pl

from kadastra.usecases.build_metro_features import BuildMetroFeatures

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _stations_csv() -> bytes:
    return (
        pl.DataFrame(
            {
                "lat": [KAZAN_LAT, KAZAN_LAT + 0.001],
                "lon": [KAZAN_LON, KAZAN_LON],
            }
        )
        .write_csv()
        .encode("utf-8")
    )


def _entrances_csv() -> bytes:
    return pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]}).write_csv().encode("utf-8")


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
) -> BuildMetroFeatures:
    return BuildMetroFeatures(
        coverage_reader=coverage_reader,
        raw_data=raw_data,
        feature_store=feature_store,
        stations_key="metro/stations.csv",
        entrances_key="metro/entrances.csv",
    )


def test_execute_reads_stations_and_entrances_from_raw_data() -> None:
    raw_data = FakeRawData({"metro/stations.csv": _stations_csv(), "metro/entrances.csv": _entrances_csv()})
    coverage_reader = FakeCoverageReader({8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[8])

    assert "metro/stations.csv" in raw_data.reads
    assert "metro/entrances.csv" in raw_data.reads


def test_execute_saves_one_feature_set_per_resolution() -> None:
    raw_data = FakeRawData({"metro/stations.csv": _stations_csv(), "metro/entrances.csv": _entrances_csv()})
    coverage_reader = FakeCoverageReader({7: _coverage_for_kazan(7), 8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[7, 8])

    saved_resolutions = sorted(r for _, r, _, _ in feature_store.saved)
    assert saved_resolutions == [7, 8]
    for region, _, feature_set, _ in feature_store.saved:
        assert region == "RU-TA"
        assert feature_set == "metro"


def test_execute_produces_features_with_expected_columns() -> None:
    raw_data = FakeRawData({"metro/stations.csv": _stations_csv(), "metro/entrances.csv": _entrances_csv()})
    coverage_reader = FakeCoverageReader({8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[8])

    _, _, _, df = feature_store.saved[0]
    expected = {
        "h3_index",
        "resolution",
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    }
    assert expected <= set(df.columns)


def test_execute_does_not_re_read_csv_per_resolution() -> None:
    raw_data = FakeRawData({"metro/stations.csv": _stations_csv(), "metro/entrances.csv": _entrances_csv()})
    coverage_reader = FakeCoverageReader({7: _coverage_for_kazan(7), 8: _coverage_for_kazan(8)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(raw_data, coverage_reader, feature_store)

    usecase.execute("RU-TA", resolutions=[7, 8])

    # CSVs read once each, not once per resolution
    assert raw_data.reads.count("metro/stations.csv") == 1
    assert raw_data.reads.count("metro/entrances.csv") == 1

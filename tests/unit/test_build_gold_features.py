import polars as pl

from kadastra.usecases.build_gold_features import BuildGoldFeatures


class FakeCoverageReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return self._by_resolution[resolution]


class FakeFeatureReader:
    def __init__(self, by_key: dict[tuple[int, str], pl.DataFrame]) -> None:
        self._by_key = by_key
        self.reads: list[tuple[int, str]] = []

    def load(self, region_code: str, resolution: int, feature_set: str) -> pl.DataFrame:
        self.reads.append((resolution, feature_set))
        return self._by_key[(resolution, feature_set)]


class FakeGoldStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, pl.DataFrame]] = []

    def save(self, region_code: str, resolution: int, df: pl.DataFrame) -> None:
        self.saved.append((region_code, resolution, df))


def test_execute_joins_all_feature_sets_into_gold() -> None:
    coverage = pl.DataFrame({"h3_index": ["h1", "h2"], "resolution": [8, 8]})
    metro = pl.DataFrame(
        {"h3_index": ["h1", "h2"], "resolution": [8, 8], "dist_metro_m": [10.0, 100.0]}
    )
    buildings = pl.DataFrame(
        {"h3_index": ["h1", "h2"], "resolution": [8, 8], "building_count": [3, 0]}
    )

    usecase = BuildGoldFeatures(
        coverage_reader=FakeCoverageReader({8: coverage}),
        feature_reader=FakeFeatureReader(
            {(8, "metro"): metro, (8, "buildings"): buildings}
        ),
        gold_store=FakeGoldStore(),
        feature_sets=["metro", "buildings"],
    )
    gold_store = usecase._gold_store
    assert isinstance(gold_store, FakeGoldStore)

    usecase.execute("RU-TA", resolutions=[8])

    assert len(gold_store.saved) == 1
    region, resolution, df = gold_store.saved[0]
    assert region == "RU-TA"
    assert resolution == 8
    assert {"h3_index", "resolution", "dist_metro_m", "building_count"} <= set(df.columns)
    assert df.height == 2


def test_execute_iterates_resolutions_and_calls_save_per_resolution() -> None:
    coverage_by_res = {
        7: pl.DataFrame({"h3_index": ["h_a"], "resolution": [7]}),
        8: pl.DataFrame({"h3_index": ["h_b"], "resolution": [8]}),
    }
    metro_by_res = {
        (7, "metro"): pl.DataFrame({"h3_index": ["h_a"], "resolution": [7], "dist_metro_m": [50.0]}),
        (8, "metro"): pl.DataFrame({"h3_index": ["h_b"], "resolution": [8], "dist_metro_m": [10.0]}),
    }
    gold_store = FakeGoldStore()
    usecase = BuildGoldFeatures(
        coverage_reader=FakeCoverageReader(coverage_by_res),
        feature_reader=FakeFeatureReader(metro_by_res),
        gold_store=gold_store,
        feature_sets=["metro"],
    )

    usecase.execute("RU-TA", resolutions=[7, 8])

    saved_resolutions = sorted(r for _, r, _ in gold_store.saved)
    assert saved_resolutions == [7, 8]


def test_execute_left_joins_so_uncovered_hexes_keep_nulls() -> None:
    coverage = pl.DataFrame({"h3_index": ["h1", "h2"], "resolution": [8, 8]})
    # metro covers only h1
    metro = pl.DataFrame({"h3_index": ["h1"], "resolution": [8], "dist_metro_m": [10.0]})
    gold_store = FakeGoldStore()

    usecase = BuildGoldFeatures(
        coverage_reader=FakeCoverageReader({8: coverage}),
        feature_reader=FakeFeatureReader({(8, "metro"): metro}),
        gold_store=gold_store,
        feature_sets=["metro"],
    )

    usecase.execute("RU-TA", resolutions=[8])

    _, _, df = gold_store.saved[0]
    assert df.height == 2  # both hexes from coverage retained
    by_hex = {row["h3_index"]: row for row in df.iter_rows(named=True)}
    assert by_hex["h1"]["dist_metro_m"] == 10.0
    assert by_hex["h2"]["dist_metro_m"] is None


def test_execute_reads_each_feature_set_once_per_resolution() -> None:
    coverage = pl.DataFrame({"h3_index": ["h1"], "resolution": [8]})
    metro = pl.DataFrame({"h3_index": ["h1"], "resolution": [8], "dist_metro_m": [10.0]})
    buildings = pl.DataFrame({"h3_index": ["h1"], "resolution": [8], "building_count": [1]})
    feature_reader = FakeFeatureReader(
        {(8, "metro"): metro, (8, "buildings"): buildings}
    )

    usecase = BuildGoldFeatures(
        coverage_reader=FakeCoverageReader({8: coverage}),
        feature_reader=feature_reader,
        gold_store=FakeGoldStore(),
        feature_sets=["metro", "buildings"],
    )

    usecase.execute("RU-TA", resolutions=[8])

    assert sorted(feature_reader.reads) == [(8, "buildings"), (8, "metro")]

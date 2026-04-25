import h3
import polars as pl

from kadastra.usecases.build_synthetic_target import BuildSyntheticTarget

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class FakeGoldReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution
        self.reads: list[tuple[str, int]] = []

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        self.reads.append((region_code, resolution))
        return self._by_resolution[resolution]


class FakeTargetStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, pl.DataFrame]] = []

    def save(self, region_code: str, resolution: int, df: pl.DataFrame) -> None:
        self.saved.append((region_code, resolution, df))


def _gold_row(resolution: int) -> pl.DataFrame:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, resolution)
    return pl.DataFrame(
        {
            "h3_index": [cell],
            "resolution": [resolution],
            "building_count": [5],
            "count_stations_1km": [1],
        }
    )


def test_execute_saves_one_table_per_resolution() -> None:
    gold_by_res = {7: _gold_row(7), 8: _gold_row(8)}
    target_store = FakeTargetStore()
    usecase = BuildSyntheticTarget(
        gold_reader=FakeGoldReader(gold_by_res),
        target_store=target_store,
        seed=42,
    )

    usecase.execute("RU-TA", resolutions=[7, 8])

    saved_resolutions = sorted(r for _, r, _ in target_store.saved)
    assert saved_resolutions == [7, 8]


def test_execute_adds_target_columns_to_saved_frame() -> None:
    gold_reader = FakeGoldReader({8: _gold_row(8)})
    target_store = FakeTargetStore()
    usecase = BuildSyntheticTarget(gold_reader=gold_reader, target_store=target_store, seed=42)

    usecase.execute("RU-TA", resolutions=[8])

    _, _, df = target_store.saved[0]
    assert "kazan_distance_km" in df.columns
    assert "synthetic_target_rub_per_m2" in df.columns


def test_execute_reads_each_resolution_once_with_region_code() -> None:
    gold_reader = FakeGoldReader({7: _gold_row(7), 8: _gold_row(8)})
    usecase = BuildSyntheticTarget(
        gold_reader=gold_reader,
        target_store=FakeTargetStore(),
        seed=42,
    )

    usecase.execute("RU-TA", resolutions=[7, 8])

    assert sorted(gold_reader.reads) == [("RU-TA", 7), ("RU-TA", 8)]


def test_execute_respects_seed_for_deterministic_output() -> None:
    gold_by_res = {8: _gold_row(8)}
    store1 = FakeTargetStore()
    store2 = FakeTargetStore()

    BuildSyntheticTarget(
        gold_reader=FakeGoldReader(gold_by_res),
        target_store=store1,
        seed=42,
    ).execute("RU-TA", resolutions=[8])
    BuildSyntheticTarget(
        gold_reader=FakeGoldReader(gold_by_res),
        target_store=store2,
        seed=42,
    ).execute("RU-TA", resolutions=[8])

    t1 = store1.saved[0][2]["synthetic_target_rub_per_m2"].to_list()
    t2 = store2.saved[0][2]["synthetic_target_rub_per_m2"].to_list()
    assert t1 == t2

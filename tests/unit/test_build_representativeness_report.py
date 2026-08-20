"""Unit tests for BuildRepresentativenessReport (эпик 001, этап 5)."""

from pathlib import Path

import h3
import polars as pl
import pytest

from kadastra.adapters.parquet_feature_store import ParquetFeatureStore
from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.build_representativeness_report import BuildRepresentativenessReport

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221
_RES = 10
_REGION = "RU-TEST"


class _FakeObjectReader:
    def __init__(self, objects: dict[AssetClass, pl.DataFrame]) -> None:
        self._objects = objects

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._objects.get(asset_class, pl.DataFrame())


def _grid_cells(n: int) -> list[str]:
    """``n`` distinct res-10 cells around Kazan (k-ring of a base cell)."""
    base = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, _RES)
    cells: list[str] = []
    ring = 0
    while len(cells) < n:
        # grid_disk returns the whole disk (inner cells included), so
        # dedupe while accumulating.
        cells = list(dict.fromkeys([*cells, *h3.grid_disk(base, ring)]))
        ring += 1
    return cells[:n]


def _write_feature_set(store: ParquetFeatureStore, feature_set: str, df: pl.DataFrame) -> None:
    store.save(_REGION, _RES, feature_set, df)


@pytest.fixture()
def store_with_grid(tmp_path: Path) -> tuple[ParquetFeatureStore, list[str]]:
    """Two feature sets over 20 cells; ``f_shift`` grows with cell order."""
    store = ParquetFeatureStore(tmp_path / "features")
    cells = _grid_cells(20)
    _write_feature_set(
        store,
        "geom_distance",
        pl.DataFrame(
            {
                "h3_index": cells,
                "resolution": [_RES] * 20,
                "f_stable": [100.0] * 20,
                "f_shift": [float(i) for i in range(20)],
            }
        ),
    )
    _write_feature_set(
        store,
        "road_density",
        pl.DataFrame(
            {
                "h3_index": cells,
                "resolution": [_RES] * 20,
                "road_length_500m": [float(i * 10) for i in range(20)],
            }
        ),
    )
    return store, cells


def _objects_in_cells(cells: list[str]) -> pl.DataFrame:
    """One object per given cell, placed at the cell centroid."""
    lats, lons = zip(*[h3.cell_to_latlng(c) for c in cells], strict=False)
    return pl.DataFrame({"lat": list(lats), "lon": list(lons)})


def test_execute_writes_parquet_and_markdown(
    tmp_path: Path, store_with_grid: tuple[ParquetFeatureStore, list[str]]
) -> None:
    store, cells = store_with_grid
    # Sample only the high-value half of the grid → shifted distribution.
    objects = _objects_in_cells(cells[10:])
    reader = _FakeObjectReader({AssetClass.APARTMENT: objects})
    usecase = BuildRepresentativenessReport(
        feature_store=store,
        object_reader=reader,
        output_base_path=tmp_path / "out",
        resolution=_RES,
        feature_sets=("geom_distance", "road_density"),
    )

    report = usecase.execute(_REGION, [AssetClass.APARTMENT])

    out_dir = tmp_path / "out" / f"region={_REGION}" / f"resolution={_RES}"
    assert (out_dir / "data.parquet").is_file()
    assert (out_dir / "report.md").is_file()

    # 3 features × 1 segment (single class → no overall roll-up).
    assert report.height == 3
    assert set(report["segment"]) == {"apartment"}

    shifted = report.filter(pl.col("feature") == "f_shift").row(0, named=True)
    assert shifted["verdict"] == "shift"
    assert shifted["coverage"] == pytest.approx(0.5)

    stable = report.filter(pl.col("feature") == "f_stable").row(0, named=True)
    assert stable["verdict"] == "n/a"  # constant population → PSI undefined


def test_execute_overall_segment_combines_classes(
    tmp_path: Path, store_with_grid: tuple[ParquetFeatureStore, list[str]]
) -> None:
    store, cells = store_with_grid
    reader = _FakeObjectReader(
        {
            AssetClass.APARTMENT: _objects_in_cells(cells[:5]),
            AssetClass.HOUSE: _objects_in_cells(cells[5:10]),
        }
    )
    usecase = BuildRepresentativenessReport(
        feature_store=store,
        object_reader=reader,
        output_base_path=tmp_path / "out",
        resolution=_RES,
        feature_sets=("geom_distance",),
    )

    report = usecase.execute(_REGION, [AssetClass.APARTMENT, AssetClass.HOUSE])

    assert set(report["segment"]) == {"apartment", "house", "overall"}
    overall = report.filter(pl.col("segment") == "overall").row(0, named=True)
    assert overall["n_sample"] == 10
    assert overall["coverage"] == pytest.approx(0.5)


def test_execute_missing_feature_set_is_skipped(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path / "features")
    cells = _grid_cells(5)
    _write_feature_set(
        store,
        "geom_distance",
        pl.DataFrame({"h3_index": cells, "resolution": [_RES] * 5, "f": [1.0] * 5}),
    )
    reader = _FakeObjectReader({AssetClass.APARTMENT: _objects_in_cells(cells[:2])})
    usecase = BuildRepresentativenessReport(
        feature_store=store,
        object_reader=reader,
        output_base_path=tmp_path / "out",
        resolution=_RES,
        feature_sets=("geom_distance", "walk_dist"),  # walk_dist absent on disk
    )

    report = usecase.execute(_REGION, [AssetClass.APARTMENT])

    assert set(report["feature"]) == {"f"}


def test_execute_no_feature_sets_raises(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path / "features")
    reader = _FakeObjectReader({AssetClass.APARTMENT: _objects_in_cells(_grid_cells(2))})
    usecase = BuildRepresentativenessReport(
        feature_store=store,
        object_reader=reader,
        output_base_path=tmp_path / "out",
        resolution=_RES,
    )
    with pytest.raises(FileNotFoundError):
        usecase.execute(_REGION, [AssetClass.APARTMENT])

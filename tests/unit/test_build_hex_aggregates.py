"""Tests for BuildHexAggregates use case.

Loads per-object gold + OOF predictions for each asset_class, joins
them on object_id, then aggregates to hex parquets at each requested
resolution.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.build_hex_aggregates import BuildHexAggregates

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class _FakeReader:
    def __init__(self, by_class: dict[AssetClass, pl.DataFrame]) -> None:
        self._by_class = by_class

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._by_class.get(asset_class, pl.DataFrame())


class _FakeOofReader:
    def __init__(
        self,
        by_class: dict[AssetClass, pl.DataFrame] | None = None,
        *,
        by_model: dict[tuple[AssetClass, str], pl.DataFrame] | None = None,
    ) -> None:
        self._by_class = by_class or {}
        self._by_model = by_model or {}
        self.calls: list[tuple[AssetClass, str]] = []

    def load_latest(self, asset_class: AssetClass, *, model: str = "catboost") -> pl.DataFrame:
        self.calls.append((asset_class, model))
        if (asset_class, model) in self._by_model:
            return self._by_model[(asset_class, model)]
        if model == "catboost" and asset_class in self._by_class:
            return self._by_class[asset_class]
        return pl.DataFrame(
            schema={
                "object_id": pl.Utf8,
                "lat": pl.Float64,
                "lon": pl.Float64,
                "fold_id": pl.Int64,
                "y_true": pl.Float64,
                "y_pred_oof": pl.Float64,
            }
        )


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "intra_city_raion": pl.Utf8,
        },
    )


def _oof(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "fold_id": pl.Int64,
            "y_true": pl.Float64,
            "y_pred_oof": pl.Float64,
        },
    )


def test_writes_one_parquet_per_resolution(tmp_path: Path) -> None:
    apt = _objects(
        [
            {
                "object_id": "a1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский",
            }
        ]
    )
    apt_oof = _oof(
        [
            {
                "object_id": "a1",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "fold_id": 0,
                "y_true": 100_000.0,
                "y_pred_oof": 95_000.0,
            }
        ]
    )

    usecase = BuildHexAggregates(
        reader=_FakeReader({AssetClass.APARTMENT: apt}),
        oof_reader=_FakeOofReader({AssetClass.APARTMENT: apt_oof}),
        output_base_path=tmp_path,
        resolutions=[8, 10],
    )
    usecase.execute("RU-KAZAN-AGG", [AssetClass.APARTMENT])

    p8 = tmp_path / "region=RU-KAZAN-AGG" / "resolution=8" / "model=catboost" / "data.parquet"
    p10 = tmp_path / "region=RU-KAZAN-AGG" / "resolution=10" / "model=catboost" / "data.parquet"
    assert p8.is_file()
    assert p10.is_file()

    df8 = pl.read_parquet(p8)
    assert (df8["resolution"] == 8).all()
    # apartment row + "all" row, same hex
    classes = sorted(df8["asset_class"].unique().to_list())
    assert classes == ["all", "apartment"]
    apt_row = df8.filter(pl.col("asset_class") == "apartment").row(0, named=True)
    assert apt_row["count"] == 1
    assert apt_row["median_target_rub_per_m2"] == 100_000.0
    assert apt_row["median_pred_oof_rub_per_m2"] == 95_000.0
    assert apt_row["median_residual_rub_per_m2"] == -5_000.0


def test_runs_without_oof_artifact_for_some_classes(tmp_path: Path) -> None:
    """A class that has no OOF artifact yet should still appear in the
    output — just with null prediction columns."""
    apt = _objects(
        [
            {
                "object_id": "a1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский",
            }
        ]
    )

    usecase = BuildHexAggregates(
        reader=_FakeReader({AssetClass.APARTMENT: apt}),
        # No OOF artifacts at all — empty mapping → empty frame.
        oof_reader=_FakeOofReader({}),
        output_base_path=tmp_path,
        resolutions=[10],
    )
    usecase.execute("RU-KAZAN-AGG", [AssetClass.APARTMENT])

    p = tmp_path / "region=RU-KAZAN-AGG" / "resolution=10" / "model=catboost" / "data.parquet"
    df = pl.read_parquet(p)
    apt_row = df.filter(pl.col("asset_class") == "apartment").row(0, named=True)
    assert apt_row["median_target_rub_per_m2"] == 100_000.0
    assert apt_row["median_pred_oof_rub_per_m2"] is None


def test_no_writes_when_all_classes_empty(tmp_path: Path) -> None:
    usecase = BuildHexAggregates(
        reader=_FakeReader({}),
        oof_reader=_FakeOofReader({}),
        output_base_path=tmp_path,
        resolutions=[10],
    )
    usecase.execute("RU-KAZAN-AGG", [AssetClass.APARTMENT])
    assert not (tmp_path / "region=RU-KAZAN-AGG").exists()


def test_writes_per_model_partition_and_threads_model_to_oof_reader(
    tmp_path: Path,
) -> None:
    """ADR-0016: each model has its own ``model={MODEL}`` partition,
    and BuildHexAggregates fetches the matching OOF artifact via
    ``oof_reader.load_latest(asset_class, model=...)``. Per-model
    medians on the same hex must differ when the underlying OOFs
    differ."""
    apt = _objects(
        [
            {
                "object_id": "a1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский",
            }
        ]
    )
    cb_oof = _oof(
        [
            {
                "object_id": "a1",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "fold_id": 0,
                "y_true": 100_000.0,
                "y_pred_oof": 95_000.0,
            }
        ]
    )
    ebm_oof = _oof(
        [
            {
                "object_id": "a1",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "fold_id": 0,
                "y_true": 100_000.0,
                "y_pred_oof": 80_000.0,
            }
        ]
    )
    fake_oof = _FakeOofReader(
        by_model={
            (AssetClass.APARTMENT, "catboost"): cb_oof,
            (AssetClass.APARTMENT, "ebm"): ebm_oof,
        }
    )

    usecase = BuildHexAggregates(
        reader=_FakeReader({AssetClass.APARTMENT: apt}),
        oof_reader=fake_oof,
        output_base_path=tmp_path,
        resolutions=[8],
    )
    usecase.execute("RU-KAZAN-AGG", [AssetClass.APARTMENT], model="ebm")

    cb_path = tmp_path / "region=RU-KAZAN-AGG" / "resolution=8" / "model=catboost" / "data.parquet"
    ebm_path = tmp_path / "region=RU-KAZAN-AGG" / "resolution=8" / "model=ebm" / "data.parquet"
    # Only ``ebm`` partition was requested → only that path written.
    assert ebm_path.is_file()
    assert not cb_path.is_file()

    df = pl.read_parquet(ebm_path)
    apt_row = df.filter(pl.col("asset_class") == "apartment").row(0, named=True)
    assert apt_row["median_pred_oof_rub_per_m2"] == 80_000.0
    assert apt_row["median_residual_rub_per_m2"] == -20_000.0
    # OOF reader must have been queried for "ebm", not "catboost".
    assert (AssetClass.APARTMENT, "ebm") in fake_oof.calls
    assert (AssetClass.APARTMENT, "catboost") not in fake_oof.calls


def test_concats_multiple_classes(tmp_path: Path) -> None:
    apt = _objects(
        [
            {
                "object_id": "a1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский",
            }
        ]
    )
    house = _objects(
        [
            {
                "object_id": "h1",
                "asset_class": "house",
                "lat": KAZAN_LAT + 1e-5,
                "lon": KAZAN_LON + 1e-5,
                "synthetic_target_rub_per_m2": 50_000.0,
                "intra_city_raion": "Советский",
            }
        ]
    )

    usecase = BuildHexAggregates(
        reader=_FakeReader({AssetClass.APARTMENT: apt, AssetClass.HOUSE: house}),
        oof_reader=_FakeOofReader({}),
        output_base_path=tmp_path,
        resolutions=[10],
    )
    usecase.execute("RU-KAZAN-AGG", [AssetClass.APARTMENT, AssetClass.HOUSE])

    df = pl.read_parquet(tmp_path / "region=RU-KAZAN-AGG" / "resolution=10" / "model=catboost" / "data.parquet")
    classes = sorted(df["asset_class"].unique().to_list())
    assert classes == ["all", "apartment", "house"]
    all_row = df.filter(pl.col("asset_class") == "all").row(0, named=True)
    assert all_row["count"] == 2
    # Median of 50k and 100k = 75k
    assert all_row["median_target_rub_per_m2"] == 75_000.0

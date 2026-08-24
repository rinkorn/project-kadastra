"""Tests for GetCellValuation — ADR-0029 cell valuation for the map UI's
«Стоимость типового объекта» mode.

Reads ParquetCellValuationStore partitions (region=/asset_class=) and
returns ``[{"hex", "value", "covered"}]`` for one (variant, metric)
combination. Mirrors the GetCellTsorf test pattern against the
cell-valuation store layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_cell_valuation_store import ParquetCellValuationStore
from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.get_cell_valuation import (
    CELL_VALUATION_METRICS,
    GetCellValuation,
)

_REGION = "RU-KAZAN-AGG"


def _rows(variant: str, cells: list[tuple[str, float, float, bool]]) -> list[dict[str, object]]:
    return [
        {
            "h3_index": h3,
            "lat": 55.74,
            "lon": 49.36,
            "reference_variant": variant,
            "reference_rub_per_m2": ref,
            "location_score_rub_per_m2": loc,
            "n_sample_objects": 3 if covered else 0,
            "sample_covered": covered,
        }
        for h3, ref, loc, covered in cells
    ]


def _write_partition(
    base: Path,
    asset_class: AssetClass,
    rows: list[dict[str, object]],
) -> None:
    ParquetCellValuationStore(base).save(_REGION, asset_class, pl.DataFrame(rows))


def test_execute_returns_hex_value_covered_for_default_variant(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    _write_partition(
        base,
        AssetClass.APARTMENT,
        _rows("default", [("8a", 80_000.0, 76_000.0, True), ("8b", 50_000.0, 49_000.0, False)]),
    )
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    out = usecase.execute(_REGION, AssetClass.APARTMENT, "default", "reference")

    assert out == [
        {"hex": "8a", "value": 80_000.0, "covered": True},
        {"hex": "8b", "value": 50_000.0, "covered": False},
    ]


def test_execute_location_score_metric_picks_location_column(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    _write_partition(
        base,
        AssetClass.HOUSE,
        _rows("default", [("8a", 60_000.0, 55_500.0, True)]),
    )
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    out = usecase.execute(_REGION, AssetClass.HOUSE, "default", "location_score")

    assert out == [{"hex": "8a", "value": 55_500.0, "covered": True}]


def test_execute_filters_to_requested_vri_variant(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    rows = _rows("default", [("8a", 10_000.0, 9_000.0, True)]) + _rows("Садоводство", [("8a", 12_000.0, 9_500.0, True)])
    _write_partition(base, AssetClass.LANDPLOT, rows)
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    out = usecase.execute(_REGION, AssetClass.LANDPLOT, "Садоводство", "reference")

    assert out == [{"hex": "8a", "value": 12_000.0, "covered": True}]


def test_execute_unknown_variant_raises_keyerror_with_available(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    _write_partition(
        base,
        AssetClass.LANDPLOT,
        _rows("default", [("8a", 10_000.0, 9_000.0, True)]),
    )
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    with pytest.raises(KeyError, match="default"):
        usecase.execute(_REGION, AssetClass.LANDPLOT, "nope", "reference")


def test_execute_unknown_metric_raises_keyerror(tmp_path: Path) -> None:
    usecase = GetCellValuation(ParquetCellValuationStore(tmp_path / "cell_valuation"))

    with pytest.raises(KeyError, match="nope"):
        usecase.execute(_REGION, AssetClass.APARTMENT, "default", "nope")


def test_execute_missing_partition_raises_filenotfound(tmp_path: Path) -> None:
    usecase = GetCellValuation(ParquetCellValuationStore(tmp_path / "cell_valuation"))

    with pytest.raises(FileNotFoundError):
        usecase.execute(_REGION, AssetClass.COMMERCIAL, "default", "reference")


def test_metrics_constant_matches_supported_metrics() -> None:
    assert set(CELL_VALUATION_METRICS) == {"reference", "location_score"}


def test_variant_map_lists_variants_per_class_and_skips_missing(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    _write_partition(
        base,
        AssetClass.APARTMENT,
        _rows("default", [("8a", 80_000.0, 76_000.0, True)]),
    )
    _write_partition(
        base,
        AssetClass.LANDPLOT,
        _rows("default", [("8a", 10_000.0, 9_000.0, True)]) + _rows("Садоводство", [("8a", 12_000.0, 9_500.0, True)]),
    )
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    m = usecase.variant_map(_REGION)

    # Every asset class is a key, even unbuilt ones.
    assert set(m.keys()) == {ac.value for ac in AssetClass}
    assert m["apartment"] == ["default"]
    assert m["landplot"] == ["default", "Садоводство"]
    assert m["house"] == []
    assert m["commercial"] == []


def test_get_cell_detail_returns_metrics_across_variants(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    rows = _rows("default", [("8a", 10_000.0, 9_000.0, True)]) + _rows(
        "Садоводство", [("8a", 12_000.0, 9_500.0, False)]
    )
    _write_partition(base, AssetClass.LANDPLOT, rows)
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    detail = usecase.get_cell_detail(_REGION, AssetClass.LANDPLOT, "8a")

    assert detail == {
        "default": {
            "reference_rub_per_m2": 10_000.0,
            "location_score_rub_per_m2": 9_000.0,
            "n_sample_objects": 3,
            "sample_covered": True,
        },
        "Садоводство": {
            "reference_rub_per_m2": 12_000.0,
            "location_score_rub_per_m2": 9_500.0,
            "n_sample_objects": 0,
            "sample_covered": False,
        },
    }


def test_get_cell_detail_parses_top_terms_from_default_variant(tmp_path: Path) -> None:
    """top_terms_json (stored on the «default» rows) leaves the usecase
    as a parsed ``top_terms`` list; variants without it get no key."""
    base = tmp_path / "cell_valuation"
    rows = _rows("default", [("8a", 10_000.0, 9_000.0, True)]) + _rows(
        "Садоводство", [("8a", 12_000.0, 9_500.0, False)]
    )
    rows[0]["top_terms_json"] = json.dumps([{"feature": "dist_metro_m", "contribution": 123.5}])
    rows[1]["top_terms_json"] = None
    _write_partition(base, AssetClass.LANDPLOT, rows)
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    detail = usecase.get_cell_detail(_REGION, AssetClass.LANDPLOT, "8a")

    assert detail["default"]["top_terms"] == [{"feature": "dist_metro_m", "contribution": 123.5}]
    assert "top_terms_json" not in detail["default"]
    assert "top_terms" not in detail["Садоводство"]


def test_get_cell_detail_returns_empty_when_cell_absent(tmp_path: Path) -> None:
    base = tmp_path / "cell_valuation"
    _write_partition(
        base,
        AssetClass.APARTMENT,
        _rows("default", [("8a", 80_000.0, 76_000.0, True)]),
    )
    usecase = GetCellValuation(ParquetCellValuationStore(base))

    assert usecase.get_cell_detail(_REGION, AssetClass.APARTMENT, "missing") == {}


def test_get_cell_detail_missing_partition_raises_filenotfound(tmp_path: Path) -> None:
    usecase = GetCellValuation(ParquetCellValuationStore(tmp_path / "cell_valuation"))

    with pytest.raises(FileNotFoundError):
        usecase.get_cell_detail(_REGION, AssetClass.HOUSE, "8a")

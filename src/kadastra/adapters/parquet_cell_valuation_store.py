"""Partitioned parquet adapter for the cell valuation layer (ADR-0029).

Layout mirrors the valuation-object store:
    {base_path}/region={region_code}/asset_class={asset_class}/data.parquet

Rows are keyed by (h3_index, reference_variant) — no object_id, so the
object store's schema contract does not apply; this adapter validates
the cell-valuation columns instead.
"""

from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass


class CellValuationSchemaError(ValueError):
    """Frame missing a required cell-valuation column or wrong dtype."""


_REQUIRED_COLUMNS: dict[str, pl.DataType] = {
    "h3_index": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "reference_variant": pl.String(),
    "reference_rub_per_m2": pl.Float64(),
    "location_score_rub_per_m2": pl.Float64(),
    "n_sample_objects": pl.Int64(),
    "sample_covered": pl.Boolean(),
}


def _validate_schema(df: pl.DataFrame, *, context: str) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CellValuationSchemaError(f"{context}: missing required columns {missing}; got {df.columns}")
    mismatches = [
        f"{name}: got {df.schema[name]}, expected {expected}"
        for name, expected in _REQUIRED_COLUMNS.items()
        if df.schema[name] != expected
    ]
    if mismatches:
        raise CellValuationSchemaError(f"{context}: dtype mismatches: {'; '.join(mismatches)}")


class ParquetCellValuationStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def _partition_dir(self, region_code: str, asset_class: AssetClass) -> Path:
        return self._base_path / f"region={region_code}" / f"asset_class={asset_class.value}"

    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None:
        _validate_schema(df, context=f"save region={region_code} asset_class={asset_class.value}")
        partition = self._partition_dir(region_code, asset_class)
        partition.mkdir(parents=True, exist_ok=True)
        df.write_parquet(partition / "data.parquet")

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        path = self._partition_dir(region_code, asset_class) / "data.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pl.read_parquet(path)
        _validate_schema(df, context=f"load {path}")
        return df

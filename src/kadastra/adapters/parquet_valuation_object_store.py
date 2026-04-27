"""Partitioned parquet adapter for valuation objects.

Layout:
    {base_path}/region={region_code}/asset_class={asset_class}/data.parquet

Used both for the assembly stage (NSPD silver → object frames with
gold/feature columns) and for downstream prediction frames — both
share the same core schema below. Per-stage extras are allowed:
the validator only enforces the four columns every consumer relies
on (object_id, asset_class, lat, lon).
"""

from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass


class ValuationObjectSchemaError(ValueError):
    """Frame missing a required column or carrying a wrong dtype.

    Raised on save *and* load so a bad write fails loudly at the source
    instead of surfacing as a confusing KeyError downstream.
    """


# Every consumer (build_object_features, train, infer, web inspector)
# reads at least these four columns. Extra columns are fine — the
# pipeline is additive (features get appended stage by stage).
_REQUIRED_COLUMNS: dict[str, pl.DataType] = {
    "object_id": pl.String(),
    "asset_class": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
}


def _validate_schema(df: pl.DataFrame, *, context: str) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValuationObjectSchemaError(f"{context}: missing required columns {missing}; got {df.columns}")
    mismatches = [
        f"{name}: got {df.schema[name]}, expected {expected}"
        for name, expected in _REQUIRED_COLUMNS.items()
        if df.schema[name] != expected
    ]
    if mismatches:
        raise ValuationObjectSchemaError(f"{context}: dtype mismatches: {'; '.join(mismatches)}")


class ParquetValuationObjectStore:
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

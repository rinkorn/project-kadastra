"""Read per-hex aggregates for the map UI.

Loads ``data/gold/hex_aggregates/region={REGION}/resolution={R}/data.parquet``,
filters by ``asset_class``, projects ``(h3_index, <feature>)`` and
returns it as ``[{"hex", "value"}, ...]``. The map UI's hex-mode
calls this for whatever (resolution, asset_class, feature) tuple is
selected.

Categorical features (``dominant_*``) are returned as strings; the
map UI colors them by category instead of by gradient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

# Numeric metrics → linear/log gradient on the map.
# Base market/model metrics + per-object descriptor means only —
# Слой-1 ЦОФ (dist/share/within/count/road/metro) are served by
# /api/cell_tsorf and deliberately not duplicated here.
NUMERIC_FEATURES: tuple[str, ...] = (
    # Base
    "count",
    "median_target_rub_per_m2",
    "median_pred_oof_rub_per_m2",
    "median_residual_rub_per_m2",
    # Building / land descriptors
    "mean_levels",
    "mean_flats",
    "mean_area_m2",
    "mean_year_built",
    "mean_age_years",
)
# Categorical metrics → categorical palette on the map.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "dominant_intra_city_raion",
    "dominant_mun_okrug_name",
    "dominant_settlement_name",
)
ASSET_CLASS_VALUES: tuple[str, ...] = (
    "all",
    "apartment",
    "house",
    "commercial",
    "landplot",
)


class GetHexAggregates:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def execute(
        self,
        region_code: str,
        resolution: int,
        asset_class: str,
        feature: str,
        *,
        model: str = "ebm",
    ) -> list[dict[str, object]]:
        path = (
            self._base_path / f"region={region_code}" / f"resolution={resolution}" / f"model={model}" / "data.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"hex aggregates not built for region={region_code} resolution={resolution}: {path}"
            )
        df = pl.read_parquet(path)
        if feature not in df.columns:
            available = sorted(c for c in df.columns if c not in {"h3_index", "resolution", "asset_class"})
            raise KeyError(f"feature {feature!r} not in hex_aggregates; available: {available}")

        filtered = df.filter(pl.col("asset_class") == asset_class)
        slim = filtered.select(["h3_index", pl.col(feature).alias("value")]).drop_nulls("value")
        return [{"hex": row["h3_index"], "value": row["value"]} for row in slim.iter_rows(named=True)]

    def get_detail(
        self,
        region_code: str,
        resolution: int,
        asset_class: str,
        h3_index: str,
        *,
        model: str = "ebm",
    ) -> dict[str, Any] | None:
        path = (
            self._base_path / f"region={region_code}" / f"resolution={resolution}" / f"model={model}" / "data.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"hex aggregates not built for region={region_code} resolution={resolution}: {path}"
            )
        df = pl.read_parquet(path)
        match = df.filter((pl.col("asset_class") == asset_class) & (pl.col("h3_index") == h3_index))
        if match.is_empty():
            return None
        return match.row(0, named=True)

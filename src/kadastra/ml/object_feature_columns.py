"""Shared rule for picking model-feature columns from a per-object frame.

Used by both TrainObjectValuationModel and InferObjectValuation so the
matrix layouts (numeric-then-categorical, with cat_feature_indices)
stay in sync. See ADR-0012 for why ``parent_h3_p{R}`` columns are
held out of the model — they are high-cardinality IDs that vanish
under spatial CV.
"""

from __future__ import annotations

import polars as pl

_NON_FEATURE_COLUMNS = frozenset(
    {
        "object_id",
        "asset_class",
        "lat",
        "lon",
        "synthetic_target_rub_per_m2",
        "cost_value_rub",
        # Block 4 (ADR-0015): identity / provenance, not features.
        "cad_num",
        "readable_address",
        "mun_source",
        # ADR-0017: passthrough geometry for the inspector. ADR-0018
        # derives 7 numeric features from it (polygon_area_m2 etc.);
        # the raw WKT itself is per-row unique and would poison cat
        # encoding if leaked into the model.
        "polygon_wkt_3857",
    }
)
_NON_FEATURE_PREFIXES = ("parent_h3_p",)


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_categorical(dtype: pl.DataType) -> bool:
    return dtype == pl.Utf8 or dtype == pl.Categorical


def _is_excluded(column: str) -> bool:
    if column in _NON_FEATURE_COLUMNS:
        return True
    return any(column.startswith(p) for p in _NON_FEATURE_PREFIXES)


def select_object_feature_columns(
    df: pl.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if _is_excluded(column):
            continue
        dtype = df.schema[column]
        if _is_numeric(dtype):
            numeric.append(column)
        elif _is_categorical(dtype):
            categorical.append(column)
    return numeric, categorical

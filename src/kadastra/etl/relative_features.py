"""Relative ZOFs from H3 parent-cell aggregates.

For each numeric ZOF column ``F`` and each parent H3 resolution
``R`` requested, attach four derived columns plus the parent-cell
metadata:

- ``parent_h3_p{R}``        — h3 index of the object's parent at res R
- ``count_p{R}``            — how many objects share that parent
- ``F__rel_p{R}_diff_med``  — F minus parent median
- ``F__rel_p{R}_ratio_med`` — F divided by parent median (NaN when median == 0)
- ``F__rel_p{R}_z_iqr``     — robust z-score: (F − median) / (p75 − p25), NaN when IQR == 0

Pure transformation: aggregates are computed over the rows of
``objects`` itself (the in-memory feature table). Caller decides
the upstream scope. ADR-0012 has the design rationale.
"""

from __future__ import annotations

import h3
import polars as pl


def compute_relative_features(
    objects: pl.DataFrame,
    *,
    parent_resolutions: list[int],
    feature_columns: list[str],
) -> pl.DataFrame:
    out = objects

    for r in parent_resolutions:
        parent_col = f"parent_h3_p{r}"
        count_col = f"count_p{r}"

        if out.is_empty():
            out = out.with_columns(
                [
                    pl.lit(None, dtype=pl.Utf8).alias(parent_col),
                    pl.lit(None, dtype=pl.UInt32).alias(count_col),
                ]
            )
        else:
            parents = [
                h3.latlng_to_cell(float(la), float(lo), r)
                for la, lo in zip(
                    out["lat"].to_list(), out["lon"].to_list(), strict=True
                )
            ]
            out = out.with_columns(pl.Series(parent_col, parents, dtype=pl.Utf8))
            out = out.with_columns(
                pl.len().over(parent_col).cast(pl.UInt32).alias(count_col)
            )

        for f in feature_columns:
            diff_col = f"{f}__rel_p{r}_diff_med"
            ratio_col = f"{f}__rel_p{r}_ratio_med"
            z_col = f"{f}__rel_p{r}_z_iqr"

            if out.is_empty():
                out = out.with_columns(
                    [
                        pl.lit(None, dtype=pl.Float64).alias(diff_col),
                        pl.lit(None, dtype=pl.Float64).alias(ratio_col),
                        pl.lit(None, dtype=pl.Float64).alias(z_col),
                    ]
                )
                continue

            f_expr = pl.col(f).cast(pl.Float64)
            median_expr = f_expr.median().over(parent_col)
            p25_expr = f_expr.quantile(0.25, "linear").over(parent_col)
            p75_expr = f_expr.quantile(0.75, "linear").over(parent_col)
            iqr_expr = p75_expr - p25_expr

            out = out.with_columns(
                [
                    (f_expr - median_expr).alias(diff_col),
                    pl.when(median_expr == 0)
                    .then(None)
                    .otherwise(f_expr / median_expr)
                    .alias(ratio_col),
                    pl.when(iqr_expr == 0)
                    .then(None)
                    .otherwise((f_expr - median_expr) / iqr_expr)
                    .alias(z_col),
                ]
            )

    return out

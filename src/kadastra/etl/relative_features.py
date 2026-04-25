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

import polars as pl


def compute_relative_features(
    objects: pl.DataFrame,
    *,
    parent_resolutions: list[int],
    feature_columns: list[str],
) -> pl.DataFrame:
    raise NotImplementedError

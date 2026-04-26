"""ADR-0020: derive age and era features from ``year_built``.

Stub — implementation in a follow-up commit per dev-rules TDD cycle."""

from __future__ import annotations

import polars as pl


def compute_object_age_features(
    objects: pl.DataFrame, *, current_year: int
) -> pl.DataFrame:
    """Append ``age_years``, ``age_years_sq``, ``era_category``,
    ``is_new_construction`` columns derived from ``year_built``.

    See [ADR-0020](info/decisions/0020-object-age-and-era-features.md)
    for the era-bin table and edge-case handling (year_built null/0 →
    `era_category="unknown"` and null numerics).
    """
    raise NotImplementedError

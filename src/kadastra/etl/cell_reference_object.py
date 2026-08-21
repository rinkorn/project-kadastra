"""Reference-object templates for the cell valuation layer (ADR-0029).

A «reference object» is the per-class typical object: medians of the
numeric attributes and modes of the categorical ones from the class's
gold slice, with the age/era features recomputed from the median
``year_built`` (not medianed themselves — the template must be
internally consistent). Landplot additionally builds one template per
top-N ``vri`` value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

# Numeric object attributes fixed by the template (ADR-0029 «Продукт 1»).
REFERENCE_NUMERIC_ATTRS: tuple[str, ...] = (
    "levels",
    "flats",
    "area_m2",
    "year_built",
    "underground_floors",
    "polygon_area_m2",
    "polygon_perimeter_m",
    "polygon_compactness",
    "polygon_convexity",
    "bbox_aspect_ratio",
    "polygon_orientation_deg",
    "polygon_n_vertices",
)

# Int-typed numerics — medians are rounded back to int.
REFERENCE_INT_ATTRS: tuple[str, ...] = (
    "levels",
    "flats",
    "year_built",
    "underground_floors",
    "polygon_n_vertices",
)

# Categorical object attributes fixed by the template (mode).
REFERENCE_CATEGORICAL_ATTRS: tuple[str, ...] = (
    "materials",
    "vri",
    "category_zem",
)


@dataclass(frozen=True)
class ReferenceObject:
    """One object-attribute template: ``default`` or a ``vri`` variant."""

    variant: str
    attributes: dict[str, Any]


def build_reference_objects(
    objects: pl.DataFrame,
    *,
    current_year: int,
    vri_top_n: int | None = None,
) -> list[ReferenceObject]:
    raise NotImplementedError

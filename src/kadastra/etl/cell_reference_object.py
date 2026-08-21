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
from typing import Any, cast

import polars as pl

from kadastra.etl.object_age_features import compute_object_age_features

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


def _median(series: pl.Series, *, as_int: bool) -> Any:
    value = series.drop_nulls().median()
    if value is None:
        return None
    numeric = float(cast("float", value))
    return round(numeric) if as_int else numeric


def _mode(series: pl.Series) -> Any:
    non_null = series.drop_nulls()
    if non_null.is_empty():
        return None
    counts = non_null.value_counts().sort("count", descending=True)
    return counts[series.name][0]


def _build_template(df: pl.DataFrame, *, variant: str, current_year: int) -> ReferenceObject:
    attrs: dict[str, Any] = {}
    for col in REFERENCE_NUMERIC_ATTRS:
        if col not in df.columns:
            continue
        series = df[col]
        # year_built == 0 is NSPD's «год не указан» encoding (ADR-0020).
        if col == "year_built":
            series = series.set(series == 0, None)
        attrs[col] = _median(series, as_int=col in REFERENCE_INT_ATTRS)
    for col in REFERENCE_CATEGORICAL_ATTRS:
        if col not in df.columns:
            continue
        attrs[col] = _mode(df[col])
    # Derived age/era features recomputed from the template's year_built
    # so the template is internally consistent (ADR-0029 «Продукт 1»).
    derived = compute_object_age_features(
        pl.DataFrame({"year_built": pl.Series([attrs.get("year_built")], dtype=pl.Int64)}),
        current_year=current_year,
    )
    for col in ("age_years", "age_years_sq", "era_category"):
        attrs[col] = derived[col][0]
    return ReferenceObject(variant=variant, attributes=attrs)


def build_reference_objects(
    objects: pl.DataFrame,
    *,
    current_year: int,
    vri_top_n: int | None = None,
) -> list[ReferenceObject]:
    """Build the reference-object templates for one asset class.

    The ``default`` template comes from the whole slice. When
    ``vri_top_n`` is set (landplot), one extra template per top-N
    ``vri`` value is built from that vri's sub-slice.
    """
    if objects.is_empty():
        raise ValueError("build_reference_objects: empty objects frame")

    refs = [_build_template(objects, variant="default", current_year=current_year)]

    if vri_top_n is not None and vri_top_n > 0 and "vri" in objects.columns:
        top_vri = (
            objects["vri"].drop_nulls().value_counts().sort("count", descending=True).head(vri_top_n)["vri"].to_list()
        )
        for vri in top_vri:
            slice_df = objects.filter(pl.col("vri") == vri)
            if slice_df.is_empty():
                continue
            refs.append(_build_template(slice_df, variant=str(vri), current_year=current_year))

    return refs

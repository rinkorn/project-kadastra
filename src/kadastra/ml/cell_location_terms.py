"""EBM term decomposition into a pure-location score (ADR-0029, product 2).

The EBM is additive: ``predict(x) = intercept + Σ fᵢ(xᵢ)`` (+ pairwise
interactions). ``location_score`` keeps only the terms whose every
feature is locational — object attributes and mixed
object×location interactions are dropped entirely (a mixed term's value
depends on the reference template, so it cannot be attributed to the
location alone).

**Semantics:** this is NOT a price prediction — it is the additive
value of the location with the object terms removed. See ADR-0029.
"""

from __future__ import annotations

import numpy as np

# Object-attribute features: the reference template fixes these
# (ADR-0029 «Продукт 1»). Everything else is locational.
_OBJECT_BASE_FEATURES: frozenset[str] = frozenset(
    {
        "levels",
        "flats",
        "area_m2",
        "year_built",
        "underground_floors",
        "materials",
        "vri",
        "category_zem",
        "era_category",
        "age_years",
        "age_years_sq",
        "polygon_area_m2",
        "polygon_perimeter_m",
        "polygon_compactness",
        "polygon_convexity",
        "bbox_aspect_ratio",
        "polygon_orientation_deg",
        "polygon_n_vertices",
    }
)

# Object attributes that carry relative (parent-vs-object) features
# (Settings.relative_feature_columns subset). Their ``*__rel_p*``
# columns depend on the object attribute value → object terms.
_OBJECT_REL_BASES: tuple[str, ...] = ("levels", "flats", "area_m2", "year_built")


def is_object_feature(name: str) -> bool:
    """True when the feature is an object attribute (or derived from one)."""
    if name in _OBJECT_BASE_FEATURES:
        return True
    return any(name.startswith(f"{base}__rel_") for base in _OBJECT_REL_BASES)


def sum_location_terms(
    term_values: np.ndarray,
    term_features: list[tuple[str, ...]],
    intercept: float,
) -> np.ndarray:
    """``intercept + Σ terms`` over terms whose features are ALL locational.

    ``term_values`` — (n_samples, n_terms) from ``EbmQuartetModel.eval_terms``;
    ``term_features`` — per-term feature names in the same order.
    """
    locational_mask = np.array(
        [not any(is_object_feature(name) for name in term) for term in term_features],
        dtype=bool,
    )
    return intercept + term_values[:, locational_mask].sum(axis=1)

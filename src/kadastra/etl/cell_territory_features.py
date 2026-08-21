"""Per-cell territory features (ADR-0029 «OKTMO-пропагация»).

Cells have no address or cadastral number, so the object-level path
(ГАР lookups) does not apply and no settlement polygons exist in the
data. Territory attributes (``oktmo_full``, ``settlement_name``,
``mun_okrug_*``, ``okato``, ``postal_index``, ``kadnum_quarter``) are
inherited from gold objects via **hierarchical H3 mode**: the modal
value among objects in the res-10 cell itself, falling back to the
mode of the res-9/8/7 parents, then null.

``intra_city_raion`` is an honest spatial join of the cell centroid
against OSM admin_level=9 raion polygons (same source as the object
pipeline, ADR-0015).

Approximation, documented in ADR-0029: on un-developed territory the
OKTMO comes from the nearest populated neighbourhood and is wrong at
municipality borders.
"""

from __future__ import annotations

import h3
import polars as pl
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

# Territory columns propagated from gold objects by hierarchical mode.
TERRITORY_PROPAGATED_COLUMNS: tuple[str, ...] = (
    "oktmo_full",
    "settlement_name",
    "mun_okrug_name",
    "mun_okrug_oktmo",
    "okato",
    "postal_index",
    "kadnum_quarter",
)

# Parent resolutions used as propagation fallback, in order.
PARENT_FALLBACK_RESOLUTIONS: tuple[int, ...] = (9, 8, 7)


def compute_cell_territory_features(
    cells: pl.DataFrame,
    objects: pl.DataFrame,
    *,
    raion_polygons: list[tuple[str, BaseGeometry]] | None = None,
    cell_resolution: int = 10,
) -> pl.DataFrame:
    """Attach territory columns to a ``(h3_index, lat, lon)`` cell frame.

    Returns ``h3_index`` + :data:`TERRITORY_PROPAGATED_COLUMNS` +
    ``intra_city_raion``. Cells whose whole fallback chain is
    object-free get nulls; columns absent from ``objects`` propagate
    as null.
    """
    out_cols = [*TERRITORY_PROPAGATED_COLUMNS, "intra_city_raion"]
    if cells.is_empty():
        return pl.DataFrame(
            schema={"h3_index": pl.Utf8, **{c: pl.Utf8 for c in out_cols}},
        )

    out = cells.select("h3_index")

    available = [c for c in TERRITORY_PROPAGATED_COLUMNS if c in objects.columns]
    if not objects.is_empty() and available:
        obj = objects.with_columns(
            pl.Series(
                "_key",
                [
                    h3.latlng_to_cell(lat, lon, cell_resolution)
                    for lat, lon in zip(objects["lat"].to_list(), objects["lon"].to_list(), strict=True)
                ],
                dtype=pl.Utf8,
            )
        )
        cell_keys = out.select("h3_index")
        for col in available:
            out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))
            # Fallback chain: the res-10 cell itself, then parent modes.
            for r in (cell_resolution, *PARENT_FALLBACK_RESOLUTIONS):
                if r == cell_resolution:
                    obj_keys = obj["_key"].alias("_level_key")
                    cell_level_keys = cell_keys["h3_index"].alias("_level_key")
                else:
                    obj_keys = pl.Series(
                        "_level_key",
                        [h3.cell_to_parent(c, r) for c in obj["_key"].to_list()],
                        dtype=pl.Utf8,
                    )
                    cell_level_keys = pl.Series(
                        "_level_key",
                        [h3.cell_to_parent(c, r) for c in cell_keys["h3_index"].to_list()],
                        dtype=pl.Utf8,
                    )
                modes = (
                    obj.with_columns(obj_keys)
                    .filter(pl.col(col).is_not_null())
                    .group_by("_level_key")
                    .agg(pl.col(col).mode().first())
                )
                joined = cell_keys.with_columns(cell_level_keys).join(modes, on="_level_key", how="left")
                out = out.with_columns(
                    pl.when(pl.col(col).is_null()).then(joined[col]).otherwise(pl.col(col)).alias(col)
                )
        for col in TERRITORY_PROPAGATED_COLUMNS:
            if col not in available:
                out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))
    else:
        out = out.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in TERRITORY_PROPAGATED_COLUMNS])

    # intra_city_raion: centroid point-in-polygon over OSM raions.
    raions: list[str | None] = []
    polygons = raion_polygons or []
    for lat, lon in zip(cells["lat"].to_list(), cells["lon"].to_list(), strict=True):
        point = Point(lon, lat)
        name = next((n for n, geom in polygons if geom.covers(point)), None)
        raions.append(name)
    out = out.with_columns(pl.Series("intra_city_raion", raions, dtype=pl.Utf8))

    return out

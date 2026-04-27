"""Build a per-OBJECTID lookup of extra ГАР PARAMS values.

Pivots active ``AS_HOUSES_PARAMS`` / ``AS_STEADS_PARAMS`` rows for the
TYPEIDs we care about into a wide table keyed on ``objectid``:

- TYPEID=7  → ``oktmo_full``     11-digit ОКТМО (settlement-grain)
- TYPEID=6  → ``okato``          11-digit ОКАТО (legacy)
- TYPEID=5  → ``postal_index``   ZIP / postal index

These are not in ``AS_MUN_HIERARCHY`` (which only carries the okrug-level
short OKTMO). The resulting table is joined into per-object features
on ``objectid`` after ``mun_lookup``.

Houses precedence on (objectid, typeid) collisions, like
``build_cadnum_index``: for an actual building the registry-of-record
is HOUSES, not the parcel.
"""

from __future__ import annotations

import polars as pl

# (typeid, friendly_column_name) — order is irrelevant; pivot columns
# are explicitly renamed at the end so missing TYPEIDs surface as
# all-null columns rather than missing-column errors downstream.
_PIVOT_COLUMNS: dict[int, str] = {
    7: "oktmo_full",
    6: "okato",
    5: "postal_index",
}


def build_object_params_lookup(*, houses: pl.DataFrame, steads: pl.DataFrame) -> pl.DataFrame:
    wanted_typeids = list(_PIVOT_COLUMNS.keys())

    def _trim(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("typeid").is_in(wanted_typeids)).select("objectid", "typeid", "value")

    combined = pl.concat([_trim(houses), _trim(steads)], how="vertical")
    if combined.is_empty():
        # Empty pivot would produce a frame with only the index column;
        # fabricate the full schema instead so downstream joins don't
        # have to special-case missing columns.
        return pl.DataFrame(
            schema={
                "objectid": pl.Int64,
                **{name: pl.Utf8 for name in _PIVOT_COLUMNS.values()},
            }
        )
    deduped = combined.unique(subset=["objectid", "typeid"], keep="first", maintain_order=True)
    pivoted = deduped.pivot(
        on="typeid",
        index="objectid",
        values="value",
        aggregate_function="first",
    )
    rename_map = {str(tid): name for tid, name in _PIVOT_COLUMNS.items()}
    pivoted = pivoted.rename({k: v for k, v in rename_map.items() if k in pivoted.columns})
    # Ensure all expected columns exist even if no row had that TYPEID.
    missing_columns = [
        pl.lit(None, dtype=pl.Utf8).alias(name) for name in _PIVOT_COLUMNS.values() if name not in pivoted.columns
    ]
    if missing_columns:
        pivoted = pivoted.with_columns(missing_columns)
    return pivoted.select(["objectid", *_PIVOT_COLUMNS.values()])

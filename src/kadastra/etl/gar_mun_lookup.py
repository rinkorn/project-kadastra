"""Build per-leaf-objectid municipality attributes from parsed ГАР.

Joins ``AS_MUN_HIERARCHY`` with ``AS_ADDR_OBJ``: explodes the
dot-separated ``PATH`` of each row into individual ancestor objectids
and looks up each segment's name and FIAS LEVEL. Then for each leaf
objectid we pick:

- ``mun_okrug_name`` / ``mun_okrug_oktmo`` — the closest-to-leaf segment
  with FIAS LEVEL ∈ {2, 3}. In Татарстан this catches both
  ``р-н`` / ``м.р-н`` (rural municipal districts) and ``г.о.`` (urban
  okrug). OKTMO is read from the mun-hierarchy row of that segment
  itself, not the leaf's row, so ``mun_okrug_oktmo`` is the short
  municipal-level OKTMO (e.g. "92701000" for Казань urban okrug).

- ``settlement_name`` — the closest-to-leaf segment with FIAS
  LEVEL ∈ {4, 5, 6}: city (``г``), urban-type settlement (``г.п.``),
  rural settlement (``с.п.``), village (``с``, ``д``, ``п``).

Rows whose path doesn't contain a level-{2,3}-or-{4,5,6} segment get
NULL for the missing field. ADR-0015 covers the empirical rationale
for the level ranges (FIAS schema for Татарстан).
"""

from __future__ import annotations

import polars as pl


def build_mun_lookup(addr_obj: pl.DataFrame, mun_hierarchy: pl.DataFrame) -> pl.DataFrame:
    if mun_hierarchy.is_empty():
        return pl.DataFrame(
            schema={
                "objectid": pl.Int64,
                "mun_okrug_name": pl.Utf8,
                "mun_okrug_oktmo": pl.Utf8,
                "settlement_name": pl.Utf8,
            }
        )

    # Per-objectid OKTMO lookup, used to attach OKTMO at the okrug-level
    # ancestor (not the leaf's own OKTMO, which is finer-grained).
    seg_oktmo = mun_hierarchy.lazy().select(
        pl.col("objectid").alias("segment_id"),
        pl.col("oktmo").alias("segment_oktmo"),
    )
    seg_levels = addr_obj.lazy().select(
        pl.col("objectid").alias("segment_id"),
        pl.col("name").alias("segment_name"),
        pl.col("level").alias("segment_level"),
    )

    exploded = (
        mun_hierarchy.lazy()
        .select(["objectid", "path"])
        .with_columns(pl.col("path").str.split(".").alias("segments"))
        .explode("segments")
        .with_columns(pl.col("segments").cast(pl.Int64).alias("segment_id"))
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int32).over("objectid").alias("seg_pos"))
        .drop("path", "segments")
    )

    annotated = exploded.join(seg_levels, on="segment_id", how="left").join(seg_oktmo, on="segment_id", how="left")

    okrug = (
        annotated.filter(pl.col("segment_level").is_in([2, 3]))
        .sort(["objectid", "seg_pos"], descending=[False, True])
        .group_by("objectid", maintain_order=True)
        .first()
        .select(
            pl.col("objectid"),
            pl.col("segment_name").alias("mun_okrug_name"),
            pl.col("segment_oktmo").alias("mun_okrug_oktmo"),
        )
    )
    settlement = (
        annotated.filter(pl.col("segment_level").is_in([4, 5, 6]))
        .sort(["objectid", "seg_pos"], descending=[False, True])
        .group_by("objectid", maintain_order=True)
        .first()
        .select(
            pl.col("objectid"),
            pl.col("segment_name").alias("settlement_name"),
        )
    )

    leaves = mun_hierarchy.lazy().select("objectid").unique(maintain_order=True)
    return leaves.join(okrug, on="objectid", how="left").join(settlement, on="objectid", how="left").collect()

"""Build a unified ``cad_num → objectid`` lookup from ГАР PARAMS.

Consumes parsed ``AS_HOUSES_PARAMS`` / ``AS_STEADS_PARAMS`` DataFrames
and projects to the ``TYPEID=8`` (CADNUM) rows. Concatenates them
with a ``source`` tag — houses take precedence on collisions, since
for an actual building the registry-of-record is the houses table,
not the parcel the building sits on.

The output is the bridge between NSPD's free-form ``cad_num`` field
and ГАР's canonical ``OBJECTID``, which then walks ``MUN_HIERARCHY``
into a municipal okrug + OKTMO via ``build_mun_lookup``.

The parsed PARAMS frames may include other TYPEIDs (when the caller
widens the parser whitelist for a parallel ``object_params`` lookup
build); we filter to TYPEID=8 here so the builder is robust against
that.
"""

from __future__ import annotations

import polars as pl

_TYPEID_CADNUM = 8


def build_cadnum_index(
    *, houses: pl.DataFrame, steads: pl.DataFrame
) -> pl.DataFrame:
    def _tag(df: pl.DataFrame, source: str) -> pl.DataFrame:
        return (
            df.lazy()
            .filter(pl.col("typeid") == _TYPEID_CADNUM)
            .select(
                pl.col("value").alias("cad_num"),
                pl.col("objectid"),
                pl.lit(source, dtype=pl.Utf8).alias("source"),
            )
            .unique(subset=["cad_num"], keep="first", maintain_order=True)
            .collect()
        )

    houses_tagged = _tag(houses, "house")
    steads_tagged = _tag(steads, "stead")
    combined = pl.concat([houses_tagged, steads_tagged], how="vertical")
    return combined.unique(subset=["cad_num"], keep="first", maintain_order=True)

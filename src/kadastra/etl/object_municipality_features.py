"""Block 4 — territorial features per object.

Attaches up to 8 columns to a per-object DataFrame:

- ``mun_okrug_name``  (Utf8)         — municipal okrug.
- ``mun_okrug_oktmo`` (Utf8)         — short OKTMO of that okrug.
- ``settlement_name`` (Utf8)         — city/town/village.
- ``intra_city_raion`` (Utf8 / null) — Kazan raion etc.
- ``mun_source`` (Utf8)              — "gar" or "address".
- ``oktmo_full``     (Utf8 / null)   — 11-digit settlement-grain OKTMO.
- ``okato``          (Utf8 / null)   — 11-digit ОКАТО (legacy).
- ``postal_index``   (Utf8 / null)   — ZIP / postal index.

The last three are only populated when an ``object_params`` lookup
(from ``AS_*_PARAMS`` pivot) is provided and the object's ``cad_num``
matches a row there. They are GAR-only — the address regex path
cannot derive 11-digit OKTMO / OKATO / ZIP from NSPD ``readable_address``.

Two-stage resolution for okrug/OKTMO/settlement:

1. **ГАР primary**: ``cad_num`` joins through ``cadnum_index`` to the
   ГАР leaf ``objectid`` and then through ``mun_lookup`` to the
   canonical okrug + OKTMO + settlement (~35 % of NSPD objects).
2. **Address fallback**: for rows still null after step 1 we parse
   ``readable_address`` (NSPD-formatted) with regex. The okrug name
   resolves through ``name_to_oktmo`` derived from ``mun_lookup`` so
   address-parsed rows share OKTMO codes with ГАР-matched ones for
   the same okrug.

``intra_city_raion`` resolution: OSM admin_level=9 polygon spatial
join (primary, when ``intra_raion_polygons`` is provided) → address
regex fallback. ФИАС does not model intra-Kazan raions, hence the
OSM-driven path. The polygon path is exact (no spelling variation,
no boundary text ambiguity) and works for objects whose address
omits the raion segment.
"""

from __future__ import annotations

import polars as pl
import shapely
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

# Polars regex uses Rust's `regex` crate — no look-around. Each
# pattern captures up to the next comma (or end of string).
# Trailing whitespace is stripped after extract.

# "Высокогорский муниципальный район" — match ONLY the explicit
# "муниципальный район" form (NSPD-structured addresses always
# carry the full word). The bare "X район" form is ambiguous: it
# matches both rural raions ("Высокогорский район") and intra-city
# raions ("Советский район"); the latter belongs in
# ``intra_city_raion`` and would mis-fill mun_okrug here.
_RX_RURAL_OKRUG = r"(?:^|,\s*)([А-Я][а-яА-Я\-]+?)\s+муниципальный\s+район"
# "г.о. город Казань"
_RX_URBAN_OKRUG = r"г\.о\.\s*([^,]+)"
# "г Казань" / "город Казань" (must be its own token after a comma; the
# whitespace-after requirement is what disqualifies "г.о." here).
_RX_CITY = r"(?:^|,\s*)(?:г|город)\s+([А-Я][^,]+)"
# Generic village/settlement marker.
_RX_VILLAGE = r"(?:^|,\s*)(?:д|с|п|пос|дер|пгт)\s+([А-Я][^,]+)"
# Intra-Kazan raions — ФИАС не моделирует, держим закрытый список.
_INTRA_KAZAN_RAIONS = (
    "Советский",
    "Приволжский",
    "Авиастроительный",
    "Кировский",
    "Московский",
    "Вахитовский",
    "Ново-Савиновский",
)
_RX_INTRA = r"(" + r"|".join(_INTRA_KAZAN_RAIONS) + r")\s+район"


def _build_name_to_oktmo(mun_lookup: pl.DataFrame) -> pl.DataFrame:
    """Bridge from okrug name to its canonical short OKTMO. Built from
    mun_lookup itself so address-derived rows share codes with ГАР-
    derived ones for the same okrug. Drops nulls and dedups by name."""
    return (
        mun_lookup.lazy()
        .filter(
            pl.col("mun_okrug_name").is_not_null()
            & pl.col("mun_okrug_oktmo").is_not_null()
            & (pl.col("mun_okrug_oktmo") != "")
        )
        .group_by("mun_okrug_name")
        .agg(pl.col("mun_okrug_oktmo").first())
        .rename({"mun_okrug_oktmo": "mun_okrug_oktmo_addr"})
        .collect()
    )


def _intra_raion_via_polygons(
    lats: list[float],
    lons: list[float],
    named_polygons: list[tuple[str, BaseGeometry]],
) -> list[str | None]:
    """Point-in-polygon assignment for intra-city raions.

    ``named_polygons``: ``(short_name, polygon_or_multipolygon)`` pairs
    in WGS84. Polygons must not overlap (Kazan raions tile the city);
    if they do, the first match wins. Returns one entry per input
    coordinate, ``None`` for points outside every polygon.
    """
    n = len(lats)
    if n == 0 or not named_polygons:
        return [None] * n
    geoms = [g for _, g in named_polygons]
    names = [n_ for n_, _ in named_polygons]
    tree = STRtree(geoms)
    points = shapely.points(lons, lats)  # x=lon, y=lat
    # shapely 2 STRtree: predicate `intersects` ↔ input.intersects(tree).
    # Returns 2×K ndarray where row 0 = input index, row 1 = tree index.
    pairs = tree.query(points, predicate="intersects")
    result: list[str | None] = [None] * n
    for k in range(pairs.shape[1]):
        input_idx = int(pairs[0, k])
        tree_idx = int(pairs[1, k])
        if result[input_idx] is None:
            result[input_idx] = names[tree_idx]
    return result


_OBJECT_PARAMS_COLUMNS = ("oktmo_full", "okato", "postal_index")


def compute_object_municipality_features(
    objects: pl.DataFrame,
    *,
    cadnum_index: pl.DataFrame,
    mun_lookup: pl.DataFrame,
    object_params: pl.DataFrame | None = None,
    intra_raion_polygons: list[tuple[str, BaseGeometry]] | None = None,
) -> pl.DataFrame:
    # Drop any output columns left from a previous run on the same
    # partition: re-running the pipeline reads the already-enriched
    # parquet, and a join with mun_lookup that adds ``mun_okrug_name``
    # etc. on top of pre-existing columns produces ``_right``-suffixed
    # duplicates and breaks the final select. Idempotency is the
    # contract — recompute always.
    _OUTPUT_COLUMNS = (
        "mun_okrug_name",
        "mun_okrug_oktmo",
        "settlement_name",
        "intra_city_raion",
        "mun_source",
        *_OBJECT_PARAMS_COLUMNS,
    )
    drop_existing = [c for c in _OUTPUT_COLUMNS if c in objects.columns]
    if drop_existing:
        objects = objects.drop(drop_existing)

    n = objects.height
    if n == 0:
        # Preserve input schema + add the new columns as Utf8 nulls.
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias(c)
                for c in _OUTPUT_COLUMNS
            ]
        )

    name_to_oktmo = _build_name_to_oktmo(mun_lookup)

    addr = pl.col("readable_address").fill_null("")

    # Optional polygon-based intra_city_raion (Kazan + others if seeded).
    if intra_raion_polygons:
        lats = objects.get_column("lat").to_list()
        lons = objects.get_column("lon").to_list()
        poly_raions = _intra_raion_via_polygons(lats, lons, intra_raion_polygons)
        objects = objects.with_columns(
            pl.Series("_intra_raion_poly", poly_raions, dtype=pl.Utf8)
        )
    else:
        objects = objects.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("_intra_raion_poly")
        )

    # Stage 1: GAR lookup via cad_num. Joins are chained with
    # ``objectid`` left in scope so the optional object_params join
    # (settlement-grain OKTMO + ОКАТО + ZIP) can key off it before we
    # drop the column.
    gar_join_lf = (
        objects.lazy()
        .join(cadnum_index.lazy().select(["cad_num", "objectid"]), on="cad_num", how="left")
        .join(mun_lookup.lazy(), on="objectid", how="left")
    )
    if object_params is not None and object_params.height > 0:
        gar_join_lf = gar_join_lf.join(
            object_params.lazy().select(
                ["objectid", *_OBJECT_PARAMS_COLUMNS]
            ),
            on="objectid",
            how="left",
        )
    else:
        gar_join_lf = gar_join_lf.with_columns(
            [pl.lit(None, dtype=pl.Utf8).alias(c) for c in _OBJECT_PARAMS_COLUMNS]
        )
    gar_join = gar_join_lf.drop("objectid")

    # Stage 2: address parse + intra_raion + okrug→OKTMO bridge.
    enriched = (
        gar_join
        # Address-extracted okrug (rural pattern OR urban "г.о." pattern).
        .with_columns(
            addr.str.extract(_RX_RURAL_OKRUG, group_index=1).alias(
                "_okrug_addr_rural"
            ),
            addr.str.extract(_RX_URBAN_OKRUG, group_index=1).alias(
                "_okrug_addr_urban"
            ),
            addr.str.extract(_RX_CITY, group_index=1).alias("_city_addr"),
            addr.str.extract(_RX_VILLAGE, group_index=1).alias("_vil_addr"),
            addr.str.extract(_RX_INTRA, group_index=1).alias("_intra_raion_addr"),
        )
        # Clean address-extracted strings: strip whitespace/punctuation
        # and trim everything from " муниципальный ..." or
        # " {INTRA_KAZAN_RAION} район" — those tokens leak into the
        # urban-okrug capture when the original text has no comma
        # before them (e.g. "г.о. город Казань Советский район").
        .with_columns(
            pl.col("_okrug_addr_urban")
            .str.replace(
                r"\s+(?:муниципальный\s+район|"
                + r"|".join(_INTRA_KAZAN_RAIONS)
                + r"\s+район).*$",
                "",
            )
            .str.strip_chars(" \t.,;")
            .alias("_okrug_addr_urban_clean"),
            pl.col("_city_addr")
            .str.replace(
                r"\s+(?:муниципальный\s+район|"
                + r"|".join(_INTRA_KAZAN_RAIONS)
                + r"\s+район).*$",
                "",
            )
            .str.strip_chars(" \t.,;")
            .alias("_city_addr_clean"),
        )
        # Settlement: ГАР first, else city ("г X"), else village.
        .with_columns(
            pl.coalesce(
                pl.col("settlement_name"),
                pl.col("_city_addr_clean"),
                pl.col("_vil_addr").str.strip_chars(),
            ).alias("settlement_name_resolved"),
        )
        # Okrug: ГАР first, else "г.о. X", else "X муниципальный район",
        # else infer "город {settlement}" for objects that just say
        # "г Казань" without г.о. — most NSPD entries inside Kazan
        # city lack the г.о. prefix despite belonging to the okrug.
        .with_columns(
            pl.coalesce(
                pl.col("mun_okrug_name"),
                pl.col("_okrug_addr_urban_clean"),
                pl.col("_okrug_addr_rural").str.strip_chars(),
                pl.when(pl.col("_city_addr_clean").is_not_null())
                .then(pl.lit("город ") + pl.col("_city_addr_clean"))
                .otherwise(None),
            ).alias("mun_okrug_name_resolved"),
        )
    )
    # Bridge resolved okrug name → OKTMO (mun_lookup drops nulls/empties).
    enriched = (
        enriched.join(
            name_to_oktmo.lazy(),
            left_on="mun_okrug_name_resolved",
            right_on="mun_okrug_name",
            how="left",
        )
        .with_columns(
            # OKTMO: ГАР first; else from name bridge.
            pl.coalesce(
                pl.col("mun_okrug_oktmo"),
                pl.col("mun_okrug_oktmo_addr"),
            ).alias("mun_okrug_oktmo_resolved"),
        )
    )

    # Source tag: "gar" if mun_okrug_name was set by ГАР join, else "address".
    enriched = enriched.with_columns(
        pl.when(pl.col("mun_okrug_name").is_not_null())
        .then(pl.lit("gar"))
        .otherwise(pl.lit("address"))
        .alias("mun_source"),
    )

    # Resolve intra_city_raion: polygon (primary) → address regex (fallback).
    enriched = enriched.with_columns(
        pl.coalesce(
            pl.col("_intra_raion_poly"),
            pl.col("_intra_raion_addr"),
        ).alias("intra_city_raion_resolved"),
    )

    # Final shape: drop intermediates (including the temp `_intra_raion_poly`
    # we added to ``objects`` for the spatial-join step), rename
    # ``_resolved`` → final names.
    output_columns = [c for c in objects.columns if c != "_intra_raion_poly"]
    final = enriched.select(
        [pl.col(c) for c in output_columns]
        + [
            pl.col("mun_okrug_name_resolved").alias("mun_okrug_name"),
            pl.col("mun_okrug_oktmo_resolved").alias("mun_okrug_oktmo"),
            pl.col("settlement_name_resolved").alias("settlement_name"),
            pl.col("intra_city_raion_resolved").alias("intra_city_raion"),
            pl.col("mun_source"),
            pl.col("oktmo_full"),
            pl.col("okato"),
            pl.col("postal_index"),
        ]
    )
    return final.collect()

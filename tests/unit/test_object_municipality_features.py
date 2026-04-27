"""Tests for compute_object_municipality_features.

Adds 5 columns to a per-object DataFrame:

- ``mun_okrug_name``  (cat) — municipal okrug.
- ``mun_okrug_oktmo`` (cat) — short-form OKTMO of that okrug.
- ``settlement_name`` (cat) — city / town / village.
- ``intra_city_raion`` (cat, nullable) — Kazan raion etc., parsed from
  address text. Null outside intra-city contexts.
- ``mun_source``      (cat) — "gar" if matched via cad_num→ГАР,
  "address" if filled in from NSPD ``readable_address`` parse.

ГАР is the primary source: ``cad_num`` joins through ``cadnum_index``
to ``mun_lookup``. Whatever stays null after the join is filled by
parsing ``readable_address`` and inferring the okrug from text. The
``name_to_oktmo`` map (built from ``mun_lookup`` itself) bridges the
parsed okrug name to its canonical OKTMO so address-derived rows get
the same code as ГАР-derived ones for the same okrug.
"""

from __future__ import annotations

import polars as pl
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from kadastra.etl.object_municipality_features import (
    compute_object_municipality_features,
)


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "cad_num": pl.Utf8,
            "readable_address": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
        },
    )


def _cadnum_ix(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"cad_num": pl.Utf8, "objectid": pl.Int64, "source": pl.Utf8},
    )


def _mun(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "objectid": pl.Int64,
            "mun_okrug_name": pl.Utf8,
            "mun_okrug_oktmo": pl.Utf8,
            "settlement_name": pl.Utf8,
        },
    )


def test_gar_primary_match() -> None:
    """When cad_num is in ГАР, the row gets the canonical
    okrug+OKTMO+settlement and mun_source='gar'."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "16:50:1:1",
                "readable_address": "Татарстан, г.о. город Казань, г Казань, ул Пушкина",
                "lat": 55.78,
                "lon": 49.12,
            }
        ]
    )
    ix = _cadnum_ix([{"cad_num": "16:50:1:1", "objectid": 1, "source": "house"}])
    mun = _mun(
        [
            {
                "objectid": 1,
                "mun_okrug_name": "город Казань",
                "mun_okrug_oktmo": "92701000",
                "settlement_name": "Казань",
            }
        ]
    )

    out = compute_object_municipality_features(objects, cadnum_index=ix, mun_lookup=mun)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["mun_okrug_name"] == "город Казань"
    assert row["mun_okrug_oktmo"] == "92701000"
    assert row["settlement_name"] == "Казань"
    assert row["mun_source"] == "gar"


def test_address_fallback_for_unmatched_kazan_object() -> None:
    """No GAR match → parse ``г.о. город Казань`` from address; the
    OKTMO comes from the name→OKTMO bridge derived from mun_lookup."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "16:50:999:999",
                "readable_address": (
                    "Российская Федерация, Республика Татарстан "
                    "(Татарстан), г.о. город Казань, г Казань, "
                    "тер. ГСК Северный"
                ),
                "lat": 55.79,
                "lon": 49.20,
            }
        ]
    )
    # Empty cadnum index = no GAR match. mun_lookup still has Kazan
    # entry so the address-parsed name "город Казань" can be bridged
    # to OKTMO via name uniqueness in mun_lookup.
    ix = _cadnum_ix([])
    mun = _mun(
        [
            {
                "objectid": 1,
                "mun_okrug_name": "город Казань",
                "mun_okrug_oktmo": "92701000",
                "settlement_name": "Казань",
            }
        ]
    )

    out = compute_object_municipality_features(objects, cadnum_index=ix, mun_lookup=mun)

    row = out.row(0, named=True)
    assert row["mun_okrug_name"] == "город Казань"
    assert row["mun_okrug_oktmo"] == "92701000"
    assert row["settlement_name"] == "Казань"
    assert row["mun_source"] == "address"


def test_address_fallback_for_rural_raion() -> None:
    """Rural object whose address mentions ``Высокогорский
    муниципальный район`` should pick up that okrug from text."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "16:16:999:999",
                "readable_address": (
                    "Республика Татарстан, Высокогорский муниципальный район, Семиозерское сельское поселение, СНТ 'X'"
                ),
                "lat": 55.96,
                "lon": 49.30,
            }
        ]
    )
    ix = _cadnum_ix([])
    mun = _mun(
        [
            {
                "objectid": 99,
                "mun_okrug_name": "Высокогорский",
                "mun_okrug_oktmo": "92215000",
                "settlement_name": "Большие Кабаны",
            }
        ]
    )

    out = compute_object_municipality_features(objects, cadnum_index=ix, mun_lookup=mun)

    row = out.row(0, named=True)
    assert row["mun_okrug_name"] == "Высокогорский"
    assert row["mun_okrug_oktmo"] == "92215000"
    assert row["mun_source"] == "address"


def test_intra_city_raion_parsed_from_address() -> None:
    """Kazan addresses that mention ``Советский район`` (etc.)
    populate ``intra_city_raion`` via regex — ГАР does not model
    intra-Kazan raions at any level."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "16:50:1:1",
                "readable_address": ("Республика Татарстан, г Казань, Советский район, ул Пушкина, д. 5"),
                "lat": 55.78,
                "lon": 49.12,
            },
            {
                "object_id": "b",
                "cad_num": "16:50:2:2",
                "readable_address": ("Татарстан, г.о. город Казань, г Казань, ул Пушкина"),
                "lat": 55.78,
                "lon": 49.12,
            },
        ]
    )
    ix = _cadnum_ix(
        [
            {"cad_num": "16:50:1:1", "objectid": 1, "source": "house"},
            {"cad_num": "16:50:2:2", "objectid": 2, "source": "house"},
        ]
    )
    mun = _mun(
        [
            {
                "objectid": 1,
                "mun_okrug_name": "город Казань",
                "mun_okrug_oktmo": "92701000",
                "settlement_name": "Казань",
            },
            {
                "objectid": 2,
                "mun_okrug_name": "город Казань",
                "mun_okrug_oktmo": "92701000",
                "settlement_name": "Казань",
            },
        ]
    )

    out = compute_object_municipality_features(objects, cadnum_index=ix, mun_lookup=mun).sort("object_id")

    assert out["intra_city_raion"].to_list() == ["Советский", None]


def test_no_matches_returns_nulls_with_address_source() -> None:
    """When neither GAR nor address-parse can resolve an okrug,
    columns stay null but mun_source is still 'address' (i.e. we
    tried the address path and it didn't find anything)."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "x",
                "readable_address": "Some unparseable string",
                "lat": 0.0,
                "lon": 0.0,
            }
        ]
    )
    out = compute_object_municipality_features(objects, cadnum_index=_cadnum_ix([]), mun_lookup=_mun([]))
    row = out.row(0, named=True)
    assert row["mun_okrug_name"] is None
    assert row["mun_okrug_oktmo"] is None
    assert row["settlement_name"] is None
    assert row["intra_city_raion"] is None
    assert row["mun_source"] == "address"


def _params_lookup(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "objectid": pl.Int64,
            "oktmo_full": pl.Utf8,
            "okato": pl.Utf8,
            "postal_index": pl.Utf8,
        },
    )


def test_object_params_join_when_cadnum_matches() -> None:
    """When cad_num matches and object_params has the OBJECTID,
    oktmo_full / okato / postal_index get filled."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "16:50:1:1",
                "readable_address": "Татарстан, г Казань, ул X",
                "lat": 55.78,
                "lon": 49.12,
            }
        ]
    )
    ix = _cadnum_ix([{"cad_num": "16:50:1:1", "objectid": 1, "source": "house"}])
    mun = _mun(
        [
            {
                "objectid": 1,
                "mun_okrug_name": "город Казань",
                "mun_okrug_oktmo": "92701000",
                "settlement_name": "Казань",
            }
        ]
    )
    params = _params_lookup(
        [
            {
                "objectid": 1,
                "oktmo_full": "92701000001",
                "okato": "92401000001",
                "postal_index": "420000",
            }
        ]
    )

    out = compute_object_municipality_features(objects, cadnum_index=ix, mun_lookup=mun, object_params=params)
    row = out.row(0, named=True)
    assert row["oktmo_full"] == "92701000001"
    assert row["okato"] == "92401000001"
    assert row["postal_index"] == "420000"


def test_object_params_columns_null_when_no_lookup_provided() -> None:
    """If object_params is None (lookup file missing), the 3 columns
    still exist in the output but are filled with nulls."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "x",
                "readable_address": "Татарстан, г Казань, ул X",
                "lat": 55.78,
                "lon": 49.12,
            }
        ]
    )
    out = compute_object_municipality_features(objects, cadnum_index=_cadnum_ix([]), mun_lookup=_mun([]))
    assert "oktmo_full" in out.columns
    assert "okato" in out.columns
    assert "postal_index" in out.columns
    row = out.row(0, named=True)
    assert row["oktmo_full"] is None
    assert row["okato"] is None
    assert row["postal_index"] is None


def test_object_params_null_for_unmatched_cadnum() -> None:
    """cad_num present in input but no GAR match → object_params join
    yields nulls (no row to copy over)."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "16:50:99:99",
                "readable_address": "Татарстан, г Казань",
                "lat": 55.78,
                "lon": 49.12,
            }
        ]
    )
    out = compute_object_municipality_features(
        objects,
        cadnum_index=_cadnum_ix([]),
        mun_lookup=_mun([]),
        object_params=_params_lookup(
            [
                {
                    "objectid": 1,
                    "oktmo_full": "92701000001",
                    "okato": "92401000001",
                    "postal_index": "420000",
                }
            ]
        ),
    )
    row = out.row(0, named=True)
    assert row["oktmo_full"] is None
    assert row["okato"] is None
    assert row["postal_index"] is None


def test_intra_raion_via_polygon_takes_precedence_over_address() -> None:
    """When ``intra_raion_polygons`` is provided, the spatial join
    takes precedence over the address regex. A point inside the
    Sovetsky polygon must come back as 'Советский' even if the
    address text contains a different raion name."""
    sovetsky = Polygon([(49.10, 55.77), (49.20, 55.77), (49.20, 55.82), (49.10, 55.82)])
    vahit = Polygon([(48.95, 55.75), (49.05, 55.75), (49.05, 55.80), (48.95, 55.80)])
    polygons: list[tuple[str, BaseGeometry]] = [("Советский", sovetsky), ("Вахитовский", vahit)]

    objects = _objects(
        [
            # Point inside Советский — address mentions Вахитовский,
            # but polygon must win.
            {
                "object_id": "a",
                "cad_num": "16:50:1:1",
                "readable_address": ("Татарстан, г.о. город Казань, г Казань, Вахитовский район, ул X"),
                "lat": 55.79,
                "lon": 49.15,
            },
            # Point outside both polygons — falls back to address regex.
            {
                "object_id": "b",
                "cad_num": "16:50:2:2",
                "readable_address": ("Татарстан, г.о. город Казань, г Казань, Приволжский район, ул Y"),
                "lat": 55.60,
                "lon": 49.50,
            },
        ]
    )

    out = compute_object_municipality_features(
        objects,
        cadnum_index=_cadnum_ix([]),
        mun_lookup=_mun([]),
        intra_raion_polygons=polygons,
    ).sort("object_id")

    assert out["intra_city_raion"].to_list() == ["Советский", "Приволжский"]


def test_intra_raion_polygon_fills_when_address_lacks_raion() -> None:
    """Polygon path resolves intra_city_raion for objects whose address
    omits the raion segment entirely (a known NSPD pattern)."""
    sovetsky = Polygon([(49.10, 55.77), (49.20, 55.77), (49.20, 55.82), (49.10, 55.82)])
    polygons: list[tuple[str, BaseGeometry]] = [("Советский", sovetsky)]

    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "x",
                "readable_address": ("Татарстан, г.о. город Казань, г Казань, ул Пушкина, д 5"),
                "lat": 55.79,
                "lon": 49.15,
            }
        ]
    )

    out = compute_object_municipality_features(
        objects,
        cadnum_index=_cadnum_ix([]),
        mun_lookup=_mun([]),
        intra_raion_polygons=polygons,
    )

    assert out.row(0, named=True)["intra_city_raion"] == "Советский"


def test_intra_raion_polygons_empty_list_behaves_like_none() -> None:
    """An empty list means no polygon source; address regex still works."""
    objects = _objects(
        [
            {
                "object_id": "a",
                "cad_num": "x",
                "readable_address": "Татарстан, г Казань, Советский район, ул X",
                "lat": 55.79,
                "lon": 49.15,
            }
        ]
    )
    out = compute_object_municipality_features(
        objects,
        cadnum_index=_cadnum_ix([]),
        mun_lookup=_mun([]),
        intra_raion_polygons=[],
    )
    assert out.row(0, named=True)["intra_city_raion"] == "Советский"


def test_idempotent_when_input_already_has_output_columns() -> None:
    """A second run on already-enriched partitions reads parquet that
    contains ``mun_okrug_name`` etc. — the function must drop them
    first and recompute, otherwise the polars join produces
    ``_right``-suffixed conflicts and the final select fails."""
    objects = pl.DataFrame(
        [
            {
                "object_id": "a",
                "cad_num": "16:50:1:1",
                "readable_address": "Татарстан, г.о. город Казань, г Казань, ул X",
                "lat": 55.78,
                "lon": 49.12,
                "mun_okrug_name": "stale value",
                "mun_okrug_oktmo": "stale oktmo",
                "settlement_name": "stale settlement",
                "intra_city_raion": "stale raion",
                "mun_source": "stale",
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "cad_num": pl.Utf8,
            "readable_address": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "mun_okrug_name": pl.Utf8,
            "mun_okrug_oktmo": pl.Utf8,
            "settlement_name": pl.Utf8,
            "intra_city_raion": pl.Utf8,
            "mun_source": pl.Utf8,
        },
    )
    ix = _cadnum_ix([{"cad_num": "16:50:1:1", "objectid": 1, "source": "house"}])
    mun = _mun(
        [
            {
                "objectid": 1,
                "mun_okrug_name": "город Казань",
                "mun_okrug_oktmo": "92701000",
                "settlement_name": "Казань",
            }
        ]
    )

    out = compute_object_municipality_features(objects, cadnum_index=ix, mun_lookup=mun)

    row = out.row(0, named=True)
    assert row["mun_okrug_name"] == "город Казань"
    assert row["mun_okrug_oktmo"] == "92701000"
    assert row["settlement_name"] == "Казань"
    assert row["mun_source"] == "gar"
    # No `_right`-suffixed survivors:
    assert all(not c.endswith("_right") for c in out.columns)


def test_preserves_input_columns_and_order() -> None:
    objects = _objects(
        [
            {"object_id": "z", "cad_num": "x", "readable_address": "x", "lat": 0.0, "lon": 0.0},
            {"object_id": "a", "cad_num": "y", "readable_address": "y", "lat": 1.0, "lon": 1.0},
        ]
    )
    out = compute_object_municipality_features(objects, cadnum_index=_cadnum_ix([]), mun_lookup=_mun([]))
    assert out["object_id"].to_list() == ["z", "a"]
    for col in ("object_id", "cad_num", "readable_address", "lat", "lon"):
        assert col in out.columns

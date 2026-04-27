"""Tests for build_mun_lookup.

Joins parsed AS_ADDR_OBJ + AS_MUN_HIERARCHY DataFrames so each leaf
``objectid`` carries the closest-to-leaf mun-okrug ancestor name + that
ancestor's OKTMO short code, plus the settlement name (city / town /
village) further up the path.

Mun-okrug mapping inside Татарстан:

- LEVEL=2 ``р-н`` — rural municipal districts (e.g. "Высокогорский").
- LEVEL=3 ``м.р-н`` — newer naming for the same municipal districts.
- LEVEL=3 ``г.о.``  — urban okrug (e.g. "г.о. город Казань").

So we treat LEVEL ∈ {2, 3} as "okrug-level"; we pick whichever such
ancestor sits closest to the leaf along the dot-separated PATH
(``root → ... → leaf``).

Settlement mapping picks LEVEL ∈ {4, 5, 6}: city (``г``), urban-type
settlement (``г.п.``), rural settlement (``с.п.``), village (``с``,
``д``, ``п``).
"""

from __future__ import annotations

import polars as pl

from kadastra.etl.gar_mun_lookup import build_mun_lookup


def _addr(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "objectid": pl.Int64,
            "objectguid": pl.Utf8,
            "name": pl.Utf8,
            "typename": pl.Utf8,
            "level": pl.Int8,
        },
    )


def _mun(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "objectid": pl.Int64,
            "parentobjid": pl.Int64,
            "oktmo": pl.Utf8,
            "path": pl.Utf8,
        },
    )


def test_picks_kazan_urban_okrug() -> None:
    """Path for a Kazan house resembles
    169363 → 95235088 (г.о.) → 169398 (г Казань) → 172729 (street) →
    leaf. We expect mun_okrug_name="город Казань", oktmo="92701000",
    settlement_name="Казань"."""
    addr = _addr(
        [
            {"objectid": 169363, "objectguid": "g1", "name": "Татарстан", "typename": "Респ", "level": 1},
            {"objectid": 95235088, "objectguid": "g2", "name": "город Казань", "typename": "г.о.", "level": 3},
            {"objectid": 169398, "objectguid": "g3", "name": "Казань", "typename": "г", "level": 5},
            {"objectid": 172729, "objectguid": "g4", "name": "Гагарина", "typename": "ул", "level": 8},
        ]
    )
    mun = _mun(
        [
            {"objectid": 169363, "parentobjid": None, "oktmo": "92000000", "path": "169363"},
            {"objectid": 95235088, "parentobjid": 169363, "oktmo": "92701000", "path": "169363.95235088"},
            {"objectid": 169398, "parentobjid": 95235088, "oktmo": "92701000001", "path": "169363.95235088.169398"},
            {
                "objectid": 172729,
                "parentobjid": 169398,
                "oktmo": "92701000001",
                "path": "169363.95235088.169398.172729",
            },
            {
                "objectid": 71273055,
                "parentobjid": 172729,
                "oktmo": "92701000001",
                "path": "169363.95235088.169398.172729.71273055",
            },
        ]
    )

    out = build_mun_lookup(addr, mun).filter(pl.col("objectid") == 71273055)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["mun_okrug_name"] == "город Казань"
    assert row["mun_okrug_oktmo"] == "92701000"
    assert row["settlement_name"] == "Казань"


def test_picks_rural_raion_for_village_object() -> None:
    """Rural village path: subject → LEVEL=2 раion → LEVEL=4 с.п. →
    LEVEL=6 с (village) → LEVEL=8 ул → leaf. Mun_okrug should be the
    LEVEL=2 раion's name, settlement should be the village (LEVEL=6)."""
    addr = _addr(
        [
            {"objectid": 169363, "objectguid": "g", "name": "Татарстан", "typename": "Респ", "level": 1},
            {"objectid": 183244, "objectguid": "g", "name": "Высокогорский", "typename": "р-н", "level": 2},
            {"objectid": 9001, "objectguid": "g", "name": "Семиозерское", "typename": "с.п.", "level": 4},
            {"objectid": 9002, "objectguid": "g", "name": "Большие Кабаны", "typename": "с", "level": 6},
        ]
    )
    mun = _mun(
        [
            {"objectid": 9999, "parentobjid": 9002, "oktmo": "92215820106", "path": "169363.183244.9001.9002.9999"},
            {"objectid": 9002, "parentobjid": 9001, "oktmo": "92215820106", "path": "169363.183244.9001.9002"},
            {"objectid": 9001, "parentobjid": 183244, "oktmo": "92215820", "path": "169363.183244.9001"},
            {"objectid": 183244, "parentobjid": 169363, "oktmo": "92215000", "path": "169363.183244"},
        ]
    )

    out = build_mun_lookup(addr, mun).filter(pl.col("objectid") == 9999)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["mun_okrug_name"] == "Высокогорский"
    assert row["mun_okrug_oktmo"] == "92215000"
    assert row["settlement_name"] == "Большие Кабаны"


def test_returns_null_when_no_okrug_in_path() -> None:
    """A pathological leaf whose ancestors don't include any LEVEL ∈
    {2, 3} entry — produce nulls rather than crashing."""
    addr = _addr(
        [
            {"objectid": 1, "objectguid": "g", "name": "Татарстан", "typename": "Респ", "level": 1},
        ]
    )
    mun = _mun(
        [
            {"objectid": 99, "parentobjid": 1, "oktmo": "92000000", "path": "1.99"},
        ]
    )

    out = build_mun_lookup(addr, mun)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["mun_okrug_name"] is None
    assert row["mun_okrug_oktmo"] is None
    assert row["settlement_name"] is None


def test_schema_is_typed() -> None:
    addr = _addr([])
    mun = _mun([])
    df = build_mun_lookup(addr, mun)
    assert df.schema == {
        "objectid": pl.Int64,
        "mun_okrug_name": pl.Utf8,
        "mun_okrug_oktmo": pl.Utf8,
        "settlement_name": pl.Utf8,
    }

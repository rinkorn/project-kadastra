"""Tests for build_object_params_lookup.

Pivots parsed AS_HOUSES_PARAMS + AS_STEADS_PARAMS rows for TYPEIDs
{5, 6, 7} into a per-objectid wide table with columns
(oktmo_full, okato, postal_index). HOUSES wins on
(objectid, typeid) collisions.
"""

from __future__ import annotations

import polars as pl

from kadastra.etl.gar_object_params_lookup import build_object_params_lookup


def _params(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "objectid": pl.Int64,
            "typeid": pl.Int16,
            "value": pl.Utf8,
        },
    )


def test_pivots_typeids_to_named_columns() -> None:
    houses = _params(
        [
            {"objectid": 100, "typeid": 7, "value": "92701000001"},
            {"objectid": 100, "typeid": 6, "value": "92401000001"},
            {"objectid": 100, "typeid": 5, "value": "420000"},
        ]
    )
    df = build_object_params_lookup(houses=houses, steads=_params([]))

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["objectid"] == 100
    assert row["oktmo_full"] == "92701000001"
    assert row["okato"] == "92401000001"
    assert row["postal_index"] == "420000"


def test_drops_unwanted_typeids() -> None:
    """The parser may emit TYPEID=8 (CADNUM), 1/2 (ИФНС), etc. — the
    pivot must ignore them, not produce extra columns."""
    houses = _params(
        [
            {"objectid": 100, "typeid": 7, "value": "92701000001"},
            {"objectid": 100, "typeid": 8, "value": "16:50:1:1"},  # cadnum
            {"objectid": 100, "typeid": 1, "value": "1675"},  # ИФНС
        ]
    )
    df = build_object_params_lookup(houses=houses, steads=_params([]))

    assert set(df.columns) == {"objectid", "oktmo_full", "okato", "postal_index"}
    assert df.row(0, named=True)["oktmo_full"] == "92701000001"


def test_houses_wins_on_objectid_typeid_collision() -> None:
    """Same OBJECTID present in both HOUSES and STEADS — HOUSES wins,
    consistent with cadnum_index precedence."""
    houses = _params(
        [{"objectid": 100, "typeid": 7, "value": "house-oktmo"}]
    )
    steads = _params(
        [{"objectid": 100, "typeid": 7, "value": "stead-oktmo"}]
    )
    df = build_object_params_lookup(houses=houses, steads=steads)
    assert df.height == 1
    assert df.row(0, named=True)["oktmo_full"] == "house-oktmo"


def test_partial_typeids_yield_null_columns() -> None:
    """Only TYPEID=7 present → oktmo_full filled, others null but
    columns still exist (downstream join should not have to handle
    missing columns)."""
    houses = _params(
        [{"objectid": 100, "typeid": 7, "value": "92701000001"}]
    )
    df = build_object_params_lookup(houses=houses, steads=_params([]))
    assert set(df.columns) == {"objectid", "oktmo_full", "okato", "postal_index"}
    row = df.row(0, named=True)
    assert row["oktmo_full"] == "92701000001"
    assert row["okato"] is None
    assert row["postal_index"] is None


def test_empty_inputs_return_empty_typed_frame() -> None:
    df = build_object_params_lookup(houses=_params([]), steads=_params([]))
    assert df.height == 0
    assert set(df.columns) == {"objectid", "oktmo_full", "okato", "postal_index"}


def test_typed_schema() -> None:
    houses = _params([{"objectid": 1, "typeid": 7, "value": "x"}])
    df = build_object_params_lookup(houses=houses, steads=_params([]))
    assert df.schema["objectid"] == pl.Int64
    assert df.schema["oktmo_full"] == pl.Utf8
    assert df.schema["okato"] == pl.Utf8
    assert df.schema["postal_index"] == pl.Utf8

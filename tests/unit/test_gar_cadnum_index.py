"""Tests for build_cadnum_index.

Concatenates parsed AS_HOUSES_PARAMS and AS_STEADS_PARAMS DataFrames
(from ``parse_object_params_xml`` with ``typeids={8}``) into a single
``cad_num → objectid`` lookup, tagging each row by source (``"house"``
or ``"stead"``). The same cad_num occasionally appears in both files;
we keep the first match (HOUSES wins, since for an actual building
that's the more specific record)."""

from __future__ import annotations

import polars as pl

from kadastra.etl.gar_cadnum_index import build_cadnum_index


def _params(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "objectid": pl.Int64,
            "typeid": pl.Int16,
            "value": pl.Utf8,
        },
    )


def test_concats_houses_and_steads_with_source_tag() -> None:
    houses = _params(
        [
            {"objectid": 100, "typeid": 8, "value": "16:50:050701:1"},
            {"objectid": 101, "typeid": 8, "value": "16:50:050701:2"},
        ]
    )
    steads = _params(
        [
            {"objectid": 200, "typeid": 8, "value": "16:50:050701:3"},
        ]
    )

    df = build_cadnum_index(houses=houses, steads=steads).sort("cad_num")

    assert df.height == 3
    assert df["cad_num"].to_list() == [
        "16:50:050701:1",
        "16:50:050701:2",
        "16:50:050701:3",
    ]
    assert df["source"].to_list() == ["house", "house", "stead"]


def test_houses_win_on_collision() -> None:
    """If the same cad_num is recorded in both HOUSES_PARAMS and
    STEADS_PARAMS, we keep the HOUSES row — the registry-of-record
    for an actual building."""
    houses = _params([{"objectid": 100, "typeid": 8, "value": "16:50:050701:42"}])
    steads = _params([{"objectid": 999, "typeid": 8, "value": "16:50:050701:42"}])

    df = build_cadnum_index(houses=houses, steads=steads)

    assert df.height == 1
    assert df["objectid"][0] == 100
    assert df["source"][0] == "house"


def test_drops_duplicate_cadnums_within_a_source() -> None:
    """``parse_object_params_xml`` already filters to the active record
    per (objectid, typeid), but the same cad_num can appear under
    different objectids in rare cases (FIAS history quirks). Keep
    the first occurrence to ensure ``cad_num`` is unique."""
    houses = _params(
        [
            {"objectid": 100, "typeid": 8, "value": "16:50:050701:1"},
            {"objectid": 200, "typeid": 8, "value": "16:50:050701:1"},
        ]
    )
    df = build_cadnum_index(houses=houses, steads=_params([]))
    assert df.height == 1
    assert df["objectid"][0] == 100


def test_empty_inputs() -> None:
    df = build_cadnum_index(houses=_params([]), steads=_params([]))
    assert df.height == 0
    assert set(df.columns) == {"cad_num", "objectid", "source"}


def test_typed_schema() -> None:
    houses = _params([{"objectid": 1, "typeid": 8, "value": "x"}])
    df = build_cadnum_index(houses=houses, steads=_params([]))
    assert df.schema == {
        "cad_num": pl.Utf8,
        "objectid": pl.Int64,
        "source": pl.Utf8,
    }


def test_filters_out_non_cadnum_typeids() -> None:
    """When the parser is widened to include other TYPEIDs (5/6/7
    for the parallel object_params lookup), the cadnum builder must
    project only TYPEID=8 rows — otherwise OKTMO/OKAТО strings would
    leak in as cad_nums."""
    houses = _params(
        [
            {"objectid": 100, "typeid": 8, "value": "16:50:050701:1"},
            {"objectid": 100, "typeid": 7, "value": "92701000001"},  # full OKTMO
            {"objectid": 100, "typeid": 5, "value": "420000"},  # postal
        ]
    )
    df = build_cadnum_index(houses=houses, steads=_params([]))
    assert df.height == 1
    assert df["cad_num"][0] == "16:50:050701:1"

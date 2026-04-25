"""Tests for parse_object_params_xml.

Streams ГАР ``AS_HOUSES_PARAMS`` / ``AS_STEADS_PARAMS`` and projects
to a lean DataFrame with one row per (objectid, typeid) pair filtered
by a TYPEID whitelist. Used by ``gar_cadnum_index`` (TYPEID=8) and
later by other lookups (OKTMO via TYPEID=21, etc.).

Sample structure:

    <PARAMS>
      <PARAM ID="..." OBJECTID="..." CHANGEID="..."
             TYPEID="8" VALUE="16:50:050701:1234"
             STARTDATE="..." ENDDATE="..."/>
      ...
    </PARAMS>

We keep only currently active rows: ``ENDDATE`` in the future (FIAS
uses 2079-06-06 as the sentinel for active records).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.etl.gar_xml_object_params import parse_object_params_xml


def _write(tmp: Path, body: str) -> Path:
    out = tmp / "params.xml"
    xml = '<?xml version="1.0" encoding="utf-8"?><PARAMS>' + body + "</PARAMS>"
    out.write_text(xml, encoding="utf-8")
    return out


def _param(**attrs: str) -> str:
    body = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<PARAM {body}/>"


def test_filters_to_whitelisted_typeids(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        _param(
            OBJECTID="100", TYPEID="8", VALUE="16:50:050701:1",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        )
        + _param(
            OBJECTID="100", TYPEID="5", VALUE="420011",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        )
        + _param(
            OBJECTID="200", TYPEID="8", VALUE="16:50:050701:2",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        ),
    )

    df = parse_object_params_xml(xml, typeids={8})

    assert df.height == 2
    assert set(df["objectid"].to_list()) == {100, 200}
    assert set(df["value"].to_list()) == {"16:50:050701:1", "16:50:050701:2"}


def test_filters_out_expired_rows(tmp_path: Path) -> None:
    """FIAS marks active rows with ENDDATE='2079-06-06'. A row with an
    earlier ENDDATE is historical and must be skipped — otherwise a
    single OBJECTID would map to several stale CADNUMs."""
    xml = _write(
        tmp_path,
        _param(
            OBJECTID="100", TYPEID="8", VALUE="OLD",
            STARTDATE="2010-01-01", ENDDATE="2018-01-01",
        )
        + _param(
            OBJECTID="100", TYPEID="8", VALUE="NEW",
            STARTDATE="2018-01-02", ENDDATE="2079-06-06",
        ),
    )

    df = parse_object_params_xml(xml, typeids={8})

    assert df.height == 1
    assert df["value"][0] == "NEW"


def test_keeps_multiple_typeids_per_objectid(tmp_path: Path) -> None:
    """If we whitelist {8, 21}, a single OBJECTID with both rows
    must produce two output rows."""
    xml = _write(
        tmp_path,
        _param(
            OBJECTID="100", TYPEID="8", VALUE="16:50:050701:1",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        )
        + _param(
            OBJECTID="100", TYPEID="21", VALUE="92701000",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        ),
    )

    df = parse_object_params_xml(xml, typeids={8, 21}).sort("typeid")

    assert df.height == 2
    assert df["typeid"].to_list() == [8, 21]
    assert df["value"].to_list() == ["16:50:050701:1", "92701000"]


def test_returns_typed_schema(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        _param(
            OBJECTID="42", TYPEID="8", VALUE="x",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        ),
    )
    df = parse_object_params_xml(xml, typeids={8})
    assert df.schema == {
        "objectid": pl.Int64,
        "typeid": pl.Int16,
        "value": pl.Utf8,
    }


def test_handles_empty_file(tmp_path: Path) -> None:
    xml = _write(tmp_path, "")
    df = parse_object_params_xml(xml, typeids={8})
    assert df.height == 0
    assert set(df.columns) == {"objectid", "typeid", "value"}


def test_handles_missing_required_attrs(tmp_path: Path) -> None:
    """Malformed rows (missing OBJECTID/TYPEID/VALUE) are silently
    skipped — production GAR is large enough that an unhandled
    AttributeError per row is too noisy."""
    xml = _write(
        tmp_path,
        '<PARAM TYPEID="8" VALUE="x" ENDDATE="2079-06-06"/>'
        + '<PARAM OBJECTID="100" VALUE="x" ENDDATE="2079-06-06"/>'
        + _param(
            OBJECTID="100", TYPEID="8", VALUE="GOOD",
            STARTDATE="2020-01-01", ENDDATE="2079-06-06",
        ),
    )
    df = parse_object_params_xml(xml, typeids={8})
    assert df.height == 1
    assert df["value"][0] == "GOOD"

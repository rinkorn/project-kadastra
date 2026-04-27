"""Tests for parse_mun_hierarchy_xml.

Streams ГАР ``AS_MUN_HIERARCHY`` XML into a typed DataFrame, keeping
only currently active rows. Each row carries the full ancestor chain
in ``PATH`` (dot-separated OBJECTID list, root → leaf), so downstream
code can join names/levels from ``AS_ADDR_OBJ`` without an iterative
walk-up.

Sample structure:

    <ITEMS>
      <ITEM ID="..." OBJECTID="96196978" PARENTOBJID="202271"
            OKTMO="92657439106" ISACTIVE="1"
            PATH="169363.95234919.95234932.201791.202271.96196978"
            ENDDATE="2079-06-06" .../>
      ...
    </ITEMS>
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.etl.gar_xml_mun_hierarchy import parse_mun_hierarchy_xml


def _write(tmp: Path, body: str) -> Path:
    out = tmp / "mun.xml"
    xml = '<?xml version="1.0" encoding="utf-8"?><ITEMS>' + body + "</ITEMS>"
    out.write_text(xml, encoding="utf-8")
    return out


def _item(**attrs: str) -> str:
    body = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<ITEM {body}/>"


def test_parses_active_row(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        _item(
            ID="1",
            OBJECTID="96196978",
            PARENTOBJID="202271",
            OKTMO="92657439106",
            ISACTIVE="1",
            ENDDATE="2079-06-06",
            PATH="169363.95234919.95234932.201791.202271.96196978",
        ),
    )

    df = parse_mun_hierarchy_xml(xml)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["objectid"] == 96196978
    assert row["parentobjid"] == 202271
    assert row["oktmo"] == "92657439106"
    assert row["path"] == "169363.95234919.95234932.201791.202271.96196978"


def test_filters_out_inactive_and_expired(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        _item(
            ID="1",
            OBJECTID="100",
            PARENTOBJID="50",
            OKTMO="92...",
            ISACTIVE="0",
            ENDDATE="2079-06-06",
            PATH="50.100",
        )
        + _item(
            ID="2",
            OBJECTID="100",
            PARENTOBJID="50",
            OKTMO="92...",
            ISACTIVE="1",
            ENDDATE="2018-01-01",
            PATH="50.100",
        )
        + _item(
            ID="3",
            OBJECTID="100",
            PARENTOBJID="50",
            OKTMO="92702",
            ISACTIVE="1",
            ENDDATE="2079-06-06",
            PATH="50.100",
        ),
    )
    df = parse_mun_hierarchy_xml(xml)
    assert df.height == 1
    assert df["oktmo"][0] == "92702"


def test_handles_missing_parent_and_oktmo(tmp_path: Path) -> None:
    """Top-of-tree nodes have PARENTOBJID="0" and may have empty OKTMO.
    Downstream code distinguishes via dedicated values, so we keep
    ``parentobjid`` as Int64 with 0 mapped to null and ``oktmo`` as
    Utf8 (empty string preserved as is)."""
    xml = _write(
        tmp_path,
        _item(
            ID="1",
            OBJECTID="169363",
            PARENTOBJID="0",
            OKTMO="",
            ISACTIVE="1",
            ENDDATE="2079-06-06",
            PATH="169363",
        ),
    )

    df = parse_mun_hierarchy_xml(xml)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["objectid"] == 169363
    assert row["parentobjid"] is None
    assert row["oktmo"] == ""


def test_returns_typed_schema(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        _item(
            ID="1",
            OBJECTID="100",
            PARENTOBJID="50",
            OKTMO="92702",
            ISACTIVE="1",
            ENDDATE="2079-06-06",
            PATH="50.100",
        ),
    )
    df = parse_mun_hierarchy_xml(xml)
    assert df.schema == {
        "objectid": pl.Int64,
        "parentobjid": pl.Int64,
        "oktmo": pl.Utf8,
        "path": pl.Utf8,
    }


def test_handles_empty_file(tmp_path: Path) -> None:
    df = parse_mun_hierarchy_xml(_write(tmp_path, ""))
    assert df.height == 0
    assert set(df.columns) == {"objectid", "parentobjid", "oktmo", "path"}

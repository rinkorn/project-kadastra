"""Tests for parse_addr_obj_xml.

Parses ГАР ``AS_ADDR_OBJ`` XML into a DataFrame, keeping only the
currently active record per OBJECTID (``ISACTUAL='1' AND ISACTIVE='1'``).

Sample structure (one tag per row, attributes only — no text/children):

    <ADDRESSOBJECTS>
      <OBJECT ID="1" OBJECTID="100" OBJECTGUID="..."
              NAME="..." TYPENAME="ул." LEVEL="8"
              ISACTUAL="1" ISACTIVE="1" .../>
      ...
    </ADDRESSOBJECTS>
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.etl.gar_xml_addr_obj import parse_addr_obj_xml


def _write_xml(tmp_path: Path, body: str) -> Path:
    out = tmp_path / "addr_obj.xml"
    xml = f'<?xml version="1.0" encoding="utf-8"?><ADDRESSOBJECTS>{body}</ADDRESSOBJECTS>'
    out.write_text(xml, encoding="utf-8")
    return out


def _obj(**attrs: str) -> str:
    body = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<OBJECT {body}/>"


def test_parses_single_active_record(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path,
        _obj(
            ID="1",
            OBJECTID="164979764",
            OBJECTGUID="7c106631-ecea-458e-9123-1b4abfcaf54d",
            NAME="Якова Свердлова",
            TYPENAME="ул.",
            LEVEL="8",
            ISACTUAL="1",
            ISACTIVE="1",
        ),
    )

    df = parse_addr_obj_xml(xml)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["objectid"] == 164979764
    assert row["objectguid"] == "7c106631-ecea-458e-9123-1b4abfcaf54d"
    assert row["name"] == "Якова Свердлова"
    assert row["typename"] == "ул."
    assert row["level"] == 8


def test_filters_out_inactive_and_unactual(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path,
        _obj(
            ID="1",
            OBJECTID="100",
            OBJECTGUID="g1",
            NAME="Старая",
            TYPENAME="ул.",
            LEVEL="8",
            ISACTUAL="0",
            ISACTIVE="0",
        )
        + _obj(
            ID="2",
            OBJECTID="100",
            OBJECTGUID="g1",
            NAME="Новая",
            TYPENAME="ул.",
            LEVEL="8",
            ISACTUAL="1",
            ISACTIVE="1",
        )
        + _obj(
            ID="3",
            OBJECTID="200",
            OBJECTGUID="g2",
            NAME="Текущая, но неактивная",
            TYPENAME="ул.",
            LEVEL="8",
            ISACTUAL="1",
            ISACTIVE="0",
        ),
    )

    df = parse_addr_obj_xml(xml)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["name"] == "Новая"


def test_schema_is_typed(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path,
        _obj(
            ID="1",
            OBJECTID="42",
            OBJECTGUID="g",
            NAME="x",
            TYPENAME="ул.",
            LEVEL="8",
            ISACTUAL="1",
            ISACTIVE="1",
        ),
    )
    df = parse_addr_obj_xml(xml)
    assert df.schema == {
        "objectid": pl.Int64,
        "objectguid": pl.Utf8,
        "name": pl.Utf8,
        "typename": pl.Utf8,
        "level": pl.Int8,
    }


def test_handles_empty_file(tmp_path: Path) -> None:
    xml = _write_xml(tmp_path, "")
    df = parse_addr_obj_xml(xml)
    assert df.height == 0
    assert set(df.columns) == {"objectid", "objectguid", "name", "typename", "level"}


def test_levels_span_admin_to_street(tmp_path: Path) -> None:
    """The hierarchy levels we care about: 1=subject, 2=district,
    3=settlement, 7=quarter/territory, 8=street. Make sure all the
    levels round-trip and end up with consistent int8 dtype."""
    xml = _write_xml(
        tmp_path,
        "".join(
            _obj(
                ID=str(i),
                OBJECTID=str(100 + i),
                OBJECTGUID=f"g{i}",
                NAME=f"o{i}",
                TYPENAME="ул.",
                LEVEL=str(level),
                ISACTUAL="1",
                ISACTIVE="1",
            )
            for i, level in enumerate([1, 2, 3, 7, 8])
        ),
    )
    df = parse_addr_obj_xml(xml).sort("level")
    assert df["level"].to_list() == [1, 2, 3, 7, 8]

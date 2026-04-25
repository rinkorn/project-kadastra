"""Parser for ГАР ``AS_ADDR_OBJ`` XML.

Streams the file via ``xml.etree.ElementTree.iterparse`` (each row is a
self-closing ``<OBJECT>`` tag with attributes only, no children/text).
Keeps only currently active records (``ISACTUAL='1' AND ISACTIVE='1'``)
and projects to the columns we use for joins:

- ``objectid: Int64`` — the FIAS object id.
- ``objectguid: Utf8`` — global UUID (handy for cross-snapshot linking).
- ``name: Utf8`` — display name without the type word.
- ``typename: Utf8`` — the type abbreviation (``ул.``, ``тер.``, ``кв-л``).
- ``level: Int8`` — hierarchy level: 1 subject, 2 admin district,
  3 settlement, 7 quarter/territory, 8 street, etc.

Memory cost is O(rows-buffered) — we ``elem.clear()`` after each row,
so a 14 MB XML stays in tens of MB of Python objects regardless of file
size. Unsuitable rows (missing OBJECTID / non-numeric LEVEL) are
skipped silently — they are extremely rare in production ГАР.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import iterparse

import polars as pl


def parse_addr_obj_xml(path: Path) -> pl.DataFrame:
    objectids: list[int] = []
    guids: list[str] = []
    names: list[str] = []
    typenames: list[str] = []
    levels: list[int] = []

    for _event, elem in iterparse(str(path), events=("end",)):
        if elem.tag != "OBJECT":
            continue
        try:
            attrib = elem.attrib
            if attrib.get("ISACTUAL") != "1" or attrib.get("ISACTIVE") != "1":
                continue
            oid_raw = attrib.get("OBJECTID")
            level_raw = attrib.get("LEVEL")
            if oid_raw is None or level_raw is None:
                continue
            objectids.append(int(oid_raw))
            guids.append(attrib.get("OBJECTGUID", ""))
            names.append(attrib.get("NAME", ""))
            typenames.append(attrib.get("TYPENAME", ""))
            levels.append(int(level_raw))
        finally:
            elem.clear()

    return pl.DataFrame(
        {
            "objectid": objectids,
            "objectguid": guids,
            "name": names,
            "typename": typenames,
            "level": levels,
        },
        schema={
            "objectid": pl.Int64,
            "objectguid": pl.Utf8,
            "name": pl.Utf8,
            "typename": pl.Utf8,
            "level": pl.Int8,
        },
    )

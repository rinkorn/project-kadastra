"""Streaming parser for ГАР ``AS_*_PARAMS`` XML.

Used by ``AS_HOUSES_PARAMS`` (~2.2 GB) and ``AS_STEADS_PARAMS`` (~2.4 GB)
to extract a few key-value pairs per objectid (CADNUM, OKTMO, …) without
loading the full file into memory.

Each row is a self-closing ``<PARAM>`` tag with attributes only; we
project to ``(objectid, typeid, value)`` and keep only:

1. Rows whose ``TYPEID`` is in the whitelist (caller decides which
   parameter types it cares about — TYPEID=8 is CADNUM in this
   snapshot, see ADR-0015 for the full taxonomy).
2. Rows that are currently in force: ``ENDDATE`` is the FIAS sentinel
   ``2079-06-06`` (FIAS uses this date for "no end" — earlier values
   mark historical rows that have been superseded).

We stream via ``xml.etree.ElementTree.iterparse`` and ``elem.clear()``
after each row, so a 2 GB XML stays in tens of MB of Python objects.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from xml.etree.ElementTree import iterparse

import polars as pl

# FIAS sentinel for "currently active". Anything earlier than this is
# a historical row that has been replaced by a newer record.
_ACTIVE_ENDDATE = "2079-06-06"


def parse_object_params_xml(path: Path, *, typeids: Iterable[int]) -> pl.DataFrame:
    wanted = {str(t) for t in typeids}
    objectids: list[int] = []
    typeids_out: list[int] = []
    values: list[str] = []

    for _event, elem in iterparse(str(path), events=("end",)):
        if elem.tag != "PARAM":
            continue
        try:
            attrib = elem.attrib
            tid = attrib.get("TYPEID")
            if tid not in wanted:
                continue
            if attrib.get("ENDDATE") != _ACTIVE_ENDDATE:
                continue
            oid = attrib.get("OBJECTID")
            val = attrib.get("VALUE")
            if oid is None or val is None:
                continue
            objectids.append(int(oid))
            typeids_out.append(int(tid))
            values.append(val)
        finally:
            elem.clear()

    return pl.DataFrame(
        {
            "objectid": objectids,
            "typeid": typeids_out,
            "value": values,
        },
        schema={
            "objectid": pl.Int64,
            "typeid": pl.Int16,
            "value": pl.Utf8,
        },
    )

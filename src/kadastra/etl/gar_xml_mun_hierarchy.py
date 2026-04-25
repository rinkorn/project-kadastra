"""Streaming parser for ГАР ``AS_MUN_HIERARCHY`` XML.

Each row is a single ``<ITEM>`` tag whose attributes describe one node
of the municipal hierarchy. Crucially, ``PATH`` already carries the
full ancestor chain (dot-separated OBJECTID list from subject down
to the node itself), which lets downstream code skip an iterative
walk-up: a single join against ``AS_ADDR_OBJ`` per path segment is
enough to materialize names and levels.

We keep only currently active rows (``ISACTIVE='1'`` and the FIAS
sentinel ``ENDDATE='2079-06-06'``); the file size (~1 GB) makes
loading all historical rows wasteful and downstream code only ever
joins against the current state.

Schema:

- ``objectid: Int64``    — the node's id (unique among active rows).
- ``parentobjid: Int64`` — parent's id; ``None`` for top-level nodes
  (FIAS represents these as ``PARENTOBJID="0"``).
- ``oktmo: Utf8``        — OKTMO at this node; empty string for nodes
  that don't carry one (subject, root).
- ``path: Utf8``         — dot-separated ancestor chain, root → leaf,
  inclusive of ``objectid``.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import iterparse

import polars as pl

_ACTIVE_ENDDATE = "2079-06-06"


def parse_mun_hierarchy_xml(path: Path) -> pl.DataFrame:
    objectids: list[int] = []
    parents: list[int | None] = []
    oktmos: list[str] = []
    paths: list[str] = []

    for _event, elem in iterparse(str(path), events=("end",)):
        if elem.tag != "ITEM":
            continue
        try:
            attrib = elem.attrib
            if attrib.get("ISACTIVE") != "1":
                continue
            if attrib.get("ENDDATE") != _ACTIVE_ENDDATE:
                continue
            oid = attrib.get("OBJECTID")
            if oid is None:
                continue
            objectids.append(int(oid))
            parent_raw = attrib.get("PARENTOBJID", "0")
            try:
                parent_int = int(parent_raw)
            except ValueError:
                parent_int = 0
            parents.append(parent_int if parent_int != 0 else None)
            oktmos.append(attrib.get("OKTMO", ""))
            paths.append(attrib.get("PATH", ""))
        finally:
            elem.clear()

    return pl.DataFrame(
        {
            "objectid": objectids,
            "parentobjid": parents,
            "oktmo": oktmos,
            "path": paths,
        },
        schema={
            "objectid": pl.Int64,
            "parentobjid": pl.Int64,
            "oktmo": pl.Utf8,
            "path": pl.Utf8,
        },
    )

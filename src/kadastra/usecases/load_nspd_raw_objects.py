"""Use case: parse a directory of raw NSPD page-NNNN.json files.

Pipeline per call:

  raw page-*.json  →  read_nspd_{buildings,landplots}_dir
                  →  filter_inside_polygon(region_boundary)
                  →  silver_store.save(region, source, df)

Returns the number of rows written so the caller can log it.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.etl.filter_inside_polygon import filter_inside_polygon
from kadastra.etl.read_nspd_dir import (
    read_nspd_buildings_dir,
    read_nspd_landplots_dir,
)
from kadastra.ports.nspd_silver_store import NspdSilverStorePort
from kadastra.ports.region_boundary import RegionBoundaryPort

_SUPPORTED_SOURCES = ("buildings", "landplots")


class LoadNspdRawObjects:
    def __init__(
        self,
        region_boundary: RegionBoundaryPort,
        silver_store: NspdSilverStorePort,
    ) -> None:
        self._region_boundary = region_boundary
        self._silver_store = silver_store

    def execute(self, *, region_code: str, source: str, raw_dir: Path) -> int:
        raise NotImplementedError

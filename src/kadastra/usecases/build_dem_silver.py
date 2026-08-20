"""Use case: raw DEM tiles → silver topographic layers (ADR-0023).

Stub — implementation follows the failing tests in
``tests/unit/test_build_dem_silver.py``.
"""

from __future__ import annotations

from pathlib import Path


class BuildDemSilver:
    def __init__(self, *, dem_raw_dir: Path, output_base_path: Path, relief_radius_m: float = 500.0) -> None:
        self._dem_raw_dir = dem_raw_dir
        self._output_base_path = output_base_path
        self._relief_radius_m = relief_radius_m

    def execute(self, region_code: str) -> None:
        raise NotImplementedError

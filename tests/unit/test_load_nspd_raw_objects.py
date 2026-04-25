"""Tests for LoadNspdRawObjects.

End-to-end at the use-case level: given a directory of raw NSPD page-NNNN.json
files plus a region polygon, parses, spatially postfilters and writes to the
silver store. Uses a fake silver store and a fake region-boundary port.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from kadastra.usecases.load_nspd_raw_objects import LoadNspdRawObjects

# A polygon roughly around Kazan center for testing.
_KAZAN_BOX = Polygon(
    [
        (49.05, 55.75),
        (49.20, 55.75),
        (49.20, 55.85),
        (49.05, 55.85),
        (49.05, 55.75),
    ]
)


class _FakeRegionBoundary:
    def __init__(self, polygon: BaseGeometry) -> None:
        self._polygon = polygon
        self.calls: list[str] = []

    def get_boundary(self, region_code: str) -> BaseGeometry:
        self.calls.append(region_code)
        return self._polygon


class _FakeNspdSilverStore:
    def __init__(self) -> None:
        self.saves: list[tuple[str, str, pl.DataFrame]] = []

    def save(self, region_code: str, source: str, df: pl.DataFrame) -> None:
        self.saves.append((region_code, source, df))

    def load(self, region_code: str, source: str) -> pl.DataFrame:
        for r, s, df in self.saves:
            if r == region_code and s == source:
                return df
        raise FileNotFoundError(f"{region_code}/{source}")


# 3857 polygon roughly at Kazan center
_KAZAN_3857 = [
    [
        [5468254.6, 7516527.1],
        [5468211.3, 7516474.3],
        [5468206.6, 7516467.9],
        [5468239.5, 7516439.2],
        [5468258.3, 7516460.8],
        [5468254.6, 7516527.1],
    ]
]
# Far away in 3857 (well outside Kazan box)
_MOSCOW_3857 = [
    [
        [4187541.0, 7509270.0],
        [4187600.0, 7509270.0],
        [4187600.0, 7509330.0],
        [4187541.0, 7509330.0],
        [4187541.0, 7509270.0],
    ]
]


def _building_feature(idx: int, *, coords: list, purpose: str = "Жилой дом") -> dict[str, Any]:
    return {
        "id": 100000 + idx,
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": {
            "category": 36369,
            "descr": f"16:50:010406:{idx}",
            "options": {
                "cad_num": f"16:50:010406:{idx}",
                "purpose": purpose,
                "build_record_area": 100.0,
                "cost_value": 1_000_000.0,
                "cost_index": 10_000.0,
            },
        },
    }


def _write_buildings_dir(directory: Path, features: list[dict[str, Any]]) -> None:
    payload = {
        "data": {"type": "FeatureCollection", "features": features},
        "meta": [{"totalCount": len(features), "categoryId": 36369}],
    }
    (directory / "page-0000.json").write_text(json.dumps(payload))


def test_execute_writes_filtered_frame_to_silver(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "buildings-kazan"
    raw_dir.mkdir(parents=True)
    _write_buildings_dir(
        raw_dir,
        [
            _building_feature(1, coords=_KAZAN_3857),  # inside
            _building_feature(2, coords=_MOSCOW_3857),  # outside
        ],
    )

    region_boundary = _FakeRegionBoundary(_KAZAN_BOX)
    store = _FakeNspdSilverStore()
    usecase = LoadNspdRawObjects(region_boundary=region_boundary, silver_store=store)

    n = usecase.execute(
        region_code="RU-KAZAN-AGG", source="buildings", raw_dir=raw_dir
    )

    assert n == 1
    assert region_boundary.calls == ["RU-KAZAN-AGG"]
    assert len(store.saves) == 1
    region_code, source, df = store.saves[0]
    assert region_code == "RU-KAZAN-AGG"
    assert source == "buildings"
    assert df.height == 1
    assert df["geom_data_id"].to_list() == [100001]


def test_execute_landplots_uses_landplot_parser(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "landplots-kazan"
    raw_dir.mkdir(parents=True)
    payload = {
        "data": {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": 200001,
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": _KAZAN_3857},
                    "properties": {
                        "category": 36368,
                        "descr": "16:50:010406:1001",
                        "options": {
                            "cad_num": "16:50:010406:1001",
                            "specified_area": 500.0,
                            "cost_value": 8_000_000.0,
                            "cost_index": 16_000.0,
                            "land_record_category_type": "Земли населенных пунктов",
                        },
                    },
                }
            ],
        },
        "meta": [{"totalCount": 1, "categoryId": 36368}],
    }
    (raw_dir / "page-0000.json").write_text(json.dumps(payload))

    region_boundary = _FakeRegionBoundary(_KAZAN_BOX)
    store = _FakeNspdSilverStore()
    usecase = LoadNspdRawObjects(region_boundary=region_boundary, silver_store=store)

    n = usecase.execute(
        region_code="RU-KAZAN-AGG", source="landplots", raw_dir=raw_dir
    )

    assert n == 1
    df = store.saves[0][2]
    assert df["asset_class"].to_list() == ["landplot"]
    assert "land_record_category_type" in df.columns


def test_execute_unknown_source_raises(tmp_path: Path) -> None:
    region_boundary = _FakeRegionBoundary(_KAZAN_BOX)
    store = _FakeNspdSilverStore()
    usecase = LoadNspdRawObjects(region_boundary=region_boundary, silver_store=store)

    import pytest

    with pytest.raises(ValueError, match="parcels"):
        usecase.execute(
            region_code="RU-KAZAN-AGG", source="parcels", raw_dir=tmp_path
        )

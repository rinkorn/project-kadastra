"""Tests for BuildCellEnrichmentFeatures (ADR-0029, Слой 1 feature_set=enrichment)."""

from __future__ import annotations

from pathlib import Path

import h3
import polars as pl
import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.build_cell_enrichment_features import BuildCellEnrichmentFeatures

_LAT, _LON = 55.79, 49.11
_CELL = h3.latlng_to_cell(_LAT, _LON, 10)
# The isochrone join maps the cell CENTRE (not the anchor point) to res 11.
_CELL_CENTER = h3.cell_to_latlng(_CELL)
_CELL_11 = h3.latlng_to_cell(_CELL_CENTER[0], _CELL_CENTER[1], 11)


class FakeCoverageReader:
    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return pl.DataFrame({"h3_index": [_CELL], "resolution": [10]})


class FakeFeatureStore:
    def __init__(self) -> None:
        self.saved: dict[str, pl.DataFrame] = {}

    def save(self, region_code: str, resolution: int, feature_set: str, df: pl.DataFrame) -> None:
        self.saved[feature_set] = df


class FakeObjectReader:
    def __init__(self, frames: dict[AssetClass, pl.DataFrame]) -> None:
        self._frames = frames

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._frames.get(asset_class, pl.DataFrame())


class FakeDemSampler:
    def sample_elevation(self, *, lat: float, lon: float) -> float:
        return 61.0

    def sample_slope_deg(self, *, lat: float, lon: float) -> float:
        return 1.5

    def sample_relative_relief(self, *, lat: float, lon: float) -> float:
        return 12.0


def _gold_objects() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lat": [_LAT],
            "lon": [_LON],
            "oktmo_full": ["92701000001"],
            "settlement_name": ["Казань"],
        }
    )


def _ways() -> dict[str, list[list[tuple[float, float]]]]:
    # One residential way passing right through the cell centre.
    return {"residential": [[(_LAT - 0.001, _LON), (_LAT + 0.001, _LON)]]}


def _heritage() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "osm_id": ["w1"],
            "ref_egrokn": [None],
            "heritage_level": ["1"],
            "name": ["ОКН"],
            "lat": [_LAT + 0.001],
            "lon": [_LON],
            "polygon_wkt": [None],
        }
    )


def _zouit_zones() -> pl.DataFrame:
    # A square around the cell centre in EPSG:3857.
    from pyproj import Transformer

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = to_3857.transform(
        [_LON - 0.01, _LON + 0.01, _LON + 0.01, _LON - 0.01],
        [_LAT - 0.01, _LAT - 0.01, _LAT + 0.01, _LAT + 0.01],
    )
    from shapely.geometry import Polygon

    wkt = Polygon(zip(xs, ys, strict=True)).wkt
    return pl.DataFrame(
        {
            "zouit_id": ["z1"],
            "type_zone": ["Водоохранная зона"],
            "category": ["water_protection"],
            "geometry_wkt_3857": [wkt],
        }
    )


def _iso_cache() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "h3_index": [_CELL_11],
            "iso15_pop_count": [1234.0],
            "iso15_amenity_count": [7],
            "iso15_metro_reach": [0],
        }
    )


def _macro_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "oktmo": ["92701000"],
            "year": [2024],
            "oktmo_avg_salary_rub": [70_000.0],
        }
    )


@pytest.fixture()
def wired_paths(tmp_path: Path) -> dict[str, Path]:
    heritage_dir = tmp_path / "heritage" / "region=RU-KAZAN-AGG"
    heritage_dir.mkdir(parents=True)
    _heritage().write_parquet(heritage_dir / "data.parquet")
    zouit_dir = tmp_path / "zouit" / "region=RU-KAZAN-AGG"
    zouit_dir.mkdir(parents=True)
    _zouit_zones().write_parquet(zouit_dir / "data.parquet")
    iso_dir = tmp_path / "iso" / "region=RU-KAZAN-AGG" / "h3_p=11"
    iso_dir.mkdir(parents=True)
    _iso_cache().write_parquet(iso_dir / "data.parquet")
    macro_dir = tmp_path / "macro" / "region=RU-KAZAN-AGG" / "year=2024"
    macro_dir.mkdir(parents=True)
    _macro_table().write_parquet(macro_dir / "data.parquet")
    return {
        "heritage": tmp_path / "heritage",
        "zouit": tmp_path / "zouit",
        "iso": tmp_path / "iso",
        "macro": tmp_path / "macro",
    }


def _usecase(store: FakeFeatureStore, paths: dict[str, Path] | None = None) -> BuildCellEnrichmentFeatures:
    p = paths or {}
    return BuildCellEnrichmentFeatures(
        coverage_reader=FakeCoverageReader(),
        feature_store=store,
        object_reader=FakeObjectReader({AssetClass.HOUSE: _gold_objects()}),
        cbd_coords={"RU-KAZAN-AGG": (_LAT, _LON)},
        dem_sampler=FakeDemSampler(),
        ways_by_class=_ways(),
        heritage_silver_path=p.get("heritage"),
        zouit_silver_path=p.get("zouit"),
        isochrone_cache_path=p.get("iso"),
        isochrone_cache_resolution=11,
        macro_oktmo_features_path=p.get("macro"),
        cadastre_target_year=2024,
    )


def test_enrichment_columns_and_values(wired_paths: dict[str, Path]) -> None:
    store = FakeFeatureStore()
    _usecase(store, wired_paths).execute("RU-KAZAN-AGG", 10)
    out = store.saved["enrichment"]
    assert out["h3_index"].to_list() == [_CELL]
    # CBD anchor is at (_LAT, _LON) — within ~70 m of the cell centre.
    assert out["dist_to_cbd_m"][0] == pytest.approx(0.0, abs=100.0)
    assert out["elevation_m"][0] == 61.0
    assert out["slope_deg_local"][0] == 1.5
    assert out["nearest_road_class"][0] == "residential"
    assert out["dist_to_residential_m"][0] == pytest.approx(0.0, abs=100.0)
    assert out["iso15_pop_count"][0] == 1234.0
    assert out["is_heritage_object"][0] == 0  # ОКН ~111 м away > 50 m buffer
    assert out["count_heritage_500m"][0] == 1
    assert out["inside_zouit"][0] == 1
    assert out["inside_water_protection"][0] == 1
    assert out["oktmo_full"][0] == "92701000001"
    assert out["settlement_name"][0] == "Казань"
    assert out["oktmo_avg_salary_rub"][0] == 70_000.0


def test_optional_blocks_skipped_when_not_wired() -> None:
    store = FakeFeatureStore()
    _usecase(store).execute("RU-KAZAN-AGG", 10)
    out = store.saved["enrichment"]
    # Wired blocks present.
    assert "dist_to_cbd_m" in out.columns
    assert "elevation_m" in out.columns
    assert "nearest_road_class" in out.columns
    assert "oktmo_full" in out.columns
    # Unwired blocks absent (the valuation layer fills them with nulls).
    assert "iso15_pop_count" not in out.columns
    assert "inside_zouit" not in out.columns
    assert "oktmo_avg_salary_rub" not in out.columns

"""Tests for ADR-0031 listings-as-target ETL: cleaning + NSPD join.

Синтетические фикстуры: чистка (границы квантилей, sanity-правила, хвост
страниц) и join (матч в радиусе, отсечение по несогласованным атрибутам,
unmatched-корзина с причинами).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.listings_target import (
    REASON_ATTRIBUTE_MISMATCH,
    REASON_NO_COORDS,
    REASON_NO_OBJECT_IN_RADIUS,
    clean_cian_listings,
    match_listings_to_objects,
)
from kadastra.usecases.build_listings_target import BuildListingsTarget

_LISTING_COLUMNS = {
    "listing_id": pl.Utf8,
    "source": pl.Utf8,
    "city": pl.Utf8,
    "price_rub": pl.Float64,
    "total_area_m2": pl.Float64,
    "floor": pl.Float64,
    "floors_count": pl.Float64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "page_file": pl.Utf8,
    "price_per_sqm_rub": pl.Float64,
}

# Казань: 0.0001° широты ≈ 11.1 м, 0.0001° долготы ≈ 6.3 м.
_BASE_LAT = 55.79
_BASE_LON = 49.12


def _listings(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_LISTING_COLUMNS)


def _listing(listing_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "listing_id": listing_id,
        "source": "cian",
        "city": "Казань",
        "price_rub": 5_000_000.0,
        "total_area_m2": 50.0,
        "floor": 3.0,
        "floors_count": 10.0,
        "lat": _BASE_LAT,
        "lon": _BASE_LON,
        "page_file": "page-010.html",
        "price_per_sqm_rub": 100_000.0,
    }
    base.update(overrides)
    return base


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "levels": pl.Int64,
        "area_m2": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _object(object_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "object_id": object_id,
        "lat": _BASE_LAT,
        "lon": _BASE_LON,
        "levels": 10,
        "area_m2": 5_000.0,
    }
    base.update(overrides)
    return base


# --- чистка: квантильная фильтрация ---------------------------------------


def test_quantile_filter_drops_tails_and_adds_ask_column() -> None:
    # 100 «нормальных» строк 1000..100_000 + 2 выброса в хвостах.
    rows = [_listing(f"ok-{i}", price_per_sqm_rub=float(1000 * (i + 1))) for i in range(100)]
    rows.append(_listing("low-outlier", price_per_sqm_rub=10.0))
    rows.append(_listing("high-outlier", price_per_sqm_rub=10_000_000.0))

    result = clean_cian_listings(_listings(rows))

    assert result.frame.height == 100
    assert "low-outlier" not in result.frame["listing_id"].to_list()
    assert "high-outlier" not in result.frame["listing_id"].to_list()
    # Границы вычислены из данных, а не захардкожены.
    assert result.price_per_m2_lower_bound == pytest.approx(2000.0)
    assert result.price_per_m2_upper_bound == pytest.approx(100_000.0)
    # Контракт ADR-0031: ask_rub_per_m2 = очищенный price_per_sqm_rub.
    assert result.frame["ask_rub_per_m2"].to_list() == result.frame["price_per_sqm_rub"].to_list()


def test_quantile_filter_keeps_rows_at_bounds() -> None:
    rows = [_listing(f"ok-{i}", price_per_sqm_rub=float(1000 * (i + 1))) for i in range(100)]
    result = clean_cian_listings(_listings(rows))
    # Без выбросов p1/p99 попадают на крайние значения — ничего не режется.
    assert result.frame.height == 100


# --- чистка: sanity-правила -----------------------------------------------


def test_area_sanity_bounds() -> None:
    rows = [
        _listing("too-small", total_area_m2=9.9),
        _listing("min-ok", total_area_m2=10.0),
        _listing("max-ok", total_area_m2=300.0),
        _listing("too-big", total_area_m2=300.1),
    ]
    result = clean_cian_listings(_listings(rows))
    assert sorted(result.frame["listing_id"].to_list()) == ["max-ok", "min-ok"]


def test_floor_not_above_floors_count() -> None:
    rows = [
        _listing("ok", floor=10.0, floors_count=10.0),
        _listing("broken", floor=11.0, floors_count=10.0),
        _listing("no-floors-count", floor=5.0, floors_count=None),
    ]
    result = clean_cian_listings(_listings(rows))
    assert sorted(result.frame["listing_id"].to_list()) == ["no-floors-count", "ok"]


def test_page_tail_cut() -> None:
    rows = [
        _listing("last-ok", page_file="page-054.html"),
        _listing("tail", page_file="page-055.html"),
        _listing("far-tail", page_file="page-183.html"),
        _listing("unparseable-page", page_file="index"),
    ]
    result = clean_cian_listings(_listings(rows))
    assert sorted(result.frame["listing_id"].to_list()) == ["last-ok", "unparseable-page"]


def test_page_threshold_is_parameter() -> None:
    rows = [_listing("p10", page_file="page-010.html"), _listing("p11", page_file="page-011.html")]
    result = clean_cian_listings(_listings(rows), max_page=10)
    assert result.frame["listing_id"].to_list() == ["p10"]


# --- join к НСПД -----------------------------------------------------------


def test_match_within_radius() -> None:
    listings = clean_cian_listings(
        _listings([_listing("l1", lat=_BASE_LAT + 0.0001)])  # ~11 м от объекта
    ).frame
    objects = _objects([_object("obj-1")])

    matched, unmatched = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched.height == 1
    assert unmatched.height == 0
    assert matched["matched_object_id"].to_list() == ["obj-1"]
    assert matched["match_distance_m"][0] == pytest.approx(11.1, abs=1.0)


def test_no_object_within_radius_goes_to_unmatched() -> None:
    listings = clean_cian_listings(_listings([_listing("l1", lat=_BASE_LAT + 0.01)])).frame  # ~1.1 км
    objects = _objects([_object("obj-1")])

    matched, unmatched = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched.height == 0
    assert unmatched["unmatched_reason"].to_list() == [REASON_NO_OBJECT_IN_RADIUS]


def test_floor_above_building_levels_is_rejected() -> None:
    listings = clean_cian_listings(_listings([_listing("l1", floor=12.0, floors_count=12.0)])).frame
    objects = _objects([_object("obj-1", levels=10)])

    matched, unmatched = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched.height == 0
    assert unmatched["unmatched_reason"].to_list() == [REASON_ATTRIBUTE_MISMATCH]


def test_flat_bigger_than_building_is_rejected() -> None:
    listings = clean_cian_listings(_listings([_listing("l1", total_area_m2=250.0)])).frame
    objects = _objects([_object("obj-1", area_m2=200.0)])

    matched, unmatched = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched.height == 0
    assert unmatched["unmatched_reason"].to_list() == [REASON_ATTRIBUTE_MISMATCH]


def test_missing_building_attributes_skip_consistency() -> None:
    listings = clean_cian_listings(_listings([_listing("l1")])).frame
    objects = _objects([_object("obj-1", levels=None, area_m2=None)])

    matched, unmatched = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched.height == 1
    assert unmatched.height == 0


def test_listing_without_coords_is_unmatched() -> None:
    listings = clean_cian_listings(_listings([_listing("l1", lat=None, lon=None)])).frame
    objects = _objects([_object("obj-1")])

    matched, unmatched = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched.height == 0
    assert unmatched["unmatched_reason"].to_list() == [REASON_NO_COORDS]


def test_nearest_object_wins() -> None:
    listings = clean_cian_listings(_listings([_listing("l1")])).frame
    objects = _objects(
        [
            _object("far", lat=_BASE_LAT + 0.0008),  # ~89 м
            _object("near", lat=_BASE_LAT + 0.0001),  # ~11 м
        ]
    )

    matched, _ = match_listings_to_objects(listings, objects, radius_m=100.0)

    assert matched["matched_object_id"].to_list() == ["near"]


# --- use case end-to-end ---------------------------------------------------


def test_usecase_writes_matched_and_unmatched_partitions(tmp_path: Path) -> None:
    listings_path = tmp_path / "all.parquet"
    rows = [
        _listing("match-me"),
        _listing("far-away", lat=_BASE_LAT + 0.01),
        _listing("other-source", source="avito"),
        _listing("other-city", city="Иркутск"),
    ]
    _listings(rows).write_parquet(listings_path)

    objects_dir = tmp_path / "gold" / "region=R" / "asset_class=apartment"
    objects_dir.mkdir(parents=True)
    _objects([_object("obj-1")]).write_parquet(objects_dir / "data.parquet")

    usecase = BuildListingsTarget(
        listings_path=listings_path,
        valuation_objects_path=tmp_path / "gold",
        output_base_path=tmp_path / "out",
    )

    stats = usecase.execute("R", AssetClass.APARTMENT)

    assert stats.n_input == 2  # только cian/Казань
    assert stats.n_clean == 2
    assert stats.n_matched == 1
    assert stats.n_unmatched == 1
    assert stats.unmatched_reasons == {REASON_NO_OBJECT_IN_RADIUS: 1}

    partition = tmp_path / "out" / "region=R" / "asset_class=apartment"
    matched = pl.read_parquet(partition / "matched.parquet")
    unmatched = pl.read_parquet(partition / "unmatched.parquet")
    assert matched["listing_id"].to_list() == ["match-me"]
    assert {"matched_object_id", "match_distance_m", "ask_rub_per_m2"} <= set(matched.columns)
    assert unmatched["listing_id"].to_list() == ["far-away"]
    assert "unmatched_reason" in unmatched.columns

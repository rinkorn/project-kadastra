"""ADR-0031 (вариант б): CIAN-листинги как market-target для apartment.

Чистка MVP-выгрузки листингов и join к объектам НСПД:

- квантильная фильтрация ``price_per_sqm_rub`` (границы p1/p99
  вычисляются из самих данных, не хардкодятся);
- sanity-правила: ``10 ≤ total_area_m2 ≤ 300``, ``floor ≤ floors_count``
  (где оба значения есть);
- отрезание хвоста выдачи ``page > max_page`` (риск отравления выдачи
  после детекта бота, ADR-0031 «Риск», митигация 3);
- join к НСПД как жёсткий мусор-фильтр: ближайший объект в радиусе
  ``radius_m`` (haversine по центроидам) + согласованность атрибутов
  (``floor ≤ levels`` здания, ``total_area_m2 ≤ area_m2`` здания — квартира
  не может быть больше дома). Несматченные листинги — отдельная корзина
  с причиной (измерение масштаба возможного отравления).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Причины попадания листинга в unmatched-корзину.
REASON_NO_COORDS = "no_coords"
REASON_NO_OBJECT_IN_RADIUS = "no_object_within_radius"
REASON_ATTRIBUTE_MISMATCH = "attribute_mismatch"


@dataclass(frozen=True)
class CleaningResult:
    """Результат чистки: очищенный frame + фактические границы квантилей."""

    frame: pl.DataFrame
    price_per_m2_lower_bound: float
    price_per_m2_upper_bound: float


def clean_cian_listings(
    listings: pl.DataFrame,
    *,
    price_quantile_low: float = 0.01,
    price_quantile_high: float = 0.99,
    min_area_m2: float = 10.0,
    max_area_m2: float = 300.0,
    max_page: int = 54,
) -> CleaningResult:
    raise NotImplementedError


def match_listings_to_objects(
    listings: pl.DataFrame,
    objects: pl.DataFrame,
    *,
    radius_m: float = 100.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Join листингов к объектам НСПД. Возвращает (matched, unmatched).

    matched: колонки листинга + ``matched_object_id`` + ``match_distance_m``.
    unmatched: колонки листинга + ``unmatched_reason``.
    """
    raise NotImplementedError

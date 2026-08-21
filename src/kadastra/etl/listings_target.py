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

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from kadastra.etl.haversine import EARTH_RADIUS_METERS

# Причины попадания листинга в unmatched-корзину.
REASON_NO_COORDS = "no_coords"
REASON_NO_OBJECT_IN_RADIUS = "no_object_within_radius"
REASON_ATTRIBUTE_MISMATCH = "attribute_mismatch"

_PAGE_NUMBER_PATTERN = r"(\d+)"


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
    """Чистка CIAN-листингов перед использованием как target.

    Границы квантилей считаются по входному frame (linear-интерполяция,
    границы инклюзивны). Добавляет контрактную колонку ``ask_rub_per_m2``.
    """
    priced = listings.filter(pl.col("price_per_sqm_rub").is_not_null())
    lower = priced["price_per_sqm_rub"].quantile(price_quantile_low, interpolation="linear")
    upper = priced["price_per_sqm_rub"].quantile(price_quantile_high, interpolation="linear")
    if lower is None or upper is None:  # пустой после фильтра null
        return CleaningResult(
            frame=priced.clear().with_columns(pl.lit(None, dtype=pl.Float64).alias("ask_rub_per_m2")),
            price_per_m2_lower_bound=math.nan,
            price_per_m2_upper_bound=math.nan,
        )

    page_number = pl.col("page_file").str.extract(_PAGE_NUMBER_PATTERN, 1).cast(pl.Int64, strict=False)
    cleaned = (
        priced.with_columns(page_number.alias("_page_number"))
        .filter(
            pl.col("price_per_sqm_rub").is_between(lower, upper)
            & pl.col("total_area_m2").is_between(min_area_m2, max_area_m2)
            & (
                pl.col("floor").is_null()
                | pl.col("floors_count").is_null()
                | (pl.col("floor") <= pl.col("floors_count"))
            )
            # Хвост выдачи отбрасываем; страницы без номера не относим к хвосту.
            & (pl.col("_page_number").is_null() | (pl.col("_page_number") <= max_page))
        )
        .drop("_page_number")
        .with_columns(pl.col("price_per_sqm_rub").alias("ask_rub_per_m2"))
    )
    return CleaningResult(
        frame=cleaned,
        price_per_m2_lower_bound=float(lower),
        price_per_m2_upper_bound=float(upper),
    )


def match_listings_to_objects(
    listings: pl.DataFrame,
    objects: pl.DataFrame,
    *,
    radius_m: float = 100.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Join листингов к объектам НСПД. Возвращает (matched, unmatched).

    matched: колонки листинга + ``matched_object_id`` + ``match_distance_m``.
    unmatched: колонки листинга + ``unmatched_reason``.

    Матч — ближайший по haversine объект в радиусе ``radius_m``, прошедший
    согласованность атрибутов. Если ближайший не проходит проверку, листинг
    уходит в unmatched (второй ближайший не пробуем — жёсткий фильтр).
    """
    candidates = objects.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    cand_ids = candidates["object_id"].to_list()
    cand_lats = np.radians(candidates["lat"].to_numpy())
    cand_lons = np.radians(candidates["lon"].to_numpy())
    cand_levels = candidates["levels"].to_numpy()
    cand_areas = candidates["area_m2"].to_numpy()

    matched_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    for row in listings.iter_rows(named=True):
        lat = row["lat"]
        lon = row["lon"]
        if lat is None or lon is None or candidates.is_empty():
            unmatched_rows.append({**row, "unmatched_reason": REASON_NO_COORDS})
            continue

        rlat, rlon = math.radians(float(lat)), math.radians(float(lon))
        dlat = cand_lats - rlat
        dlon = cand_lons - rlon
        a = np.sin(dlat / 2) ** 2 + math.cos(rlat) * np.cos(cand_lats) * np.sin(dlon / 2) ** 2
        distances = 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
        nearest = int(distances.argmin())
        distance = float(distances[nearest])

        if distance > radius_m:
            unmatched_rows.append({**row, "unmatched_reason": REASON_NO_OBJECT_IN_RADIUS})
            continue
        if not _attributes_consistent(row, cand_levels[nearest], cand_areas[nearest]):
            unmatched_rows.append({**row, "unmatched_reason": REASON_ATTRIBUTE_MISMATCH})
            continue
        matched_rows.append({**row, "matched_object_id": cand_ids[nearest], "match_distance_m": distance})

    matched = _rows_to_frame(
        matched_rows, listings.schema, {"matched_object_id": pl.Utf8, "match_distance_m": pl.Float64}
    )
    unmatched = _rows_to_frame(unmatched_rows, listings.schema, {"unmatched_reason": pl.Utf8})
    return matched, unmatched


def _attributes_consistent(row: dict[str, object], levels: object, area_m2: object) -> bool:
    floor = row["floor"]
    if floor is not None and levels is not None and float(floor) > float(levels):  # type: ignore[arg-type]
        return False
    area = row["total_area_m2"]
    return not (area is not None and area_m2 is not None and float(area) > float(area_m2))  # type: ignore[arg-type]


def _rows_to_frame(
    rows: list[dict[str, object]],
    base_schema: pl.Schema,
    extra_schema: dict[str, pl.DataType | type[pl.DataType]],
) -> pl.DataFrame:
    schema = pl.Schema({**base_schema, **extra_schema})
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).cast(schema, strict=False).select(schema.names())

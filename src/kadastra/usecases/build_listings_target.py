"""Use case: CIAN-листинги → market-target для apartment (ADR-0031, вариант б).

Читает унифицированный long-дамп листингов (``data/silver/listings-mvp/all.parquet``),
фильтрует ``source/city``, чистит (квантили ₽/м², sanity-правила, хвост страниц)
и джойнит к объектам НСПД нужного класса. Пишет две партиции:

``{output_base}/region={region}/asset_class={ac}/matched.parquet``
``{output_base}/region={region}/asset_class={ac}/unmatched.parquet``

Переобучение модели на этом target — отдельный этап, здесь только ETL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kadastra.domain.asset_class import AssetClass


@dataclass(frozen=True)
class ListingsTargetStats:
    """Счётчики прогона — печатаются скриптом и идут в измерения ADR-0031."""

    n_input: int
    n_clean: int
    price_per_m2_lower_bound: float
    price_per_m2_upper_bound: float
    n_matched: int
    n_unmatched: int
    unmatched_reasons: dict[str, int]


class BuildListingsTarget:
    def __init__(
        self,
        *,
        listings_path: Path,
        valuation_objects_path: Path,
        output_base_path: Path,
        match_radius_m: float = 100.0,
        max_page: int = 54,
    ) -> None:
        self._listings_path = listings_path
        self._valuation_objects_path = valuation_objects_path
        self._output_base_path = output_base_path
        self._match_radius_m = match_radius_m
        self._max_page = max_page

    def execute(
        self,
        region_code: str,
        asset_class: AssetClass,
        *,
        source: str = "cian",
        city: str = "Казань",
    ) -> ListingsTargetStats:
        raise NotImplementedError

"""Read EMISS / Росстат #61781 silver and surface a market reference.

#61781 = «Средняя цена 1 кв.м. квартир по центрам субъектов РФ»
(quarterly, primary + secondary, all apartment qualities). Used as
an independent market anchor next to the quartet's OOF predictions
in the inspector, so the human can compare «модель воспроизводит
ЕГРН-кадастр» vs «реальная цена рынка» — typically a 30 % gap
because ЕГРН is broken (ADR-0010).

Apartment-only — the indicator does not cover house / commercial /
landplot. Non-apartment classes return ``None``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# RU-region-code → EMISS region_okato. EMISS uses the OKATO of the
# subject (region) and ``mestdom_code='2'`` to encode «center city of
# that subject». For Татарстан the center city is Казань.
_REGION_TO_OKATO: dict[str, str] = {
    "RU-KAZAN-AGG": "92000000000",
    "RU-TA": "92000000000",  # used by integration test fixtures
}

_INDICATOR_DIR = "61781"
_MESTDOM_CENTER_CITY = "2"
_TIPKVARTIR_ALL = "1"
_RYNZHEL_PRIMARY = "1"
_RYNZHEL_SECONDARY = "3"


class GetMarketReference:
    def __init__(self, silver_base_path: Path) -> None:
        self._base_path = silver_base_path

    def execute(
        self,
        *,
        region_code: str,
        asset_class: str,
        year: int,
    ) -> dict[str, object] | None:
        if asset_class != "apartment":
            return None
        okato = _REGION_TO_OKATO.get(region_code)
        if okato is None:
            return None

        path = self._base_path / _INDICATOR_DIR / "data.parquet"
        if not path.is_file():
            return None

        df = pl.read_parquet(path).filter(
            (pl.col("region_okato") == okato)
            & (pl.col("mestdom_code") == _MESTDOM_CENTER_CITY)
            & (pl.col("tipkvartir_code") == _TIPKVARTIR_ALL)
            & (pl.col("year") == year)
        )
        if df.is_empty():
            return None

        secondary = df.filter(pl.col("rynzhel_code") == _RYNZHEL_SECONDARY)["value_rub_per_m2"].mean()
        primary = df.filter(pl.col("rynzhel_code") == _RYNZHEL_PRIMARY)["value_rub_per_m2"].mean()

        return {
            "source": f"EMISS-{_INDICATOR_DIR}",
            "region_okato": okato,
            "asset_class": asset_class,
            "year": year,
            "primary_rub_per_m2": (float(primary) if isinstance(primary, (int, float)) else None),
            "secondary_rub_per_m2": (float(secondary) if isinstance(secondary, (int, float)) else None),
        }

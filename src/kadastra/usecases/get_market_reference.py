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
        raise NotImplementedError

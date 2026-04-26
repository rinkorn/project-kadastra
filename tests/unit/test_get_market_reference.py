"""Tests for GetMarketReference.

Reads ЕМИСС/Росстат #61781 silver parquet and returns the most-recent
year's average ₽/м² for the given region's center-subject city,
broken down by primary / secondary market. Apartments only.

Used by /api/market_reference for the quartet inspector panel —
shows the EMISS market anchor next to the 4 quartet OOF predictions
so the human sees the gap between «модель воспроизводит ЕГРН» and
«реальный рынок Казани».
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kadastra.usecases.get_market_reference import GetMarketReference


def _emiss_row(
    *,
    region_okato: str,
    year: int,
    quarter: int,
    rynzhel_code: str,
    tipkvartir_code: str,
    value: float,
    mestdom_code: str = "2",
) -> dict[str, object]:
    return {
        "indicator_id": "61781",
        "region_okato": region_okato,
        "region_name": "Республика Татарстан (Татарстан)",
        "mestdom_code": mestdom_code,
        "mestdom_name": "Центр субъекта Российской Федерации",
        "unit_code": "rub_per_m2",
        "unit_name": "руб/м²",
        "period_code": f"q{quarter}",
        "period_name": f"Q{quarter}",
        "period_quarter": quarter,
        "rynzhel_code": rynzhel_code,
        "rynzhel_name": (
            "Первичный рынок жилья" if rynzhel_code == "1" else "Вторичный рынок жилья"
        ),
        "tipkvartir_code": tipkvartir_code,
        "tipkvartir_name": "Все типы квартир",
        "year": year,
        "period_label": f"{year}-Q{quarter}",
        "value_rub_per_m2": value,
    }


def _seed_emiss(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    base = tmp_path / "61781"
    base.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "indicator_id": pl.Utf8,
            "region_okato": pl.Utf8,
            "region_name": pl.Utf8,
            "mestdom_code": pl.Utf8,
            "mestdom_name": pl.Utf8,
            "unit_code": pl.Utf8,
            "unit_name": pl.Utf8,
            "period_code": pl.Utf8,
            "period_name": pl.Utf8,
            "period_quarter": pl.Int64,
            "rynzhel_code": pl.Utf8,
            "rynzhel_name": pl.Utf8,
            "tipkvartir_code": pl.Utf8,
            "tipkvartir_name": pl.Utf8,
            "year": pl.Int64,
            "period_label": pl.Utf8,
            "value_rub_per_m2": pl.Float64,
        },
    ).write_parquet(base / "data.parquet")
    return tmp_path


# --------------------------------------------------------------------------
# Happy path: avg over 4 quarters of 2025, both markets, Kazan
# --------------------------------------------------------------------------


def test_returns_avg_over_year_for_both_markets(tmp_path: Path) -> None:
    """Tatarstan center city = Казань. EMISS region_okato = '92000000000'.
    Result is the mean of 4 quarters, separately for primary vs secondary."""
    rows = []
    # Secondary 2025: 154, 155, 156, 157 → avg 155.5
    for q, val in zip([1, 2, 3, 4], [154_000, 155_000, 156_000, 157_000], strict=True):
        rows.append(_emiss_row(
            region_okato="92000000000", year=2025, quarter=q,
            rynzhel_code="3", tipkvartir_code="1", value=val,
        ))
    # Primary 2025: 234, 236, 238, 240 → avg 237
    for q, val in zip([1, 2, 3, 4], [234_000, 236_000, 238_000, 240_000], strict=True):
        rows.append(_emiss_row(
            region_okato="92000000000", year=2025, quarter=q,
            rynzhel_code="1", tipkvartir_code="1", value=val,
        ))
    base = _seed_emiss(tmp_path, rows)

    out = GetMarketReference(base).execute(
        region_code="RU-KAZAN-AGG", asset_class="apartment", year=2025,
    )
    assert out is not None
    assert out["source"] == "EMISS-61781"
    assert out["region_okato"] == "92000000000"
    assert out["year"] == 2025
    assert out["asset_class"] == "apartment"
    assert out["secondary_rub_per_m2"] == pytest.approx(155_500.0)
    assert out["primary_rub_per_m2"] == pytest.approx(237_000.0)


# --------------------------------------------------------------------------
# Year filter: only year-of-interest rows averaged
# --------------------------------------------------------------------------


def test_year_filter_excludes_other_years(tmp_path: Path) -> None:
    rows = [
        _emiss_row(region_okato="92000000000", year=2025, quarter=1,
                   rynzhel_code="3", tipkvartir_code="1", value=200_000),
        _emiss_row(region_okato="92000000000", year=2024, quarter=1,
                   rynzhel_code="3", tipkvartir_code="1", value=999_999),
    ]
    base = _seed_emiss(tmp_path, rows)
    out = GetMarketReference(base).execute(
        region_code="RU-KAZAN-AGG", asset_class="apartment", year=2025,
    )
    assert out is not None
    # Only 2025 row → mean is 200_000, not contaminated by 2024.
    assert out["secondary_rub_per_m2"] == 200_000.0


# --------------------------------------------------------------------------
# Apartment-only contract — non-apartment asset_class returns None
# --------------------------------------------------------------------------


def test_returns_none_for_non_apartment_classes(tmp_path: Path) -> None:
    """EMISS #61781 is an apartment-market indicator only. House,
    commercial, landplot are not covered — return None so the API
    surfaces null and UI renders «—»."""
    rows = [
        _emiss_row(region_okato="92000000000", year=2025, quarter=1,
                   rynzhel_code="3", tipkvartir_code="1", value=156_000),
    ]
    base = _seed_emiss(tmp_path, rows)
    usecase = GetMarketReference(base)
    for cls in ("house", "commercial", "landplot", "all"):
        assert usecase.execute(region_code="RU-KAZAN-AGG", asset_class=cls, year=2025) is None


# --------------------------------------------------------------------------
# Unknown region → None (rather than KeyError) — UI just hides the row
# --------------------------------------------------------------------------


def test_returns_none_for_unknown_region(tmp_path: Path) -> None:
    rows = [
        _emiss_row(region_okato="92000000000", year=2025, quarter=1,
                   rynzhel_code="3", tipkvartir_code="1", value=156_000),
    ]
    base = _seed_emiss(tmp_path, rows)
    out = GetMarketReference(base).execute(
        region_code="RU-IRKUTSK", asset_class="apartment", year=2025,
    )
    assert out is None


# --------------------------------------------------------------------------
# No data for that year → None
# --------------------------------------------------------------------------


def test_returns_none_when_year_has_no_data(tmp_path: Path) -> None:
    rows = [
        _emiss_row(region_okato="92000000000", year=2024, quarter=1,
                   rynzhel_code="3", tipkvartir_code="1", value=130_000),
    ]
    base = _seed_emiss(tmp_path, rows)
    out = GetMarketReference(base).execute(
        region_code="RU-KAZAN-AGG", asset_class="apartment", year=2025,
    )
    assert out is None


# --------------------------------------------------------------------------
# Tipkvartir filter — only «Все типы» (code='1') goes into average
# --------------------------------------------------------------------------


def test_filters_to_all_quartiles_tipkvartir(tmp_path: Path) -> None:
    """Five tipkvartir codes exist (all/low/mid/improved/elite). We
    only average over the rolled-up «Все типы» (code 1) — otherwise
    the elite/economy variants would skew the average."""
    rows = [
        # «Все типы» = code '1' → must be picked.
        _emiss_row(region_okato="92000000000", year=2025, quarter=1,
                   rynzhel_code="3", tipkvartir_code="1", value=156_000),
        # «Элитные» = code '5' → must be ignored.
        _emiss_row(region_okato="92000000000", year=2025, quarter=1,
                   rynzhel_code="3", tipkvartir_code="5", value=999_999),
    ]
    base = _seed_emiss(tmp_path, rows)
    out = GetMarketReference(base).execute(
        region_code="RU-KAZAN-AGG", asset_class="apartment", year=2025,
    )
    assert out is not None
    assert out["secondary_rub_per_m2"] == 156_000.0


# --------------------------------------------------------------------------
# Missing parquet → None (rather than crash)
# --------------------------------------------------------------------------


def test_returns_none_when_emiss_parquet_missing(tmp_path: Path) -> None:
    """If EMISS data hasn't been ingested yet, the use case returns
    None — the UI/API treats this as «no reference available» and
    hides the EMISS row in the quartet panel."""
    out = GetMarketReference(tmp_path).execute(
        region_code="RU-KAZAN-AGG", asset_class="apartment", year=2025,
    )
    assert out is None

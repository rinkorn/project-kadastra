"""Собираем wide-таблицу macro_oktmo_features из silver ЕМИСС (ADR-0022).

Читает long-таблицы ``data/silver/emiss/{id}/data.parquet`` (generic
schema: territory_code/territory_name/year/value, см.
scripts/parse_emiss_xls_generic.py), нарезает по индикаторам в
производные фичи и пишет wide-таблицу:

    data/silver/macro_oktmo_features/region={code}/year={Y}/data.parquet
      oktmo (Utf8, 8 знаков), year (Int64),
      oktmo_avg_salary_rub, oktmo_population, oktmo_population_density,
      oktmo_housing_volume_5y_m2, oktmo_unemployment_pct,
      oktmo_retail_turnover_per_capita

Одна строка на (oktmo, year) где есть хоть одна фича. Year alignment
(«последний доступный год ≤ target_year») делает НЕ этот скрипт, а
compute_object_macro_features при джойне — здесь сохраняем все годы.

Производные:
- oktmo_housing_volume_5y_m2 — скользящая сумма 34466 за 5 лет
  (Y-4..Y) по каждому ОКТМО;
- oktmo_retail_turnover_per_capita — 40464 / население того же года;
- oktmo_population_density — население / площадь. Площади муниципалитетов
  в GAR нет — источник площади опционален (--area-parquet с колонками
  oktmo, area_km2); без него фича null (задокументировано в ADR-0022).

Запуск:
    uv run python scripts/build_macro_oktmo_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from kadastra.config import Settings

# indicator_id → выходная фича (прямое переименование значения).
DIRECT_INDICATORS = {
    "57792": "oktmo_avg_salary_rub",
    "44164": "oktmo_unemployment_pct",
}
POPULATION_INDICATOR = "31557"
HOUSING_INDICATOR = "34466"
RETAIL_INDICATOR = "40464"

FEATURE_COLUMNS = [
    "oktmo_avg_salary_rub",
    "oktmo_population",
    "oktmo_population_density",
    "oktmo_housing_volume_5y_m2",
    "oktmo_unemployment_pct",
    "oktmo_retail_turnover_per_capita",
]


def load_indicator(base: Path, indicator_id: str) -> pl.DataFrame | None:
    path = base / indicator_id / "data.parquet"
    if not path.is_file():
        print(f"   ! {indicator_id}: {path} missing — skipped", flush=True)
        return None
    return pl.read_parquet(path)


def oktmo8(df: pl.DataFrame) -> pl.DataFrame:
    """Нормализация территориального кода к 8-значному ОКТМО."""
    return df.with_columns(pl.col("territory_code").str.slice(0, 8).alias("oktmo"))


def direct_feature(df: pl.DataFrame, feature: str, oktmo_prefix: str) -> pl.DataFrame:
    return (
        oktmo8(df)
        .filter(pl.col("oktmo").str.starts_with(oktmo_prefix))
        .group_by(["oktmo", "year"])
        .agg(pl.col("value").mean().alias(feature))
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oktmo-prefix", default="92", help="префикс ОКТМО субъекта (92 = Татарстан)")
    ap.add_argument("--area-parquet", type=Path, default=None, help="опционально: oktmo, area_km2")
    args = ap.parse_args()

    settings = Settings()
    base = settings.emiss_silver_base_path
    out_base = settings.macro_oktmo_features_path / f"region={settings.region_code}"
    prefix = args.oktmo_prefix

    frames: dict[str, pl.DataFrame] = {}

    for indicator_id, feature in DIRECT_INDICATORS.items():
        df = load_indicator(base, indicator_id)
        if df is not None:
            frames[feature] = direct_feature(df, feature, prefix)

    pop_df = load_indicator(base, POPULATION_INDICATOR)
    population = None
    if pop_df is not None:
        population = direct_feature(pop_df, "oktmo_population", prefix)
        frames["oktmo_population"] = population

    housing_df = load_indicator(base, HOUSING_INDICATOR)
    if housing_df is not None:
        yearly = direct_feature(housing_df, "_housing", prefix)
        housing_5y = (
            yearly.sort(["oktmo", "year"])
            .with_columns(
                pl.col("_housing")
                .rolling_sum(window_size=5, min_samples=1)
                .over("oktmo")
                .alias("oktmo_housing_volume_5y_m2")
            )
            .select(["oktmo", "year", "oktmo_housing_volume_5y_m2"])
        )
        frames["oktmo_housing_volume_5y_m2"] = housing_5y

    retail_df = load_indicator(base, RETAIL_INDICATOR)
    if retail_df is not None and population is not None:
        retail = direct_feature(retail_df, "_retail", prefix)
        per_capita = (
            retail.join(population, on=["oktmo", "year"], how="left")
            .with_columns((pl.col("_retail") / pl.col("oktmo_population")).alias("oktmo_retail_turnover_per_capita"))
            .select(["oktmo", "year", "oktmo_retail_turnover_per_capita"])
        )
        frames["oktmo_retail_turnover_per_capita"] = per_capita
    elif retail_df is not None:
        print("   ! retail without population — per_capita skipped", flush=True)

    if args.area_parquet is not None and population is not None and args.area_parquet.is_file():
        area = pl.read_parquet(args.area_parquet)
        density = (
            population.join(area, on="oktmo", how="left")
            .with_columns((pl.col("oktmo_population") / pl.col("area_km2")).alias("oktmo_population_density"))
            .select(["oktmo", "year", "oktmo_population_density"])
        )
        frames["oktmo_population_density"] = density

    if not frames:
        sys.exit("no indicators parsed — nothing to build")

    # Full outer join всех фич по (oktmo, year).
    wide: pl.DataFrame | None = None
    for df in frames.values():
        wide = df if wide is None else wide.join(df, on=["oktmo", "year"], how="full", coalesce=True)
    assert wide is not None
    for col in FEATURE_COLUMNS:
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    wide = wide.select(["oktmo", "year", *FEATURE_COLUMNS]).sort(["oktmo", "year"])

    years = wide["year"].unique().sort().to_list()
    print(f"=> wide rows: {wide.height:,}  oktmo: {wide['oktmo'].n_unique()}  years: {years}", flush=True)
    for y in years:
        part = out_base / f"year={y}"
        part.mkdir(parents=True, exist_ok=True)
        slice_df = wide.filter(pl.col("year") == y)
        slice_df.write_parquet(part / "data.parquet")
    print(f"=> wrote {out_base}/year=*/data.parquet", flush=True)

    # Coverage summary для ADR.
    print("\n=> feature coverage (non-null share over all rows):")
    for col in FEATURE_COLUMNS:
        share = wide.select(pl.col(col).is_not_null().mean()).item()
        print(f"   {col}: {share:.1%}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Собираем wide-таблицу macro_oktmo_features из silver (ADR-0022).

Источники (generic long schema: territory_code/year/value):
- ``bdmo_8112027`` — численность населения на 1 января (БД ПМО, ОКТМО,
  2009-2025);
- ``bdmo_8213002`` — среднемесячная зарплата работников организаций
  ГО/МР (БД ПМО, ОКТМО, 2008-2024);
- ``bdmo_8010001`` — ввод в действие жилых домов, м² (БД ПМО, ОКТМО,
  2006-2022);
- ``bdmo_8401003`` — оборот розничной торговли, тыс. руб (БД ПМО,
  ОКТМО, 2017-2024);
- ``bdmo_8006001`` — общая площадь земель МО, га (БД ПМО, ОКТМО,
  2006-2023) — для плотности населения;
- ``43062`` — уровень безработицы по методологии МОТ, % (ЕМИСС,
  субъект РФ, 2000-2026) — константа по Татарстану, broadcast на все
  ОКТМО.

Производные:
- ``oktmo_housing_volume_5y_m2`` — скользящая сумма ввода жилья за 5 лет
  (Y-4..Y) по каждому ОКТМО;
- ``oktmo_population_density`` — население / (площадь_га / 100);
- ``oktmo_retail_turnover_per_capita`` — оборот*1000 / население
  того же года, руб/чел.

Пишет wide-таблицу:

    data/silver/macro_oktmo_features/region={code}/year={Y}/data.parquet
      oktmo (Utf8, 8 знаков), year (Int64), 6 feature-колонок

Одна строка на (oktmo, year) где есть хоть одна фича. Year alignment
(«последний доступный год ≤ target_year») делает НЕ этот скрипт, а
compute_object_macro_features при джойне — здесь сохраняем все годы.

Запуск:
    uv run python scripts/build_macro_oktmo_features.py
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from kadastra.config import Settings

POPULATION_INDICATOR = "bdmo_8112027"
SALARY_INDICATOR = "bdmo_8213002"
HOUSING_INDICATOR = "bdmo_8010001"
RETAIL_INDICATOR = "bdmo_8401003"
AREA_INDICATOR = "bdmo_8006001"
UNEMPLOYMENT_INDICATOR = "43062"

FEATURE_COLUMNS = [
    "oktmo_avg_salary_rub",
    "oktmo_population",
    "oktmo_population_density",
    "oktmo_housing_volume_5y_m2",
    "oktmo_unemployment_pct",
    "oktmo_retail_turnover_per_capita",
]


def load_indicator(base, indicator_id: str) -> pl.DataFrame | None:
    path = base / indicator_id / "data.parquet"
    if not path.is_file():
        print(f"   ! {indicator_id}: {path} missing — skipped", flush=True)
        return None
    return pl.read_parquet(path)


def yearly_feature(df: pl.DataFrame, feature: str, oktmo_prefix: str) -> pl.DataFrame:
    """Long → (oktmo, year, feature), только ОКТМО субъекта."""
    return (
        df.filter(pl.col("territory_code").str.starts_with(oktmo_prefix))
        .group_by(["territory_code", "year"])
        .agg(pl.col("value").mean().alias(feature))
        .rename({"territory_code": "oktmo"})
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oktmo-prefix", default="92", help="префикс ОКТМО субъекта (92 = Татарстан)")
    args = ap.parse_args()

    settings = Settings()
    base = settings.emiss_silver_base_path
    out_base = settings.macro_oktmo_features_path / f"region={settings.region_code}"
    prefix: str = args.oktmo_prefix

    frames: dict[str, pl.DataFrame] = {}

    salary_df = load_indicator(base, SALARY_INDICATOR)
    if salary_df is not None:
        frames["oktmo_avg_salary_rub"] = yearly_feature(salary_df, "oktmo_avg_salary_rub", prefix)

    population = None
    pop_df = load_indicator(base, POPULATION_INDICATOR)
    if pop_df is not None:
        population = yearly_feature(pop_df, "oktmo_population", prefix)
        frames["oktmo_population"] = population

    housing_df = load_indicator(base, HOUSING_INDICATOR)
    if housing_df is not None:
        yearly = yearly_feature(housing_df, "_housing", prefix)
        frames["oktmo_housing_volume_5y_m2"] = (
            yearly.sort(["oktmo", "year"])
            .with_columns(
                pl.col("_housing")
                .rolling_sum(window_size=5, min_samples=1)
                .over("oktmo")
                .alias("oktmo_housing_volume_5y_m2")
            )
            .select(["oktmo", "year", "oktmo_housing_volume_5y_m2"])
        )

    area_df = load_indicator(base, AREA_INDICATOR)
    if area_df is not None and population is not None:
        area = yearly_feature(area_df, "_area_ha", prefix)
        frames["oktmo_population_density"] = (
            population.join(area, on=["oktmo", "year"], how="left")
            .with_columns((pl.col("oktmo_population") / (pl.col("_area_ha") / 100.0)).alias("oktmo_population_density"))
            .select(["oktmo", "year", "oktmo_population_density"])
        )
    elif area_df is None:
        print("   ! area indicator missing — population_density skipped", flush=True)

    retail_df = load_indicator(base, RETAIL_INDICATOR)
    if retail_df is not None and population is not None:
        retail = yearly_feature(retail_df, "_retail_krub", prefix)
        frames["oktmo_retail_turnover_per_capita"] = (
            retail.join(population, on=["oktmo", "year"], how="left")
            .with_columns(
                (pl.col("_retail_krub") * 1000.0 / pl.col("oktmo_population")).alias("oktmo_retail_turnover_per_capita")
            )
            .select(["oktmo", "year", "oktmo_retail_turnover_per_capita"])
        )
    elif retail_df is not None:
        print("   ! retail without population — per_capita skipped", flush=True)

    if not frames:
        sys.exit("no indicators parsed — nothing to build")

    # Full outer join всех фич по (oktmo, year).
    wide: pl.DataFrame | None = None
    for df in frames.values():
        wide = df if wide is None else wide.join(df, on=["oktmo", "year"], how="full", coalesce=True)
    assert wide is not None

    # Безработица — субъектная константа: broadcast по year на все ОКТМО.
    unemp_df = load_indicator(base, UNEMPLOYMENT_INDICATOR)
    if unemp_df is not None:
        unemp = unemp_df.group_by("year").agg(pl.col("value").mean().alias("oktmo_unemployment_pct"))
        wide = wide.join(unemp, on="year", how="left")

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

"""Парсим CSV БД ПМО (tochno.st зеркало Росстата) в generic long parquet.

Источник: https://tochno.st/datasets/bdmo (CC-BY), снапшот
``data_bdmo_118_v20250918``. Per-indicator zip содержит один CSV
``data_Y{код}_112_v20250918.csv`` (UTF-8, разделитель ``;``) с колонками:
region_id, mun_level, municipality, oktmo, mun_type, ..., year,
indicator_value + опциональные extra-измерения (mest / okved2 / mosh).

Фильтры по умолчанию (под задачу ADR-0022):
- ``oktmo`` начинается с ``--oktmo-prefix`` (92 = Татарстан);
- ``mun_level == 'Муниципальное образование верхнего уровня'`` —
  районы/округа, джойнятся с GAR ``mun_okrug_oktmo`` (сверено: 45/45);
- агрегатные строки вида «Городские округа Республики ...» (ОКТМО
  92700000 и аналоги) отбрасываем;
- per-indicator фильтры extra-измерений (см. ``INDICATOR_FILTERS``):
  население — только «Все население»; розница — только итоговый ОКВЭД2
  «Торговля розничная, кроме ...» за «Январь-декабрь»; ввод жилья —
  «Жилые здания» (всего), без разбивки по ИЖС.

Выход: ``data/silver/emiss/bdmo_{code}/data.parquet`` в той же generic
схеме, что и scripts/parse_emiss_xls_generic.py:
indicator_id, territory_dim, territory_code, territory_name, extra_dims,
year, value.

Запуск:
    uv run python scripts/parse_bdmo_csv_to_parquet.py --code 8112027
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw/bdmo")
SILVER_BASE = Path("data/silver/emiss")

UPPER_LEVEL = "Муниципальное образование верхнего уровня"

# code → фильтры по extra-измерениям (колонка → значение).
INDICATOR_FILTERS: dict[str, dict[str, str]] = {
    "8112027": {  # Оценка численности населения на 1 января
        "mest": "Все население",
        "indicator_period": "На 1 января",
    },
    "8213002": {},  # Среднемесячная зарплата работников орг-й ГО/МР
    "8401003": {  # Оборот розничной торговли
        "okved2": "Торговля розничная, кроме торговли автотранспортными средствами и мотоциклами",
        "indicator_period": "Январь-декабрь",
    },
    "8010001": {  # Введено в действие жилых домов
        "mosh": "Жилые здания",
        "indicator_period": "Значение показателя за год",
    },
    "8006001": {},  # Общая площадь земель МО
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="код показателя БД ПМО, напр. 8112027")
    ap.add_argument("--oktmo-prefix", default="92", help="префикс ОКТМО субъекта (92 = Татарстан)")
    ap.add_argument("--src", type=Path, default=None, help="путь к CSV (default: data/raw/bdmo/data_Y{code}_...)")
    ap.add_argument("--dst", type=Path, default=None, help="default: data/silver/emiss/bdmo_{code}/data.parquet")
    args = ap.parse_args()

    code = args.code
    src = args.src or RAW_DIR / f"data_Y4{code}_112_v20250918.csv"
    dst = args.dst or SILVER_BASE / f"bdmo_{code}" / "data.parquet"
    if not src.is_file():
        sys.exit(f"src not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(
        src,
        separator=";",
        schema_overrides={
            "oktmo": pl.Utf8,
            "region_id": pl.Utf8,
            "oktmo_stable": pl.Utf8,
            "oktmo_history": pl.Utf8,
            "oktmo_year_from": pl.Utf8,
            "oktmo_year_to": pl.Utf8,
        },
        null_values=["CD"],
        infer_schema_length=10000,
    )
    print(f"=> raw rows: {df.height:,}  cols: {df.columns}", flush=True)

    out = df.filter(
        pl.col("oktmo").str.starts_with(args.oktmo_prefix)
        & (pl.col("mun_level") == UPPER_LEVEL)
        & ~pl.col("municipality").str.contains(
            r"^(Городские округа |Муниципальные районы |Городские поселения |Сельские поселения )"
        )
    )

    filters = INDICATOR_FILTERS.get(code, {})
    for col, val in filters.items():
        if col in out.columns:
            before = out.height
            out = out.filter(pl.col(col) == val)
            print(f"   filter {col}={val!r}: {before:,} → {out.height:,}", flush=True)
        else:
            print(f"   ! filter column {col} missing — skipped", flush=True)

    # Страховочный дедуп по (oktmo, year) — редкие двойные строки
    # (напр. salary 92655000/2009 с двумя period-метками).
    out = (
        out.group_by(["oktmo", "municipality", "year"])
        .agg(pl.col("indicator_value").mean().alias("value"))
        .rename({"oktmo": "territory_code", "municipality": "territory_name"})
        .with_columns(
            pl.lit(int(code)).alias("indicator_id"),
            pl.lit("oktmo").alias("territory_dim"),
            pl.lit(json.dumps(filters, ensure_ascii=False)).alias("extra_dims"),
        )
        .select(["indicator_id", "territory_dim", "territory_code", "territory_name", "extra_dims", "year", "value"])
        .sort(["territory_code", "year"])
    )

    out.write_parquet(dst)
    years = out["year"]
    print(
        f"=> wrote {dst}  rows={out.height:,}  oktmo={out['territory_code'].n_unique()}"
        f"  years={years.min()}..{years.max()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

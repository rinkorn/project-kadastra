"""Парсим выгрузку ЕМИСС/fedstat в формате .xls (pivot wide → long parquet).

Поддерживаются индикаторы #61781 и #31452 (одинаковая структура «Данные»):
- Левые колонки: измерения как «<код> <название>»
- Правые колонки: годы; ячейка = значение в ₽/м² за этот год+квартал.

Парсер не привязан к конкретному ID — определяет состав dimension-колонок
по заголовку таблицы. Период (квартал) уже сидит как отдельный ряд внутри
pivot, поэтому one row in xls = один (индикатор, регион, ..., квартал, год)
кортеж после анпивота.

Запуск:
    uv run --with xlrd python scripts/parse_emiss_xls_to_parquet.py \\
        --indicator-id 31452 \\
        --src data/raw/emiss/31452/raw_2026-04-26.xls \\
        --dst data/silver/emiss/31452/data.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl
import xlrd

CODE_RE = re.compile(r"^\s*(\S+)\s+(.+?)\s*$")
ROMAN_TO_QUARTER = {"I": 1, "II": 2, "III": 3, "IV": 4}


def split_code(cell: str) -> tuple[str, str]:
    """`'        14000000000 Белгородская область'` → ('14000000000', 'Белгородская область')."""
    if cell is None:
        return ("", "")
    s = str(cell).strip()
    if not s:
        return ("", "")
    m = CODE_RE.match(s)
    if not m:
        return ("", s)
    return (m.group(1), m.group(2))


def quarter_from(period_name: str) -> int | None:
    if not period_name:
        return None
    head = period_name.strip().split()[0].upper()
    return ROMAN_TO_QUARTER.get(head)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicator-id", required=True, type=int)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"src not found: {args.src}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    wb = xlrd.open_workbook(str(args.src), formatting_info=False)
    sheet_name = "Данные" if "Данные" in wb.sheet_names() else wb.sheet_names()[0]
    sh = wb.sheet_by_name(sheet_name)

    # row 0 — title, row 1 — empty, row 2 — header. Data starts at row 3.
    header = sh.row_values(2)
    # dim cols: cells before first 4-digit year
    year_re = re.compile(r"^\s*(19|20)\d{2}\s*$")
    dim_cols: list[str] = []
    year_cols: list[tuple[int, int]] = []  # (col_index, year)
    for ci, h in enumerate(header):
        s = str(h).strip()
        m = year_re.match(s)
        if m:
            year_cols.append((ci, int(s)))
        else:
            dim_cols.append(s)
    print(
        f"=> sheet={sheet_name!r}  rows={sh.nrows}  cols={sh.ncols}\n"
        f"   dim cols: {dim_cols}\n"
        f"   year cols: {[y for _, y in year_cols]}",
        flush=True,
    )

    # We expect at least these dimension columns; mestdom is optional.
    # Map header label → key the parser uses.
    header_to_key = {
        "Классификатор объектов административно-территориального деления (ОКАТО)": "okato",
        "Место расположения жилья": "mestdom",
        "Единица измерения": "unit",
        "Период": "period",
        "Рынок жилья": "rynzhel",
        "Типы квартир": "tipkvartir",
    }
    dim_to_key: dict[int, str] = {}
    for ci, h in enumerate(header[: len(dim_cols)]):
        s = str(h).strip()
        key = header_to_key.get(s)
        if key is None:
            # also try fuzzy
            for k_label, k_key in header_to_key.items():
                if s.startswith(k_label[:30]):
                    key = k_key
                    break
        if key is None:
            sys.exit(f"unknown dim header: {s!r}")
        dim_to_key[ci] = key
    print(f"   dim mapping: {dim_to_key}", flush=True)

    has_mestdom = "mestdom" in dim_to_key.values()

    rows_out: list[dict] = []
    for ri in range(3, sh.nrows):
        row = sh.row_values(ri)
        dims_parsed: dict[str, tuple[str, str]] = {}
        for ci, key in dim_to_key.items():
            dims_parsed[key] = split_code(row[ci])
        okato_code, okato_name = dims_parsed["okato"]
        rynzhel_code, rynzhel_name = dims_parsed.get("rynzhel", ("", ""))
        tip_code, tip_name = dims_parsed.get("tipkvartir", ("", ""))
        unit_code, unit_name = dims_parsed.get("unit", ("", ""))
        period_code, period_name = dims_parsed.get("period", ("", ""))
        period_quarter = quarter_from(period_name)
        if has_mestdom:
            mestdom_code, mestdom_name = dims_parsed.get("mestdom", ("", ""))
        else:
            mestdom_code, mestdom_name = "", ""

        for ci, year in year_cols:
            cell = row[ci]
            if cell == "" or cell is None:
                continue
            try:
                value = float(str(cell).replace(",", "."))
            except (ValueError, TypeError):
                continue
            rows_out.append(
                {
                    "indicator_id": args.indicator_id,
                    "region_okato": okato_code,
                    "region_name": okato_name,
                    "mestdom_code": mestdom_code,
                    "mestdom_name": mestdom_name,
                    "unit_code": unit_code,
                    "unit_name": unit_name,
                    "period_code": period_code,
                    "period_name": period_name,
                    "period_quarter": period_quarter,
                    "rynzhel_code": rynzhel_code,
                    "rynzhel_name": rynzhel_name,
                    "tipkvartir_code": tip_code,
                    "tipkvartir_name": tip_name,
                    "year": year,
                    "period_label": (f"{year}-Q{period_quarter}" if period_quarter else str(year)),
                    "value_rub_per_m2": value,
                }
            )

    print(f"=> long rows: {len(rows_out):,}", flush=True)
    if not rows_out:
        sys.exit("no rows parsed")

    df = pl.DataFrame(
        rows_out,
        schema={
            "indicator_id": pl.Int64,
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
    )
    df.write_parquet(args.dst)
    print(f"=> wrote {args.dst}  shape={df.shape}", flush=True)

    # mini-summary
    print("\n=> coverage summary:")
    print(
        f"   regions: {df['region_okato'].n_unique()}\n"
        f"   years: {df['year'].min()}..{df['year'].max()}\n"
        f"   quarters: {sorted(df['period_quarter'].unique().drop_nulls().to_list())}\n"
        f"   rynzhel: {df['rynzhel_name'].unique().to_list()}\n"
        f"   tipkvartir: {df['tipkvartir_name'].unique().to_list()}\n"
        f"   mestdom: {df['mestdom_name'].unique().to_list()}"
    )
    print("\n=> Иркутская обл / Татарстан / Москва — последние записи:")
    cities_okato = ["25000000000", "92000000000", "45000000000"]  # Иркутск, Татарстан, Москва
    sample = df.filter(pl.col("region_okato").is_in(cities_okato))
    if not sample.is_empty():
        print(
            sample.group_by(["region_name", "year"])
            .agg(pl.col("value_rub_per_m2").median().round(0).alias("median_per_sqm"))
            .sort(["region_name", "year"])
            .tail(20)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Парсим выгрузку ЕМИСС/fedstat (excel pivot) в generic long parquet.

В отличие от scripts/parse_emiss_xls_to_parquet.py (заточен под pivot
#61781/#31452 с фиксированным набором измерений), этот парсер не знает
семантику измерений заранее — для годовых макро-индикаторов ADR-0022
(57792 зарплата, население, 34466 ввод жилья, 44164 безработица,
40464 розничный оборот):

- Первая не-gодовая колонка — территория; ячейка вида
  ``<код> <название>`` (ОКАТО для субъектного разреза, ОКТМО для
  муниципального) → territory_code / territory_name. Заголовок колонки
  сохраняем в territory_dim (из него видно, ОКАТО это или ОКТМО).
- Остальные измерения (единица измерения, период, виды показателя...)
  складываем в extra_dims JSON: ``{header: {"code": ..., "name": ...}}``.
- Годовые колонки анпивотим в (year, value).

Запуск:
    uv run --with xlrd python scripts/parse_emiss_xls_generic.py \\
        --indicator-id 57792 \\
        --src data/raw/emiss/57792/raw_2026-08-20.xls \\
        --dst data/silver/emiss/57792/data.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl
import xlrd

CODE_RE = re.compile(r"^\s*(\S+)\s+(.+?)\s*$")
YEAR_RE = re.compile(r"^\s*(19|20)\d{2}\s*$")


def split_code(cell: object) -> tuple[str, str]:
    """`'        92601000 Агрызский муниципальный район'` → ('92601000', 'Агрызский муниципальный район')."""
    if cell is None:
        return ("", "")
    s = str(cell).strip()
    if not s:
        return ("", "")
    m = CODE_RE.match(s)
    if not m:
        return ("", s)
    return (m.group(1), m.group(2))


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
    header = [str(h).strip() for h in sh.row_values(2)]
    dim_cols: list[tuple[int, str]] = []  # (col_index, header)
    year_cols: list[tuple[int, int]] = []  # (col_index, year)
    for ci, h in enumerate(header):
        m = YEAR_RE.match(h)
        if m:
            year_cols.append((ci, int(h)))
        else:
            dim_cols.append((ci, h))
    if not dim_cols or not year_cols:
        sys.exit(f"unexpected layout: header={header}")
    print(
        f"=> sheet={sheet_name!r}  rows={sh.nrows}  cols={sh.ncols}\n"
        f"   dim cols: {[h for _, h in dim_cols]}\n"
        f"   year cols: {[y for _, y in year_cols]}",
        flush=True,
    )

    territory_ci, territory_dim = dim_cols[0]
    other_dims = dim_cols[1:]

    rows_out: list[dict] = []
    for ri in range(3, sh.nrows):
        row = sh.row_values(ri)
        territory_code, territory_name = split_code(row[territory_ci])
        if not territory_name:
            continue
        extra: dict[str, dict[str, str]] = {}
        for ci, h in other_dims:
            code, name = split_code(row[ci])
            if name:
                extra[h] = {"code": code, "name": name}
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
                    "territory_dim": territory_dim,
                    "territory_code": territory_code,
                    "territory_name": territory_name,
                    "extra_dims": json.dumps(extra, ensure_ascii=False),
                    "year": year,
                    "value": value,
                }
            )

    print(f"=> long rows: {len(rows_out):,}", flush=True)
    if not rows_out:
        sys.exit("no rows parsed")

    df = pl.DataFrame(
        rows_out,
        schema={
            "indicator_id": pl.Int64,
            "territory_dim": pl.Utf8,
            "territory_code": pl.Utf8,
            "territory_name": pl.Utf8,
            "extra_dims": pl.Utf8,
            "year": pl.Int64,
            "value": pl.Float64,
        },
    )
    df.write_parquet(args.dst)
    print(f"=> wrote {args.dst}  shape={df.shape}", flush=True)

    print("\n=> coverage summary:")
    print(
        f"   territories: {df['territory_code'].n_unique()}\n"
        f"   years: {df['year'].min()}..{df['year'].max()}\n"
        f"   territory_dim: {df['territory_dim'].unique().to_list()}"
    )
    print("\n=> sample territories:")
    print(df.select(["territory_code", "territory_name"]).unique().head(15))
    return 0


if __name__ == "__main__":
    sys.exit(main())

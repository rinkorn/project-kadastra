"""Парсим субъектный индикатор ЕМИСС (downloadData.do xls) в long parquet.

Layout выгрузки fedstat downloadData.do отличается от pivot-формата
scripts/parse_emiss_xls_to_parquet.py: заголовки dim-колонок пустые,
годы в row 2 (col 4+), row 3 — подзаголовок периода, данные с row 4,
территории БЕЗ числовых кодов (только названия с отступами).

Для ADR-0022 нужен субъектный уровень безработицы (43062) — константа
по Татарстану. Парсер вытаскивает строку одного субъекта
(``--territory-name``, точное совпадение после strip) и анпивотит годы.

Выход — та же generic схема (territory_dim='subject',
territory_code=--territory-code, напр. '16' для Татарстана).

Запуск:
    uv run --with xlrd python scripts/parse_emiss_subject_xls.py \
        --indicator-id 43062 --territory-name "Республика Татарстан (Татарстан)" \
        --territory-code 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl
import xlrd

YEAR_RE = re.compile(r"^\s*(19|20)\d{2}\s*$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicator-id", required=True, type=int)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--territory-name", default="Республика Татарстан (Татарстан)")
    ap.add_argument("--territory-code", default="16")
    args = ap.parse_args()

    src = args.src or Path(f"data/raw/emiss/{args.indicator_id}/raw_2026-08-20.xls")
    dst = args.dst or Path(f"data/silver/emiss/{args.indicator_id}/data.parquet")
    if not src.is_file():
        sys.exit(f"src not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    wb = xlrd.open_workbook(str(src))
    sh = wb.sheet_by_name("Данные") if "Данные" in wb.sheet_names() else wb.sheet_by_index(0)

    header = [str(h).strip() for h in sh.row_values(2)]
    year_cols = [(ci, int(h)) for ci, h in enumerate(header) if YEAR_RE.match(h)]
    if not year_cols:
        sys.exit(f"no year columns in row 2: {header[:10]}")

    rows_out: list[dict] = []
    matched: list[str] = []
    for ri in range(4, sh.nrows):
        row = sh.row_values(ri)
        # Территория — вторая колонка (col 1), со вложенными отступами.
        name = str(row[1]).strip() if len(row) > 1 else ""
        if name != args.territory_name:
            continue
        matched.append(name)
        unit = str(row[2]).strip() if len(row) > 2 else ""
        for ci, year in year_cols:
            cell = row[ci]
            if cell in ("", None):
                continue
            try:
                value = float(str(cell).replace(",", "."))
            except (ValueError, TypeError):
                continue
            rows_out.append(
                {
                    "indicator_id": args.indicator_id,
                    "territory_dim": "subject",
                    "territory_code": args.territory_code,
                    "territory_name": name,
                    "extra_dims": json.dumps({"unit": unit}, ensure_ascii=False),
                    "year": year,
                    "value": value,
                }
            )

    if not rows_out:
        sys.exit(f"territory {args.territory_name!r} not found or all values empty")
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
    ).unique(subset=["territory_code", "year"], keep="first")
    df.write_parquet(dst)
    print(
        f"=> wrote {dst}  rows={df.height}  years={df['year'].min()}..{df['year'].max()}"
        f"  matched rows in xls: {len(matched)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

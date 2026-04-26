"""Сравнительный анализ трёх источников листингов (Yandex Realty, CIAN,
Avito) на mvp-выборке Иркутск+Казань.

Цель — понять, **с кем стоит заключать договор о доступе к данным** для
production-кадастровой оценки. Считаем:

1. **Volume** — кол-во уникальных и сырых объявлений на источник × город.
2. **Schema richness** — кол-во полей и какие из них «полезные» для модели.
3. **Field completeness** — % non-null по ключевым model-фичам
   (price, area, floor, rooms, build_year, materialType, lat/lon).
4. **Price reasonableness** — медиана + p10/p25/p75/p90 ₽ и ₽/м²;
   доля «битых» строк (price/area вне здравых рамок).
5. **Anchor delta** — отклонение медианы ₽/м² источника от ЕМИСС #61781
   2025-Q4 (вторичка) для центра субъекта. Источник, ближе к якорю —
   ближе к Росстат-валидной картине рынка.
6. **Pagination dedup ratio** — какая доля карточек дублируется между
   страницами выдачи (продвигаемые объявления). Высокий dedup → плохая
   масштабируемость через пагинацию.

На выходе:
* `data/silver/listings-paginated/_compare.csv` — табличка для пользователя.
* stdout — markdown-сводка с рекомендацией.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

SILVER = Path("data/silver/listings-paginated")
EMISS = Path("data/silver/emiss/61781/data.parquet")
OUT_CSV = SILVER / "_compare.csv"
OUT_MD = SILVER / "_compare.md"

# city -> (region_name_substring, market_type) для якоря ЕМИСС #61781 2025-Q4
ANCHORS_QUERY = {
    "Иркутск": ("Иркутск", "Вторичный рынок жилья"),
    "Казань": ("Татарстан", "Вторичный рынок жилья"),
}

KEY_FIELDS_PER_SOURCE = {
    "yandex_realty": ["price_rub", "total_area_m2", "floor", "floors_count",
                      "rooms_str", "url"],
    "cian": ["price_rub", "total_area_m2", "floor", "floors_count",
             "rooms_count", "build_year", "material_type", "lat", "lon", "url"],
    "avito": ["price_rub", "total_area_m2", "floor", "floors_count",
              "rooms_str", "url"],
}


def load_emiss_anchors() -> dict[str, float]:
    df = pl.read_parquet(EMISS)
    out: dict[str, float] = {}
    for city, (region_kw, market) in ANCHORS_QUERY.items():
        row = (
            df.filter(
                pl.col("region_name").str.contains(region_kw)
                & (pl.col("tipkvartir_name") == "Все типы квартир")
                & (pl.col("rynzhel_name") == market)
                & (pl.col("year") == 2025)
                & (pl.col("period_quarter") == 4)
            )
            .select("value_rub_per_m2")
            .item()
        )
        out[city] = float(row)
    return out


def stats_for_source_city(source: str, city_ru: str) -> dict:
    fp = SILVER / f"{source}_{'irkutsk' if city_ru == 'Иркутск' else 'kazan'}.parquet"
    if not fp.exists():
        return {"source": source, "city": city_ru, "rows_unique": 0}
    df = pl.read_parquet(fp)
    n_unique = df.shape[0]

    # raw count = из per-page-counts если есть в _summary.json
    summary = json.loads((SILVER / "_summary.json").read_text())
    raw_match = next(
        (s for s in summary
         if s["source"] == source and s["city"] == ("irkutsk"
            if city_ru == "Иркутск" else "kazan")),
        None,
    )
    rows_raw = raw_match.get("rows_raw", n_unique) if raw_match else n_unique

    # completeness
    fields = KEY_FIELDS_PER_SOURCE[source]
    completeness = {}
    for f in fields:
        if f not in df.columns:
            completeness[f] = 0.0
        else:
            n_non_null = df.filter(pl.col(f).is_not_null()).shape[0]
            completeness[f] = round(100 * n_non_null / n_unique, 1) if n_unique else 0.0

    # ₽/м²: для Yandex есть price_per_sqm_rub отдельно;
    # для остальных — price_rub / total_area_m2
    if source == "yandex_realty" and "price_per_sqm_rub" in df.columns:
        ppsqm = df.select(
            pl.col("price_per_sqm_rub").cast(pl.Float64).alias("ppsqm")
        )
    else:
        ppsqm = df.select(
            (pl.col("price_rub").cast(pl.Float64)
             / pl.col("total_area_m2").cast(pl.Float64)).alias("ppsqm")
        )
    ppsqm = ppsqm.filter(pl.col("ppsqm").is_not_null()
                         & (pl.col("ppsqm") > 30000)
                         & (pl.col("ppsqm") < 1_000_000))
    n_valid_ppsqm = ppsqm.shape[0]

    if n_valid_ppsqm > 0:
        med_ppsqm = ppsqm.select(pl.col("ppsqm").median()).item()
        p25 = ppsqm.select(pl.col("ppsqm").quantile(0.25)).item()
        p75 = ppsqm.select(pl.col("ppsqm").quantile(0.75)).item()
    else:
        med_ppsqm = p25 = p75 = None

    med_price = (
        df.filter(pl.col("price_rub").is_not_null())
        .select(pl.col("price_rub").median()).item()
        if "price_rub" in df.columns else None
    )

    # dedup ratio
    dedup_ratio = round(1 - n_unique / rows_raw, 2) if rows_raw else 0.0

    return {
        "source": source,
        "city": city_ru,
        "rows_raw": rows_raw,
        "rows_unique": n_unique,
        "dedup_ratio": dedup_ratio,
        "n_cols": df.shape[1],
        "median_price_rub": int(med_price) if med_price else None,
        "median_ppsqm_rub": int(med_ppsqm) if med_ppsqm else None,
        "p25_ppsqm_rub": int(p25) if p25 else None,
        "p75_ppsqm_rub": int(p75) if p75 else None,
        "n_valid_ppsqm": n_valid_ppsqm,
        **{f"compl_{f}_pct": v for f, v in completeness.items()},
    }


def main() -> None:
    anchors = load_emiss_anchors()
    print(f"== ЕМИСС #61781 якоря 2025-Q4 (вторичный рынок) ==", flush=True)
    for c, v in anchors.items():
        print(f"  {c}: {v:.0f} ₽/м²", flush=True)

    rows = []
    for src in ("yandex_realty", "cian", "avito"):
        for city in ("Иркутск", "Казань"):
            r = stats_for_source_city(src, city)
            anchor = anchors.get(city)
            if anchor and r.get("median_ppsqm_rub"):
                r["anchor_delta_pct"] = round(
                    100 * (r["median_ppsqm_rub"] - anchor) / anchor, 1
                )
            else:
                r["anchor_delta_pct"] = None
            rows.append(r)

    df = pl.DataFrame(rows, infer_schema_length=20)
    df.write_csv(OUT_CSV)

    # короткая markdown-таблица
    cols = ["source", "city", "rows_unique", "rows_raw", "dedup_ratio",
            "median_price_rub", "median_ppsqm_rub", "anchor_delta_pct",
            "n_valid_ppsqm"]
    md = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    for r in rows:
        md += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    print("\n== Сводка ==", flush=True)
    print(md, flush=True)
    OUT_MD.write_text(
        f"# Сравнение источников листингов (Иркутск + Казань, 3 страницы)\n\n"
        f"Якорь: ЕМИСС #61781 2025-Q4 вторичный рынок — "
        f"Иркутск {anchors['Иркутск']:.0f} ₽/м², Казань {anchors['Казань']:.0f} ₽/м².\n\n"
        f"{md}\n",
        encoding="utf-8",
    )

    # completeness отдельно (он широкий)
    completeness_cols = [c for c in df.columns if c.startswith("compl_")]
    if completeness_cols:
        compl_df = df.select(["source", "city"] + completeness_cols)
        print("\n== Полнота ключевых полей (% non-null) ==", flush=True)
        with pl.Config(tbl_cols=20, tbl_width_chars=200):
            print(compl_df, flush=True)


if __name__ == "__main__":
    main()

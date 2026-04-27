"""Извлечение и унификация листингов из data/raw/listings-mvp/.

Берёт все скачанные страницы (3 источника × 2 города × N страниц), парсит
каждый источник своей стратегией:

* **CIAN** — встроенный JSON `"offers":[ ... ]` в HTML; самая богатая мета:
  build_year, materialType, deadline, lat/lon, описание, площади.
* **Avito** — JSON-LD `<script type="application/ld+json">` →
  `Product.offers (AggregateOffer).offers[]`. Schema.org-формат, name
  содержит «N-к. квартира, M м², F/F эт.», парсится regex'ом.
* **Yandex Realty** — нет структурированного JSON; SSR-карточки
  `data-test="OffersSerpItem"` парсятся regex'ом по DOM.

На выходе:
* `data/silver/listings-mvp/{source}_{city}.parquet` — per-source-city
  с полным набором полей источника (для CIAN-rich модели).
* `data/silver/listings-mvp/all.parquet` — объединённый long-format с
  общими ML-фичами для cross-source модели.
* `data/silver/listings-mvp/_summary.json`.

Schema unified:
  listing_id, source, city, price_rub, total_area_m2, rooms, floor,
  floors_count, floor_share, build_year (CIAN only), material_type
  (CIAN only), lat, lon, url, page_file, price_per_sqm_rub
"""

from __future__ import annotations

import html as _html
import json
import re
import sys
from pathlib import Path

import polars as pl

RAW = Path("data/raw/listings-mvp")
OUT = Path("data/silver/listings-mvp")
OUT.mkdir(parents=True, exist_ok=True)

CITY_RU = {"irkutsk": "Иркутск", "kazan": "Казань"}
RUB = "₽"


# ============================ PARSERS =======================================


# ---- CIAN ----


def extract_cian_offers(html: str) -> list[dict]:
    """Find embedded `"offers":[ ... ]` array in CIAN HTML and decode."""
    i = html.find('"offers":[')
    if i < 0:
        return []
    arr_start = html.find("[", i)
    decoder = json.JSONDecoder()
    try:
        arr, _end = decoder.raw_decode(html, arr_start)
    except json.JSONDecodeError as exc:
        print(f"  CIAN raw_decode err: {exc}", file=sys.stderr)
        return []
    return arr if isinstance(arr, list) else []


def flatten_cian(o: dict) -> dict:
    bld = o.get("building") or {}
    geo = (o.get("geo") or {}).get("coordinates") or {}
    return {
        "source": "cian",
        "id": o.get("id"),
        "category": o.get("category"),
        "deal_type": o.get("dealType"),
        "offer_type": o.get("offerType"),
        "price_full_rub_text": o.get("formattedFullPrice"),
        "price_rub": _rub_text_to_int(o.get("formattedFullPrice")),
        "url": o.get("fullUrl"),
        "title": o.get("title"),
        "description": (o.get("description") or "")[:600],
        "creation_date": o.get("creationDate"),
        "humanized_timedelta": o.get("humanizedTimedelta"),
        "additional_info": o.get("formattedAdditionalInfo"),
        "rooms_count": o.get("roomsCount"),
        "total_area_m2": o.get("totalArea") or o.get("totalAreaM2"),
        "living_area_m2": o.get("livingArea"),
        "kitchen_area_m2": o.get("kitchenArea"),
        "floor": o.get("floorNumber"),
        "build_year": bld.get("buildYear"),
        "floors_count": bld.get("floorsCount"),
        "material_type": bld.get("materialType"),
        "deadline_year": (bld.get("deadline") or {}).get("year"),
        "deadline_quarter": (bld.get("deadline") or {}).get("quarter"),
        "lat": geo.get("lat"),
        "lon": geo.get("lng"),
        "raw_offer_json": json.dumps(o, ensure_ascii=False),
    }


# ---- Avito ----


def extract_avito_offers(html: str) -> list[dict]:
    out = []
    for body in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.+?)</script>',
        html,
        re.DOTALL,
    ):
        try:
            j = json.loads(body.strip())
        except json.JSONDecodeError:
            continue
        graph = j.get("@graph") if isinstance(j, dict) else None
        candidates = graph if isinstance(graph, list) else [j]
        for it in candidates:
            if not isinstance(it, dict):
                continue
            if it.get("@type") != "Product":
                continue
            agg = it.get("offers")
            if not isinstance(agg, dict) or agg.get("@type") != "AggregateOffer":
                continue
            offers = agg.get("offers") or []
            for off in offers:
                if isinstance(off, dict):
                    out.append(off)
    return out


_AVITO_NAME_RE = re.compile(
    r"^(?:(\d+|студия)-?к\.\s*квартира|студия)"
    r"(?:[,\s]+([\d.,]+)\s*м[²2])?"
    r"(?:[,\s]+(\d+)/(\d+)\s*эт\.)?",
    re.IGNORECASE,
)


def flatten_avito(o: dict) -> dict:
    name = o.get("name") or ""
    rooms_str = area_str = floor = floors = None
    m = _AVITO_NAME_RE.match(name)
    if m:
        rooms_str = m.group(1)
        area_str = m.group(2)
        floor = m.group(3)
        floors = m.group(4)
    return {
        "source": "avito",
        "id": _avito_id_from_url(o.get("url")),
        "name": name,
        "url": o.get("url"),
        "image": o.get("image"),
        "valid_from": o.get("validFrom"),
        "availability": o.get("availability"),
        "price_rub": _to_int(o.get("price")),
        "price_currency": o.get("priceCurrency"),
        "rooms_str": rooms_str,
        "total_area_m2": _to_float(area_str),
        "floor": _to_int(floor),
        "floors_count": _to_int(floors),
    }


# ---- Yandex Realty ----


_YA_CARD_RE = re.compile(
    r'data-test="OffersSerpItem"(?P<rest>.*?)'
    r'(?=data-test="OffersSerpItem"|</body>|<footer)',
    re.DOTALL,
)
_YA_PPSQM_RE = re.compile(r"(\d[\d\s\xa0]+)\s*" + RUB + r"\s*за\s*м[²2]")
_YA_PRICE_RE = re.compile(r"(\d[\d\s\xa0]{4,})\s*" + RUB + r"(?!\s*за)")
_YA_AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*м[²2]")
_YA_ROOMS_RE = re.compile(r"(\d+)-комнатная\s*квартира|студия", re.IGNORECASE)
_YA_FLOOR_RE = re.compile(r"(\d+)\s*этаж\s*из\s*(\d+)|(\d+)\s*из\s*(\d+)\s*эт")
_YA_HREF_RE = re.compile(r'href="(/offer/\d+/?[^"]*)"')


def extract_yandex_cards(html_text: str) -> list[dict]:
    out = []
    for m in _YA_CARD_RE.finditer(html_text):
        chunk = m.group("rest")
        text = re.sub(r"<[^>]+>", " ", chunk)
        text = _html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        ppsqm_match = _YA_PPSQM_RE.search(text)
        price_match = _YA_PRICE_RE.search(text)
        area_match = _YA_AREA_RE.search(text)
        rooms_match = _YA_ROOMS_RE.search(text)
        floor_match = _YA_FLOOR_RE.search(text)
        href_match = _YA_HREF_RE.search(chunk)
        floor = floors = None
        if floor_match:
            g = floor_match.groups()
            floor = g[0] or g[2]
            floors = g[1] or g[3]
        out.append(
            {
                "source": "yandex_realty",
                "id": _ya_offer_id_from_href(href_match.group(1)) if href_match else None,
                "url": ("https://realty.yandex.ru" + href_match.group(1) if href_match else None),
                "raw_text": text[:400],
                "price_rub": (_to_int(re.sub(r"\D", "", price_match.group(1))) if price_match else None),
                "price_per_sqm_rub": (_to_int(re.sub(r"\D", "", ppsqm_match.group(1))) if ppsqm_match else None),
                "total_area_m2": (_to_float(area_match.group(1).replace(",", ".")) if area_match else None),
                "rooms_str": rooms_match.group(0) if rooms_match else None,
                "floor": _to_int(floor),
                "floors_count": _to_int(floors),
            }
        )
    return out


# ---- helpers ----


def _rub_text_to_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def _to_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(str(v).replace(" ", "").replace("\xa0", ""))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ".").replace(" ", "").replace("\xa0", ""))
    except (TypeError, ValueError):
        return None


def _avito_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"_(\d+)(?:[/?#]|$)", url)
    return m.group(1) if m else None


def _ya_offer_id_from_href(href: str | None) -> str | None:
    if not href:
        return None
    m = re.search(r"/offer/(\d+)", href)
    return m.group(1) if m else None


def _parse_rooms(rooms_str: str | None, rooms_count: int | None) -> int | None:
    """Извлечь число комнат: студия → 0, '1-комн.' → 1 и т.д."""
    if rooms_count is not None:
        return int(rooms_count)
    if not rooms_str:
        return None
    s = str(rooms_str).lower()
    if "студ" in s:
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


# ============================ PIPELINE ======================================


EXTRACTORS = {
    "yandex_realty": (extract_yandex_cards, lambda x: x),
    "cian": (extract_cian_offers, flatten_cian),
    "avito": (extract_avito_offers, flatten_avito),
}


def process_dir(source: str, city_slug: str) -> tuple[pl.DataFrame | None, dict]:
    page_dir = RAW / f"{source}_{city_slug}"
    if not page_dir.exists():
        return None, {"source": source, "city": city_slug, "rows_raw": 0, "note": "dir not found"}
    extractor, flattener = EXTRACTORS[source]
    rows: list[dict] = []
    pages = sorted(page_dir.glob("page-*.html"))
    per_page: list[int] = []
    for p in pages:
        try:
            html = p.read_text(encoding="utf-8")
        except Exception:
            per_page.append(0)
            continue
        if len(html) < 100_000:
            per_page.append(0)
            continue
        raw = extractor(html)
        flat = [{**flattener(r), "city": CITY_RU[city_slug], "page_file": p.name} for r in raw]
        rows.extend(flat)
        per_page.append(len(flat))
    if not rows:
        return None, {"source": source, "city": city_slug, "pages": len(pages), "rows_raw": 0, "per_page": per_page}
    df = pl.DataFrame(rows, infer_schema_length=20000)
    rows_raw = df.shape[0]
    dedup_col = "id" if "id" in df.columns else ("url" if "url" in df.columns else None)
    if dedup_col:
        df = df.unique(subset=[dedup_col], keep="first")
    rows_uniq = df.shape[0]
    out_path = OUT / f"{source}_{city_slug}.parquet"
    df.write_parquet(out_path)
    return df, {
        "source": source,
        "city": city_slug,
        "pages": len(pages),
        "rows_raw": rows_raw,
        "rows_unique": rows_uniq,
        "per_page": per_page,
        "out": str(out_path),
    }


def to_unified(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Привести к общей ML-схеме."""
    if source == "yandex_realty":
        out = df.select(
            [
                pl.col("id").cast(pl.Utf8).alias("listing_id"),
                pl.lit("yandex_realty").alias("source"),
                pl.col("city"),
                pl.col("price_rub").cast(pl.Float64),
                pl.col("total_area_m2").cast(pl.Float64),
                pl.col("rooms_str"),
                pl.col("floor").cast(pl.Float64),
                pl.col("floors_count").cast(pl.Float64),
                pl.lit(None, dtype=pl.Int64).alias("rooms_count"),
                pl.lit(None, dtype=pl.Int64).alias("build_year"),
                pl.lit(None, dtype=pl.Utf8).alias("material_type"),
                pl.lit(None, dtype=pl.Float64).alias("lat"),
                pl.lit(None, dtype=pl.Float64).alias("lon"),
                pl.col("url"),
                pl.col("page_file"),
            ]
        )
    elif source == "cian":
        out = df.select(
            [
                pl.col("id").cast(pl.Utf8).alias("listing_id"),
                pl.lit("cian").alias("source"),
                pl.col("city"),
                pl.col("price_rub").cast(pl.Float64),
                pl.col("total_area_m2").map_elements(_to_float, return_dtype=pl.Float64).alias("total_area_m2"),
                pl.lit(None, dtype=pl.Utf8).alias("rooms_str"),
                pl.col("floor").cast(pl.Float64),
                pl.col("floors_count").cast(pl.Float64),
                pl.col("rooms_count").cast(pl.Int64),
                pl.col("build_year").cast(pl.Int64),
                pl.col("material_type"),
                pl.col("lat").cast(pl.Float64),
                pl.col("lon").cast(pl.Float64),
                pl.col("url"),
                pl.col("page_file"),
            ]
        )
    elif source == "avito":
        out = df.select(
            [
                pl.col("id").cast(pl.Utf8).alias("listing_id"),
                pl.lit("avito").alias("source"),
                pl.col("city"),
                pl.col("price_rub").cast(pl.Float64),
                pl.col("total_area_m2").cast(pl.Float64),
                pl.col("rooms_str"),
                pl.col("floor").cast(pl.Float64),
                pl.col("floors_count").cast(pl.Float64),
                pl.lit(None, dtype=pl.Int64).alias("rooms_count"),
                pl.lit(None, dtype=pl.Int64).alias("build_year"),
                pl.lit(None, dtype=pl.Utf8).alias("material_type"),
                pl.lit(None, dtype=pl.Float64).alias("lat"),
                pl.lit(None, dtype=pl.Float64).alias("lon"),
                pl.col("url"),
                pl.col("page_file"),
            ]
        )
    else:
        raise ValueError(source)

    out = out.with_columns(
        [
            pl.when(pl.col("rooms_count").is_not_null())
            .then(pl.col("rooms_count"))
            .otherwise(pl.col("rooms_str").map_elements(lambda s: _parse_rooms(s, None), return_dtype=pl.Int64))
            .alias("rooms"),
        ]
    )
    out = out.with_columns(
        [
            (pl.col("price_rub") / pl.col("total_area_m2")).alias("price_per_sqm_rub"),
            (pl.col("floor") / pl.col("floors_count")).alias("floor_share"),
        ]
    )
    return out


def main() -> int:
    summary: list[dict] = []
    unified_frames: list[pl.DataFrame] = []
    for source in ("yandex_realty", "cian", "avito"):
        for city_slug in ("kazan", "irkutsk"):
            df, stats = process_dir(source, city_slug)
            summary.append(stats)
            if df is not None:
                unified_frames.append(to_unified(df, source))

    if unified_frames:
        all_df = pl.concat(unified_frames, how="diagonal_relaxed")
        all_df.write_parquet(OUT / "all.parquet")
        print(f"\n=> all.parquet: {all_df.shape}", flush=True)
        print(
            all_df.group_by(["source", "city"])
            .agg(
                [
                    pl.len().alias("rows"),
                    pl.col("price_rub").median().alias("med_price"),
                    pl.col("price_per_sqm_rub").median().alias("med_ppsqm"),
                ]
            )
            .sort(["source", "city"]),
            flush=True,
        )

    (OUT / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in summary:
        print(
            f"  {s.get('source')}/{s.get('city')}: pages={s.get('pages', 0)}"
            f" raw={s.get('rows_raw', 0)} uniq={s.get('rows_unique', 0)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

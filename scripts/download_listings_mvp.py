"""MVP-загрузка листингов: 50 страниц пагинации на источник-город,
3 источника (Yandex Realty / CIAN / Avito) × 2 города (Иркутск, Казань)
= **300 HTML страниц**. Через ту же patchright-сессию (system Chrome
+ persistent profile, headed). Между запросами 4-5с паузы. Это
**research-snapshot** для сравнения источников и обучения MVP-моделей.

Инкрементальный: если страница уже скачана и parsable — пропускаем.
Если упало в середине — повторный запуск продолжит со следующей.

Запуск:
    uv run --with patchright python scripts/download_listings_mvp.py
    uv run --with patchright python scripts/download_listings_mvp.py --n-pages 30  # быстрее

Лог в /tmp/listings_mvp_download.log + STDOUT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from patchright.sync_api import Page, sync_playwright

OUT = Path("data/raw/listings-mvp")
PROFILE_YANDEX = Path("data/raw/yandex-realty-probe/profile")
PROFILE_CIAN_AVITO = Path("data/raw/cian-avito-probe/profile")

PLAN: list[dict] = [
    {
        "source": "yandex_realty",
        "city": "irkutsk",
        "profile": PROFILE_YANDEX,
        "base": "https://realty.yandex.ru/irkutsk/kupit/kvartira/",
        "page_param": "page",
        "page_indexing": 0,
    },
    {
        "source": "yandex_realty",
        "city": "kazan",
        "profile": PROFILE_YANDEX,
        "base": "https://realty.yandex.ru/kazan/kupit/kvartira/",
        "page_param": "page",
        "page_indexing": 0,
    },
    # ВАЖНО: CIAN — cat.php URL с region_id. Subdomain
    # `<city>.cian.ru/kupit-kvartiru/?p=N` НЕ ПАГИНИРУЕТ (отдаёт топ-28
    # на любом p). Region для Казани=4777 (валидирован probe). Region
    # для Иркутской обл. **не валидирован** — перед запуском Иркутска
    # нужен отдельный probe (поиск через https://www.cian.ru/regions/).
    {
        "source": "cian",
        "city": "irkutsk",
        "profile": PROFILE_CIAN_AVITO,
        "base": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=__TODO_VALIDATE__",
        "page_param": "p",
        "page_indexing": 1,
    },
    {
        "source": "cian",
        "city": "kazan",
        "profile": PROFILE_CIAN_AVITO,
        "base": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4777",
        "page_param": "p",
        "page_indexing": 1,
    },
    {
        "source": "avito",
        "city": "irkutsk",
        "profile": PROFILE_CIAN_AVITO,
        "base": "https://www.avito.ru/irkutsk/kvartiry/prodam",
        "page_param": "p",
        "page_indexing": 1,
    },
    {
        "source": "avito",
        "city": "kazan",
        "profile": PROFILE_CIAN_AVITO,
        "base": "https://www.avito.ru/kazan/kvartiry/prodam",
        "page_param": "p",
        "page_indexing": 1,
    },
]

PER_PAGE_WAIT = 8
SCROLL_STEPS = 3
INTER_REQUEST_PAUSE = 4.5
MIN_HTML_SIZE = 100_000  # пустые/блок-страницы — обычно <50К
EMPTY_PAGE_SIZE = 200_000  # ниже — считаем «выдача исчерпана / антибот»
EMPTY_STREAK_LIMIT = 3  # столько подряд маленьких страниц → стоп для этого src/city


def safe_title(page: Page) -> str:
    try:
        return page.title()
    except Exception:
        return ""


def is_existing_ok(path: Path) -> bool:
    """Файл уже скачан и не похож на огрызок-блок."""
    return path.exists() and path.stat().st_size >= MIN_HTML_SIZE


def grab(page: Page, url: str, out_path: Path) -> dict:
    print(f"   GET {url}", flush=True)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as exc:
        print(f"      goto err: {exc}", flush=True)
        return {"url": url, "error": str(exc)}
    time.sleep(PER_PAGE_WAIT)
    for i in range(SCROLL_STEPS):
        try:
            page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {(i + 1) / SCROLL_STEPS})")
            time.sleep(1.2)
        except Exception:
            pass
    title = safe_title(page)
    final_url = page.url
    try:
        html = page.content()
        out_path.write_text(html, encoding="utf-8")
        size = len(html)
    except Exception as exc:
        print(f"      content err: {exc}", flush=True)
        return {"url": url, "final_url": final_url, "title": title, "error": str(exc)}
    print(f"      title={title[:90]!r}  size={size:,}", flush=True)
    return {"url": url, "final_url": final_url, "title": title, "size_bytes": size, "saved_to": str(out_path)}


def run_for_profile(profile_path: Path, plans_for_profile: list[dict], n_pages: int) -> list[dict]:
    out: list[dict] = []
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_path.resolve()),
            headless=False,
            channel="chrome",
            no_viewport=True,
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            viewport={"width": 1300, "height": 800},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        warm = plans_for_profile[0]["base"].split("/")[2]
        try:
            page.goto(f"https://{warm}/", wait_until="domcontentloaded", timeout=45000)
            time.sleep(4)
        except Exception:
            pass
        for plan in plans_for_profile:
            print(f"\n=> [{plan['source']}/{plan['city']}]", flush=True)
            page_dir = OUT / f"{plan['source']}_{plan['city']}"
            page_dir.mkdir(parents=True, exist_ok=True)
            empty_streak = 0
            for k in range(n_pages):
                page_idx = plan["page_indexing"] + k
                joiner = "&" if "?" in plan["base"] else "?"
                url = plan["base"] if k == 0 else f"{plan['base']}{joiner}{plan['page_param']}={page_idx}"
                page_file = page_dir / f"page-{page_idx:03d}.html"
                if is_existing_ok(page_file):
                    print(f"   SKIP {page_file.name} (already {page_file.stat().st_size:,}B)", flush=True)
                    out.append(
                        {
                            "url": url,
                            "saved_to": str(page_file),
                            "skipped": True,
                            "source": plan["source"],
                            "city": plan["city"],
                            "plan_index": page_idx,
                        }
                    )
                    empty_streak = 0  # уже был ОК → сбрасываем
                    continue
                rec = grab(page, url, page_file)
                rec["plan_index"] = page_idx
                rec["source"] = plan["source"]
                rec["city"] = plan["city"]
                out.append(rec)
                size = rec.get("size_bytes", 0)
                if size and size < EMPTY_PAGE_SIZE:
                    empty_streak += 1
                    print(f"   small-page streak={empty_streak} (size={size:,}B)", flush=True)
                    if empty_streak >= EMPTY_STREAK_LIMIT:
                        print(
                            f"   STOP {plan['source']}/{plan['city']}: "
                            f"{EMPTY_STREAK_LIMIT} мелких страниц подряд → "
                            f"выдача исчерпана / антибот",
                            flush=True,
                        )
                        break
                else:
                    empty_streak = 0
                time.sleep(INTER_REQUEST_PAUSE)
        ctx.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pages", type=int, default=50, help="страниц пагинации на каждый источник-город")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["kazan", "irkutsk"],
        choices=["kazan", "irkutsk"],
        help="какие города загружать (по умолчанию оба)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["yandex_realty", "cian", "avito"],
        choices=["yandex_realty", "cian", "avito"],
        help="какие источники загружать (по умолчанию все три)",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plans = [p for p in PLAN if p["city"] in args.cities and p["source"] in args.sources]
    print(f"=> загружаем {len(plans)} плана(ов): {', '.join(f'{p["source"]}/{p["city"]}' for p in plans)}", flush=True)
    by_profile: dict[Path, list[dict]] = {}
    for plan in plans:
        by_profile.setdefault(plan["profile"], []).append(plan)

    manifest_path = OUT / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("runs", []).append(
            {
                "started_at_utc": datetime.now(UTC).isoformat(),
                "n_pages": args.n_pages,
            }
        )
    else:
        manifest = {
            "snapshot_date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "n_pages": args.n_pages,
            "items": [],
            "runs": [],
        }

    all_items: list[dict] = []
    for prof, plans in by_profile.items():
        print(f"\n=== profile {prof} ===", flush=True)
        all_items.extend(run_for_profile(prof, plans, args.n_pages))

    manifest["items"] = all_items
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in all_items if "error" not in r)
    err = sum(1 for r in all_items if "error" in r)
    skipped = sum(1 for r in all_items if r.get("skipped"))
    fresh = ok - skipped
    print(f"\n=> manifest: {manifest_path}", flush=True)
    print(f"   ok={ok}  fresh={fresh}  skipped={skipped}  err={err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

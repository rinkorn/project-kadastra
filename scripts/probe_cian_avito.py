"""Probe CIAN + Avito через patchright (system Chrome, persistent profile,
headed). Открываем главные + страницу выдачи квартир Иркутска как тест,
сохраняем HTML и HAR. Цель — понять, проходим ли антибот, и какие XHR
делает страница (или это SSR с данными в HTML).

Запуск:
    uv run --with patchright python scripts/probe_cian_avito.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from patchright.sync_api import Page, sync_playwright

OUT = Path("data/raw/cian-avito-probe")
PAGES_OUT = OUT / "pages"
PROFILE = OUT / "profile"

# CIAN deal_type=2 = аренда; deal_type=1 = продажа; offer_type=flat
# region_id для Иркутска у CIAN = 4827, Москва = 1, Казань = 4777 (приблизительные;
# кликнем главную, если фильтрация не та — увидим в результате)
TARGETS: list[tuple[str, str]] = [
    ("00_cian_root", "https://www.cian.ru/"),
    (
        "01_cian_irkutsk_kupit",
        "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4827",
    ),
    (
        "02_cian_kazan_kupit",
        "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4777",
    ),
    ("03_avito_root", "https://www.avito.ru/"),
    ("04_avito_irkutsk", "https://www.avito.ru/irkutsk/kvartiry/prodam"),
    ("05_avito_kazan", "https://www.avito.ru/kazan/kvartiry/prodam"),
]

PER_PAGE_WAIT = 8
SCROLL_STEPS = 3


def safe_title(page: Page) -> str:
    try:
        return page.title()
    except Exception:  # noqa: BLE001
        return ""


def visit(page: Page, name: str, url: str) -> None:
    print(f"\n=> [{name}] {url}", flush=True)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as exc:  # noqa: BLE001
        print(f"   goto err: {exc}", flush=True)

    time.sleep(PER_PAGE_WAIT)

    for i in range(SCROLL_STEPS):
        try:
            page.evaluate(
                f"window.scrollTo(0, document.body.scrollHeight * {(i + 1) / SCROLL_STEPS})"
            )
            time.sleep(1.5)
        except Exception:  # noqa: BLE001, S110
            pass

    print(f"   url   : {page.url}", flush=True)
    print(f"   title : {safe_title(page)!r}", flush=True)

    try:
        html = page.content()
        (PAGES_OUT / f"{name}.html").write_text(html, encoding="utf-8")
        print(f"   html  : {len(html):,} байт", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"   content err: {exc}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PAGES_OUT.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)

    har_path = OUT / "network.har"
    if har_path.exists():
        har_path.unlink()

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE.resolve()),
            headless=False,
            channel="chrome",
            no_viewport=True,
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            record_har_path=str(har_path),
            record_har_content="embed",
            args=[
                "--no-default-browser-check",
                "--no-first-run",
            ],
        )

        page = context.pages[0] if context.pages else context.new_page()
        for name, url in TARGETS:
            visit(page, name, url)

        state = context.storage_state()
        (OUT / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"\n=> сохранено: cookies={len(state.get('cookies', []))} "
            f"origins={len(state.get('origins', []))}",
            flush=True,
        )
        context.close()

    print(f"=> артефакты: {OUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

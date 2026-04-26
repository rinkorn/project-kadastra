"""Probe Yandex Realty: открываем главную + страницы цен по нескольким
городам через patchright (system Chrome, persistent profile, headed),
запоминаем HAR со всеми XHR. Цель — найти API-эндпоинты с агрегатами
цен ₽/м² по регионам/городам.

Запуск:
    uv run --with patchright python scripts/probe_yandex_realty.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from patchright.sync_api import Page, sync_playwright

OUT = Path("data/raw/yandex-realty-probe")
PAGES_OUT = OUT / "pages"
PROFILE = OUT / "profile"

# Кандидатные URL: главная (прогрев), журнал-аналитика, городские страницы.
# Для городов пробуем оба формата slug; правильный будет редиректить на себя сам.
TARGETS: list[tuple[str, str]] = [
    ("00_root", "https://realty.yandex.ru/"),
    ("01_journal_analitika", "https://realty.yandex.ru/journal/category/analitika/"),
    ("02_calculator", "https://realty.yandex.ru/calculator-stoimosti/kvartira/"),
    ("03_kazan_cena", "https://realty.yandex.ru/kazan/cena-kvartiry/"),
    ("04_moskva_cena", "https://realty.yandex.ru/moskva/cena-kvartiry/"),
    ("05_spb_cena", "https://realty.yandex.ru/sankt-peterburg/cena-kvartiry/"),
    ("06_irkutsk_cena", "https://realty.yandex.ru/irkutsk/cena-kvartiry/"),
    ("07_irkutsk_kupit", "https://realty.yandex.ru/irkutsk/kupit/kvartira/"),
    ("08_kazan_kupit", "https://realty.yandex.ru/kazan/kupit/kvartira/"),
    ("09_moskva_kupit", "https://realty.yandex.ru/moskva/kupit/kvartira/"),
    # Альтернативные форматы для городского отчёта:
    ("10_irkutsk_obl_cena", "https://realty.yandex.ru/irkutskaya_oblast/cena-kvartiry/"),
    ("11_kazan_arenda", "https://realty.yandex.ru/kazan/snyat/kvartira/"),
]

PER_PAGE_WAIT = 8  # сек на «гидрацию»
SCROLL_STEPS = 4


def safe_title(page: Page) -> str:
    try:
        return page.title()
    except Exception:  # noqa: BLE001
        return ""


def visit(page: Page, name: str, url: str) -> None:
    print(f"\n=> [{name}] {url}", flush=True)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception as exc:  # noqa: BLE001
        print(f"   goto err: {exc}", flush=True)

    time.sleep(PER_PAGE_WAIT)

    for i in range(SCROLL_STEPS):
        try:
            page.evaluate(
                f"window.scrollTo(0, document.body.scrollHeight * {(i + 1) / SCROLL_STEPS})"
            )
            time.sleep(1.2)
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

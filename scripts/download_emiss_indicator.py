"""Скачиваем индикатор ЕМИСС/fedstat через реальный браузер (patchright).

fedstat.ru отдаёт plain-HTTP клиентам (curl/httpx) SPA-оболочку без данных
(TLS-fingerprint + JS-challenge на GET страницы индикатора), поэтому:

1. Открываем страницу индикатора в Chrome (persistent profile —
   data/raw/emiss-profile/, куки/JS-challenge переживают перезапуски).
2. Eval'ом вытаскиваем объект ``new FGrid({...})`` из inline-скрипта:
   там фильтры (filter_field_id → values {filter_value_id: title}) и
   раскладка pivot (left_columns/top_columns). Логика по мотивам
   R-пакета fedstatAPIr (fedstat_get_data_ids / parse_js1 / parse_js2).
3. POST form-urlencoded на https://www.fedstat.ru/indicator/data.do?format=excel
   прямо из контекста страницы (fetch) — все значения всех фильтров
   (полная выгрузка индикатора), бинарный xls возвращаем как base64.
4. Сохраняем в data/raw/emiss/{id}/raw_{дата}.xls (+ filters js рядом,
   чтобы парсер/отладка не ходили на сайт повторно).

Запуск:
    uv run --with patchright python scripts/download_emiss_indicator.py 31074
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.parse
from datetime import date
from pathlib import Path

from patchright.sync_api import sync_playwright

PAGE_URL = "https://www.fedstat.ru/indicator/{id}"
POST_URL = "https://www.fedstat.ru/indicator/data.do?format=excel"
PROFILE_DIR = Path("data/raw/emiss-profile")

WAIT_FILTERS_JS = """() => Array.from(document.querySelectorAll('script'))
  .some(x => (x.textContent || '').includes('left_columns')
         && (x.textContent || '').includes('filters:'))"""

# Balanced-brace extraction INSIDE the page: pull only the pure-data
# pieces (filters object + layout arrays). Evaluating the whole FGrid
# literal fails — it contains live calls like `block: $('#grid')` and
# jQuery is not guaranteed in the eval scope.
EXTRACT_FGRID_JS = """() => {
  const s = Array.from(document.querySelectorAll('script'))
    .map(x => x.textContent || '')
    .find(t => t.includes('left_columns') && t.includes('filters:'));
  if (!s) return null;
  function scanBalanced(str, openIdx, openCh, closeCh) {
    let depth = 0, quote = null;
    for (let i = openIdx; i < str.length; i++) {
      const c = str[i];
      if (quote) {
        if (c === '\\\\') { i++; continue; }
        if (c === quote) quote = null;
        continue;
      }
      if (c === "'" || c === '"') { quote = c; continue; }
      if (c === openCh) depth++;
      else if (c === closeCh) { depth--; if (depth === 0) return str.slice(openIdx, i + 1); }
    }
    return null;
  }
  function extract(str, key, openCh, closeCh) {
    const k = str.indexOf(key + ':');
    if (k < 0) return null;
    const open = str.indexOf(openCh, k + key.length + 1);
    return scanBalanced(str, open, openCh, closeCh);
  }
  const filtersLit = extract(s, 'filters', '{', '}');
  if (!filtersLit) return null;
  const filters = eval('(' + filtersLit + ')');
  const arr = (key) => {
    const lit = extract(s, key, '[', ']');
    return lit ? eval(lit) : [];
  };
  return JSON.stringify({
    filters,
    left_columns: arr('left_columns'), top_columns: arr('top_columns'),
    groups: arr('groups'), filterObjectIds: arr('filterObjectIds')
  });
}"""

POST_FETCH_JS = """async ({url, body}) => {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body
  });
  const contentType = resp.headers.get('content-type') || '';
  if (!resp.ok) return {status: resp.status, contentType};
  const bytes = new Uint8Array(await resp.arrayBuffer());
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return {status: resp.status, contentType, b64: btoa(binary)};
}"""


def build_post_body(fgrid: dict) -> str:
    """Form-urlencoded body по образцу fedstatAPIr::fedstat_post_data_ids_filtered.

    Все значения всех фильтров → полная выгрузка индикатора. Раскладка
    pivot (lineObjectIds/columnObjectIds) — дефолтная со страницы, как в
    parse_js2: left_columns/groups/filterObjectIds → lineObjectIds,
    top_columns → columnObjectIds; плюс filterObjectIds=0 (индикатор),
    если "0" не встретился среди object ids.
    """
    filters = fgrid["filters"]
    indicator_field = filters["0"]
    indicator_value_id, indicator_value = next(iter(indicator_field["values"].items()))

    pairs: list[tuple[str, str]] = [
        ("format", "excel"),
        ("id", indicator_value_id),
        ("indicator_title", indicator_value["title"]),
    ]

    object_ids_seen: list[str] = []
    for key in ("left_columns", "groups", "filterObjectIds"):
        for fid in fgrid.get(key) or []:
            pairs.append(("lineObjectIds", str(fid)))
            object_ids_seen.append(str(fid))
    for fid in fgrid.get("top_columns") or []:
        pairs.append(("columnObjectIds", str(fid)))
        object_ids_seen.append(str(fid))
    if "0" not in object_ids_seen:
        pairs.append(("filterObjectIds", "0"))

    for fid, field in filters.items():
        for vid in field.get("values", {}):
            pairs.append(("selectedFilterIds", f"{fid}_{vid}"))

    return urllib.parse.urlencode(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("indicator_id")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()

    out_dir = args.out_dir or Path("data/raw/emiss") / args.indicator_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            channel="chrome",
            no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        raw = None
        for attempt in range(6):
            try:
                page.goto(PAGE_URL.format(id=args.indicator_id), timeout=120_000)
                # Ручной поллинг вместо wait_for_function: на обрезанной
                # SPA-оболочке собственный JS fedstat падает, и playwright
                # пробрасывает чужой TypeError как ошибку нашего wait.
                for _ in range(30):
                    try:
                        if page.evaluate(WAIT_FILTERS_JS):
                            break
                    except Exception:
                        pass
                    page.wait_for_timeout(3000)
                raw = page.evaluate(EXTRACT_FGRID_JS)
                if raw:
                    break
                print(f"attempt {attempt + 1}: filters script not in DOM yet", flush=True)
            except Exception as e:  # fedstat лагает — ретраим с backoff
                print(f"attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
            time.sleep(5 * (attempt + 1))
        if not raw:
            ctx.close()
            sys.exit(f"filters script never loaded for indicator {args.indicator_id}")

        (out_dir / f"filters_{args.date}.json").write_text(raw, encoding="utf-8")
        fgrid = json.loads(raw)
        indicator_field = fgrid["filters"]["0"]
        indicator_value = next(iter(indicator_field["values"].values()))
        print(f"=> {args.indicator_id}: {indicator_value['title']}", flush=True)
        for fid, field in fgrid["filters"].items():
            print(f"   field {fid}: {field.get('title')!r} nvalues={len(field.get('values', {}))}", flush=True)

        body = build_post_body(fgrid)
        print(f"=> POST body params: {body.count('&') + 1}", flush=True)
        result = None
        for attempt in range(4):
            result = page.evaluate(POST_FETCH_JS, {"url": POST_URL, "body": body})
            print(
                f"=> POST attempt {attempt + 1}: status={result.get('status')} "
                f"content-type={result.get('contentType')}",
                flush=True,
            )
            if result.get("b64"):
                break
            time.sleep(5 * (attempt + 1))
        ctx.close()

    if not result or not result.get("b64"):
        sys.exit(f"POST data.do failed: {result}")
    payload = base64.b64decode(result["b64"])
    dst = out_dir / f"raw_{args.date}.xls"
    dst.write_bytes(payload)
    print(f"=> wrote {dst}  bytes={len(payload):,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Скачиваем индикатор ЕМИСС/fedstat через реальный браузер (patchright).

fedstat.ru отдаёт plain-HTTP клиентам (curl/httpx) SPA-оболочку без данных
(TLS-fingerprint + JS-challenge на GET страницы индикатора), поэтому:

1. Открываем страницу индикатора в Chrome (persistent profile —
   data/raw/emiss-profile/, куки/JS-challenge переживают перезапуски).
2. Eval'ом вытаскиваем объект ``new FGrid({...})`` из inline-скрипта:
   там фильтры (filter_field_id → values {filter_value_id: title}) и
   раскладка pivot (left_columns/top_columns). Логика по мотивам
   R-пакета fedstatAPIr (fedstat_get_data_ids / parse_js1 / parse_js2).
3. Скачивание воспроизводит родной экспорт UI (FGrid.downloadFile):
   form POST на /indicator/downloadData.do?format=excel с hidden-полями
   (id, lineObjectIds/columnObjectIds/groupObjectIds/filterObjectIds,
   selectedFilterIds на КАЖДОЕ значение каждого фильтра = полная
   выгрузка, title и struts-токен из #downloadTokenHolder). Файл
   ловим через expect_download. Важно: data.do?format=excel из
   fedstatAPIr больше не работает — сервер отдаёт HTML-страницу
   индикатора; актуальный endpoint — downloadData.do + struts token.
4. Сохраняем в data/raw/emiss/{id}/raw_{дата}.xls (+ filters json рядом,
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

# fetch POST на downloadData.do (актуальный export endpoint; data.do из
# fedstatAPIr мёртв — отдаёт HTML). Struts token обязателен — берём из
# #downloadTokenHolder. Бинарный xls возвращаем как base64.
POST_DOWNLOAD_JS = """async ({url, body}) => {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'},
    body
  });
  const contentType = resp.headers.get('content-type') || '';
  const disposition = resp.headers.get('content-disposition') || '';
  const bytes = new Uint8Array(await resp.arrayBuffer());
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return {status: resp.status, contentType, disposition, size: bytes.length, b64: btoa(binary)};
}"""

# Родной экспорт (FGrid.downloadFile): <form method=post> + submit —
# браузер качает файл сам (стриминг на диск), playwright ловит через
# expect_download. Фолбэк для крупных выгрузок, где fetch+b64 падает.
SUBMIT_DOWNLOAD_JS = """({format, title, pairs, tokenName, token}) => {
  const form = document.createElement('form');
  form.method = 'post';
  form.action = '/indicator/downloadData.do?format=' + format;
  const add = (name, value) => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = name;
    input.value = value;
    form.appendChild(input);
  };
  add('title', title);
  add('struts.token.name', tokenName);
  add('token', token);
  for (const [name, value] of pairs) add(name, value);
  form.style.display = 'none';
  document.body.appendChild(form);
  form.submit();
}"""

READ_TOKEN_JS = """() => {
  const holder = document.querySelector('#downloadTokenHolder');
  if (!holder) return null;
  const nameInput = holder.querySelector('input[name="struts.token.name"]');
  const tokenInput = holder.querySelector('input[name="token"]');
  if (!nameInput || !tokenInput) return null;
  return {tokenName: nameInput.value, token: tokenInput.value};
}"""


def build_download_params(fgrid: dict) -> tuple[str, list[tuple[str, str]]]:
    """Параметры для downloadData.do по образцу FGrid.savePreview(true).

    Раскладка pivot — дефолтная со страницы: left_columns →
    lineObjectIds, top_columns → columnObjectIds, groups →
    groupObjectIds, filterObjectIds → filterObjectIds. selectedFilterIds
    — на каждое значение каждого фильтра (полная выгрузка индикатора).
    Возвращает (indicator_title, pairs).
    """
    filters = fgrid["filters"]
    indicator_value = next(iter(filters["0"]["values"].values()))

    pairs: list[tuple[str, str]] = [("id", str(fgrid_id(filters)))]
    for fid in fgrid.get("left_columns") or []:
        pairs.append(("lineObjectIds", str(fid)))
    for fid in fgrid.get("top_columns") or []:
        pairs.append(("columnObjectIds", str(fid)))
    for fid in fgrid.get("groups") or []:
        pairs.append(("groupObjectIds", str(fid)))
    for fid, field in filters.items():
        for vid in field.get("values", {}):
            pairs.append(("selectedFilterIds", f"{fid}_{vid}"))
    for fid in fgrid.get("filterObjectIds") or []:
        pairs.append(("filterObjectIds", str(fid)))

    return indicator_value["title"], pairs


def fgrid_id(filters: dict) -> str:
    """Id индикатора = единственный value id поля '0' («Показатель»)."""
    return next(iter(filters["0"]["values"].keys()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("indicator_id")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument(
        "--format",
        choices=["excel", "sdmx"],
        default="excel",
        help="excel — pivot xls (названия без кодов территорий); "
        "sdmx — XML с кодами классификаторов (ОКТМО/ОКАТО), нужен для джойнов",
    )
    args = ap.parse_args()

    ext = "xls" if args.format == "excel" else "xml"
    ok_content_type = "application/vnd.ms-excel" if args.format == "excel" else "text/xml"

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

        token = page.evaluate(READ_TOKEN_JS)
        if not token:
            ctx.close()
            sys.exit("struts token (#downloadTokenHolder) not found on page")

        title, pairs = build_download_params(fgrid)
        pairs.append(("struts.token.name", token["tokenName"]))
        pairs.append(("token", token["token"]))
        pairs.append(("title", title))
        body = urllib.parse.urlencode(pairs)
        print(f"=> download params: {len(pairs)}", flush=True)

        result = None
        for attempt in range(4):
            result = page.evaluate(
                POST_DOWNLOAD_JS,
                {
                    "url": f"https://www.fedstat.ru/indicator/downloadData.do?format={args.format}",
                    "body": body,
                },
            )
            print(
                f"=> POST attempt {attempt + 1}: status={result.get('status')} "
                f"ct={result.get('contentType')} size={result.get('size')}",
                flush=True,
            )
            if result.get("contentType", "").startswith(ok_content_type):
                break
            time.sleep(5 * (attempt + 1))

        dst = out_dir / f"raw_{args.date}.{ext}"
        if result and result.get("contentType", "").startswith(ok_content_type):
            payload = base64.b64decode(result["b64"])
            dst.write_bytes(payload)
            ctx.close()
        else:
            # fetch+b64 падает на крупных выгрузках ("Failed to fetch") —
            # фолбэк на родной form-submit, файл стримится браузером на диск.
            print("=> fetch failed, fallback to form submit + expect_download", flush=True)
            with page.expect_download(timeout=900_000) as dl_info:
                page.evaluate(
                    SUBMIT_DOWNLOAD_JS,
                    {
                        "format": args.format,
                        "title": title,
                        "pairs": pairs,
                        "tokenName": token["tokenName"],
                        "token": token["token"],
                    },
                )
            download = dl_info.value
            print(f"=> download: {download.suggested_filename}", flush=True)
            download.save_as(str(dst))
            ctx.close()

    size = dst.stat().st_size
    print(f"=> wrote {dst}  bytes={size:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

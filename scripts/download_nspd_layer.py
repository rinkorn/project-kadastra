"""Polite, resumable bulk downloader for NSPD attrib-search.

One JSON file per page under ``--out-dir/page-NNNN.json`` plus a sidecar
``_state.json`` with progress. Re-running the script picks up where it
left off (any page already on disk is skipped).

Defaults are deliberately conservative: 1 connection, 2-second base delay
with ±0.5 sec jitter, exponential backoff on transient errors, hard stop
on a NSPD WAF rule (HTTP 403 with ``Rule:`` header). The Russian
Trusted CA is not in certifi, so we disable verification for this
read-only public-data scrape; production code should add the Russian
root to its CA bundle.

Examples
--------
    # Все земельные участки с адресом ⊂ "Казань"
    uv run python scripts/download_nspd_layer.py \\
        --layer-id 36048 \\
        --out-dir data/raw/nspd/landplots-kazan \\
        --address-contains "Казань"

    # Smoke-test: 5 страниц зданий
    uv run python scripts/download_nspd_layer.py \\
        --layer-id 36049 \\
        --out-dir /tmp/smoke_buildings \\
        --address-contains "Казань" \\
        --max-pages 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

NSPD_BASE = "https://nspd.gov.ru"
ATTRIB_SEARCH_PATH = "/api/geoportal/v3/geoportal/{layer_id}/attrib-search"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass(slots=True)
class FetchPlan:
    layer_id: int
    body: dict[str, Any]
    out_dir: Path
    count_per_page: int
    rate_limit_sec: float
    jitter_sec: float
    max_pages: int | None
    timeout_sec: float


class NspdWafBlocked(RuntimeError):
    """Raised when NSPD WAF returns 403 — we stop hard."""


def build_address_filter(address_contains: str) -> dict[str, Any]:
    return {
        "textQueryAttrib": [
            {"keyName": "options.readable_address", "value": address_contains}
        ]
    }


def page_path(out_dir: Path, page: int) -> Path:
    return out_dir / f"page-{page:04d}.json"


def state_path(out_dir: Path) -> Path:
    return out_dir / "_state.json"


def load_state(out_dir: Path) -> dict[str, Any] | None:
    p = state_path(out_dir)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(out_dir: Path, state: dict[str, Any]) -> None:
    state_path(out_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fetch_page(
    client: httpx.Client, layer_id: int, body: dict[str, Any], page: int, count: int
) -> dict[str, Any]:
    url = NSPD_BASE + ATTRIB_SEARCH_PATH.format(layer_id=layer_id)
    params = {"page": page, "count": count, "withTotalCount": "true"}

    backoff_schedule = [30, 90, 300]  # 30s, 1.5min, 5min
    attempt = 0
    while True:
        try:
            r = client.post(url, json=body, params=params)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt >= len(backoff_schedule):
                raise
            wait = backoff_schedule[attempt]
            print(f"  [page={page}] network error: {e!r}; sleeping {wait}s", flush=True)
            time.sleep(wait)
            attempt += 1
            continue

        if r.status_code == 403:
            rule_hdr = "Rule:" in r.text
            raise NspdWafBlocked(
                f"WAF 403 on page={page}; "
                f"likely IP-level block. Body: {r.text[:200]!r} (rule_hint={rule_hdr})"
            )

        if r.status_code in (429, 503):
            if attempt >= len(backoff_schedule):
                r.raise_for_status()
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff_schedule[attempt]
            print(f"  [page={page}] {r.status_code}; sleeping {wait}s", flush=True)
            time.sleep(wait)
            attempt += 1
            continue

        r.raise_for_status()
        return r.json()


def run(plan: FetchPlan) -> None:
    plan.out_dir.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(
        verify=False,
        timeout=plan.timeout_sec,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
            "Origin": NSPD_BASE,
            "Referer": f"{NSPD_BASE}/map",
            "Content-Type": "application/json",
        },
        http2=False,
    )

    state = load_state(plan.out_dir) or {}
    started_at = state.get("started_at") or time.time()

    page = 0
    pages_fetched = 0
    total = state.get("total")
    last_total: int | None = None

    try:
        while True:
            if plan.max_pages is not None and pages_fetched >= plan.max_pages:
                print(f"Reached max-pages={plan.max_pages}, stopping.", flush=True)
                break

            existing = page_path(plan.out_dir, page)
            if existing.is_file():
                page += 1
                continue

            payload = fetch_page(
                client, plan.layer_id, plan.body, page=page, count=plan.count_per_page
            )

            features = payload.get("data", {}).get("features", [])
            meta = payload.get("meta") or [{}]
            page_total = meta[0].get("totalCount") if meta else None
            if page_total is not None:
                total = page_total
                last_total = page_total

            existing.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            pages_fetched += 1
            elapsed = time.time() - started_at
            done_objs = (page + 1) * plan.count_per_page
            print(
                f"  page={page:4d} got={len(features):3d} total={total} "
                f"~done={min(done_objs, total or done_objs)} elapsed={elapsed:.0f}s",
                flush=True,
            )

            save_state(
                plan.out_dir,
                {
                    "layer_id": plan.layer_id,
                    "body": plan.body,
                    "count_per_page": plan.count_per_page,
                    "total": total,
                    "last_page_done": page,
                    "started_at": started_at,
                    "updated_at": time.time(),
                },
            )

            if not features:
                print("Empty page — done.", flush=True)
                break
            if total is not None and (page + 1) * plan.count_per_page >= total:
                print(f"Covered totalCount={total} — done.", flush=True)
                break

            page += 1
            time.sleep(plan.rate_limit_sec + random.uniform(0, plan.jitter_sec))
    finally:
        client.close()

    if last_total is not None:
        save_state(
            plan.out_dir,
            {
                **(load_state(plan.out_dir) or {}),
                "completed_at": time.time(),
            },
        )


def parse_args(argv: list[str] | None = None) -> FetchPlan:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer-id", type=int, required=True, help="NSPD layer id (e.g. 36048 parcels, 36049 buildings)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--address-contains",
        type=str,
        default=None,
        help="Convenience filter: textQueryAttrib options.readable_address ⊂ <value>",
    )
    p.add_argument(
        "--filter-json",
        type=str,
        default=None,
        help="Raw JSON body, takes precedence over --address-contains",
    )
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--rate-limit-sec", type=float, default=2.0)
    p.add_argument("--jitter-sec", type=float, default=0.5)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--timeout-sec", type=float, default=30.0)
    a = p.parse_args(argv)

    if a.filter_json is not None:
        body = json.loads(a.filter_json)
    elif a.address_contains is not None:
        body = build_address_filter(a.address_contains)
    else:
        sys.exit("Provide --address-contains or --filter-json")

    return FetchPlan(
        layer_id=a.layer_id,
        body=body,
        out_dir=a.out_dir,
        count_per_page=a.count,
        rate_limit_sec=a.rate_limit_sec,
        jitter_sec=a.jitter_sec,
        max_pages=a.max_pages,
        timeout_sec=a.timeout_sec,
    )


def main() -> None:
    plan = parse_args()
    print(
        f"Layer {plan.layer_id} → {plan.out_dir}; count={plan.count_per_page}; "
        f"rate={plan.rate_limit_sec}±{plan.jitter_sec}s; max_pages={plan.max_pages}",
        flush=True,
    )
    print(f"Body: {json.dumps(plan.body, ensure_ascii=False)}", flush=True)
    try:
        run(plan)
    except NspdWafBlocked as e:
        print(f"STOP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()

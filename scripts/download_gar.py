"""Selective download of ГАР (FIAS) XML files for region 16 (Татарстан).

ADR-0015 picks 4 of the 18 files from ``s3://{bucket}/Kadatastr/gar_xml/16/``:

- AS_ADDR_OBJ            — names/types of address objects (street → district → subject).
- AS_MUN_HIERARCHY       — municipal hierarchy + OKTMO at each node.
- AS_HOUSES_PARAMS       — TYPEID=8 → cadnum → objectid (for buildings).
- AS_STEADS_PARAMS       — TYPEID=8 → cadnum → objectid (for parcels).

Files have UUID suffixes (e.g. ``AS_HOUSES_PARAMS_20260406_…XML``); we
match by family prefix and download the latest if multiple.

Idempotent: skips a file when the local copy exists with matching size.

Earlier we also downloaded AS_HOUSES, AS_STEADS, AS_REESTR_OBJECTS;
on closer inspection AS_REESTR_OBJECTS does not carry CADNUM (it lives
in *_PARAMS under TYPEID=8) and AS_HOUSES/AS_STEADS only carry physical
attrs (house number, type) that we don't need for block 4. Leaving the
previously-downloaded files on disk; they are harmless."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

# 4 minimal-set families per ADR-0015 (revised). Order = download priority.
_FAMILIES = [
    "AS_ADDR_OBJ_",  # smallest, sanity check first
    "AS_HOUSES_PARAMS_",  # cadnum → objectid for buildings
    "AS_STEADS_PARAMS_",  # cadnum → objectid for parcels
    "AS_MUN_HIERARCHY_",  # walk to municipality
    "AS_ADM_HIERARCHY_",  # intra-city raions (Советский / Приволжский …)
]
_PREFIX = "Kadatastr/gar_xml/16/"
_OUT_ROOT = Path("data/raw/gar/16")


def _belongs(family: str, key: str) -> bool:
    """True iff ``key`` matches the family (rest must start with the date prefix)."""
    name = key.removeprefix(_PREFIX)
    if not name.startswith(family):
        return False
    rest = name.removeprefix(family)
    return rest[:1].isdigit()


def main() -> None:
    load_dotenv()
    bucket = os.environ["S3_BUCKET"]
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": os.environ.get("S3_ADDRESSING_STYLE", "path")}),
    )

    paginator = client.get_paginator("list_objects_v2")
    listing: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=_PREFIX):
        listing.extend(page.get("Contents") or [])

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    skipped = 0
    downloaded = 0
    for family in _FAMILIES:
        candidates = [obj for obj in listing if _belongs(family, obj["Key"])]
        if not candidates:
            print(f"WARN: family {family} not found under {_PREFIX}", file=sys.stderr)
            continue
        # Prefer the lexicographically largest key — UUID suffix has no
        # natural ordering, but date prefix (YYYYMMDD) does, so latest
        # snapshot wins if the bucket ever holds multiples.
        obj = max(candidates, key=lambda o: o["Key"])
        key = obj["Key"]
        size = obj["Size"]
        out_path = _OUT_ROOT / Path(key).name
        if out_path.is_file() and out_path.stat().st_size == size:
            skipped += 1
            print(f"SKIP  {out_path.name}  ({size / 1024 / 1024:.1f} MB)")
            continue
        print(f"GET   {out_path.name}  ({size / 1024 / 1024:.1f} MB)  → {out_path}")
        client.download_file(bucket, key, str(out_path))
        downloaded += 1
        total_bytes += size

    print()
    print(f"done: downloaded={downloaded}  skipped={skipped}  new_bytes={total_bytes / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()

"""Idempotent downloader: mirror an S3 prefix to a local directory tree.

Reverse of ``upload_dir_to_s3.py``. Walks ``s3://{bucket}/{prefix}/``
recursively and pulls each object into ``{dst}/{relpath}``. Skips any
local file whose byte size already matches the remote — so the script
is safe to schedule on a cold-start container, and re-runs after a
partial pull pick up where they stopped.

Reads S3 credentials from the environment (same vars as ``Settings``):
``S3_ENDPOINT_URL``, ``S3_BUCKET``, ``S3_ACCESS_KEY``, ``S3_SECRET_KEY``,
``S3_REGION``, ``S3_ADDRESSING_STYLE``. ``.env`` is loaded if present.

Example
-------
    uv run python scripts/download_dir_from_s3.py \\
        --prefix Kadatastr/gold \\
        --dst data/gold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from kadastra.config import Settings


def make_client(s: Settings):
    if not (s.s3_endpoint_url and s.s3_bucket and s.s3_access_key and s.s3_secret_key):
        sys.exit("S3 credentials not set in .env (S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY)")
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(s3={"addressing_style": s.s3_addressing_style}),
    )


def iter_remote(client, bucket: str, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            yield obj["Key"], int(obj["Size"])


def download(prefix: str, dst: Path, *, dry_run: bool = False) -> tuple[int, int, int]:
    """Returns (downloaded, skipped, failed)."""
    settings = Settings()
    client = make_client(settings)
    bucket = settings.s3_bucket
    assert bucket is not None  # checked in make_client
    prefix = prefix.rstrip("/")

    print(
        f"Downloading s3://{bucket}/{prefix}/ → {dst}/  (endpoint={settings.s3_endpoint_url})",
        flush=True,
    )

    keys = list(iter_remote(client, bucket, f"{prefix}/"))
    print(f"  found {len(keys)} remote objects", flush=True)

    downloaded = skipped = failed = 0
    for i, (key, remote_size) in enumerate(keys, 1):
        rel = key[len(prefix) + 1 :]
        if not rel:
            continue
        local_path = dst / rel
        local_size = local_path.stat().st_size if local_path.is_file() else None

        if local_size == remote_size:
            skipped += 1
            if i % 200 == 0 or i == len(keys):
                print(
                    f"  [{i:4d}/{len(keys)}] {rel}  skipped (size match {remote_size}B)",
                    flush=True,
                )
            continue

        if dry_run:
            print(
                f"  [{i:4d}/{len(keys)}] would download {rel} ({remote_size}B; local={local_size})",
                flush=True,
            )
            continue

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            downloaded += 1
            if downloaded == 1 or downloaded % 50 == 0 or i == len(keys):
                print(
                    f"  [{i:4d}/{len(keys)}] {rel}  ok ({remote_size}B)",
                    flush=True,
                )
        except ClientError as e:
            print(f"  [{i:4d}/{len(keys)}] GET {key}: {e}", flush=True)
            failed += 1

    return downloaded, skipped, failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="S3 prefix to mirror (no leading slash)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Local destination directory (created if missing)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = download(args.prefix, args.dst, dry_run=args.dry_run)
    print(
        f"\nDone. downloaded={downloaded} skipped={skipped} failed={failed}",
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

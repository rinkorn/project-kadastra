"""Idempotent uploader for a local directory tree to S3.

Walks ``--src`` recursively and uploads files under ``--prefix`` in the
configured bucket (Synology S3, path-style addressing). Skips any key
that already exists with the same byte size — so re-runs after a partial
failure pick up where they stopped, and the script is safe to schedule
on a directory that already has most of its content uploaded.

Reads S3 credentials from the environment (same vars as ``Settings``):
``S3_ENDPOINT_URL``, ``S3_BUCKET``, ``S3_ACCESS_KEY``, ``S3_SECRET_KEY``,
``S3_REGION``, ``S3_ADDRESSING_STYLE``. ``.env`` is loaded if present.

Example
-------
    uv run python scripts/upload_dir_to_s3.py \\
        --src data/raw/nspd/buildings-kazan \\
        --prefix Kadatastr/nspd/buildings-kazan
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
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


def remote_size(client, bucket: str, key: str) -> int | None:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        return int(head["ContentLength"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def iter_files(src: Path) -> Iterator[Path]:
    for path in sorted(src.rglob("*")):
        if path.is_file():
            yield path


def upload(
    src: Path, prefix: str, *, dry_run: bool = False, content_type: str | None = None
) -> tuple[int, int, int]:
    """Returns (uploaded, skipped, failed)."""
    settings = Settings()
    client = make_client(settings)
    bucket = settings.s3_bucket
    assert bucket is not None  # checked in make_client
    prefix = prefix.rstrip("/")

    uploaded = skipped = failed = 0
    files = list(iter_files(src))
    print(
        f"Uploading {len(files)} files from {src} → "
        f"s3://{bucket}/{prefix}/  (endpoint={settings.s3_endpoint_url})",
        flush=True,
    )

    for i, path in enumerate(files, 1):
        rel = path.relative_to(src).as_posix()
        key = f"{prefix}/{rel}"
        local_size = path.stat().st_size

        try:
            existing = remote_size(client, bucket, key)
        except ClientError as e:
            print(f"  [{i:4d}/{len(files)}] HEAD {key}: {e}", flush=True)
            failed += 1
            continue

        if existing == local_size:
            skipped += 1
            if i % 200 == 0 or i == len(files):
                print(
                    f"  [{i:4d}/{len(files)}] {key}  skipped (size match {local_size}B)",
                    flush=True,
                )
            continue

        if dry_run:
            print(
                f"  [{i:4d}/{len(files)}] would upload {key} ({local_size}B; remote={existing})",
                flush=True,
            )
            continue

        try:
            extra: dict[str, str] = {}
            if content_type:
                extra["ContentType"] = content_type
            client.upload_file(str(path), bucket, key, ExtraArgs=extra or None)
            uploaded += 1
            if uploaded == 1 or uploaded % 50 == 0 or i == len(files):
                print(
                    f"  [{i:4d}/{len(files)}] {key}  ok ({local_size}B)",
                    flush=True,
                )
        except ClientError as e:
            print(f"  [{i:4d}/{len(files)}] PUT {key}: {e}", flush=True)
            failed += 1

    return uploaded, skipped, failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Local directory to upload")
    p.add_argument("--prefix", type=str, required=True, help="S3 prefix (no leading slash)")
    p.add_argument("--content-type", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.src.is_dir():
        sys.exit(f"--src must be an existing directory; got {args.src}")

    uploaded, skipped, failed = upload(
        args.src, args.prefix, dry_run=args.dry_run, content_type=args.content_type
    )
    print(
        f"\nDone. uploaded={uploaded} skipped={skipped} failed={failed}",
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

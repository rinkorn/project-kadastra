"""Download a single S3 object to a local file.

Companion to ``download_dir_from_s3.py`` (which mirrors whole prefixes)
for the cases where one file out of a large prefix is needed — e.g. the
OSM raions GeoJSON-seq out of the multi-GB ``Kadatastr/raw/osm/`` tree.

Reads S3 credentials from the environment (same vars as ``Settings``);
``.env`` is loaded if present.

Example
-------
    uv run python scripts/download_file_from_s3.py \\
        --key Kadatastr/raw/osm/kazan-agg-raions.geojsonseq \\
        --dst data/raw/osm/kazan-agg-raions.geojsonseq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from kadastra.config import Settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--key", type=str, required=True, help="Full S3 object key")
    p.add_argument("--dst", type=Path, required=True, help="Local destination file path")
    args = p.parse_args()

    s = Settings()
    if not (s.s3_endpoint_url and s.s3_bucket and s.s3_access_key and s.s3_secret_key):
        sys.exit("S3 credentials not set in .env (S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY)")
    client = boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(s3={"addressing_style": s.s3_addressing_style}),
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(s.s3_bucket, args.key, str(args.dst))
    except ClientError as e:
        sys.exit(f"GET s3://{s.s3_bucket}/{args.key}: {e}")
    print(f"Downloaded s3://{s.s3_bucket}/{args.key} → {args.dst}", flush=True)


if __name__ == "__main__":
    main()

#!/bin/sh
set -e

# S3 pull runs in the background so the app comes up immediately and
# /health responds before the (potentially multi-GB) sync finishes —
# critical for the GHA deploy step's 150s healthcheck window. The
# downloader is idempotent (size-match skip), so a re-run after the
# first cold-start is cheap. Endpoints that need data files will see
# them gradually appear; the app already tolerates a missing parquet
# (404 from /api/hex_aggregates etc.) rather than crashing.
#
# Pull prefixes are explicit: by default we mirror only the runtime-
# critical bits (`gold/` for hex aggregates and inspection, `models/`
# for quartet OOFs, `silver/` for EMISS market reference + road graph
# + GAR lookups). ETL-only prefixes (`gar_xml/` 3 GB+ XML dumps, raw
# OSM/NSPD, listings, …) are *not* pulled — they live on S3 as cold
# storage and only need to be present on the dev box that rebuilds
# features. This keeps cold-start to ~1 minute on LAN instead of
# blocking the inspector behind tens of GB of irrelevant data.
DATA_DST="${PULL_DATA_ON_START_DST:-data}"
DEFAULT_PREFIXES="Kadatastr/gold,Kadatastr/models,Kadatastr/silver"
PREFIXES="${PULL_DATA_ON_START_PREFIXES:-$DEFAULT_PREFIXES}"

if [ "${PULL_DATA_ON_START:-false}" = "true" ]; then
    echo "[entrypoint] PULL_DATA_ON_START=true → background mirror s3://${S3_BUCKET}/{${PREFIXES}} → ${DATA_DST}/" >&2
    (
        for prefix in $(echo "$PREFIXES" | tr ',' ' '); do
            # The downloader strips its `--prefix` from each S3 key when
            # forming the local path. So if --prefix=Kadatastr/gold, key
            # `Kadatastr/gold/hex_aggregates/...` becomes `hex_aggregates/...`
            # under --dst — losing the `gold/` level. Restore it by mapping
            # each prefix to its matching local subdirectory:
            #   Kadatastr/gold → data/gold
            #   Kadatastr/models → data/models
            #   Kadatastr/silver → data/silver
            relpath=${prefix#Kadatastr/}
            sub_dst="$DATA_DST/$relpath"
            echo "[entrypoint] pulling $prefix → $sub_dst" >&2
            .venv/bin/python scripts/download_dir_from_s3.py \
                --prefix "$prefix" --dst "$sub_dst" \
                || { echo "[entrypoint] pull FAILED for $prefix" >&2; exit 1; }
        done
        echo "[entrypoint] background data pull done" >&2
    ) &
    echo "[entrypoint] data pull PID=$! running in background" >&2
fi

# scripts/serve.py reads serve_host / serve_port from Settings, which
# Pydantic-settings populates from SERVE_HOST / SERVE_PORT env vars.
exec .venv/bin/python scripts/serve.py

#!/bin/sh
set -e

# S3 pull runs in the background so the app comes up immediately and
# /health responds before the (potentially multi-GB) sync finishes —
# critical for the GHA deploy step's 150s healthcheck window. The
# downloader is idempotent (size-match skip), so a re-run after the
# first cold-start is cheap. Endpoints that need data files will see
# them gradually appear; the app already tolerates a missing parquet
# (404 from /api/hex_aggregates etc.) rather than crashing.
if [ "${PULL_DATA_ON_START:-false}" = "true" ]; then
    echo "[entrypoint] PULL_DATA_ON_START=true → background mirror s3://${S3_BUCKET}/${PULL_DATA_ON_START_PREFIX:-Kadatastr}/ → ${PULL_DATA_ON_START_DST:-data}/" >&2
    (
        .venv/bin/python scripts/download_dir_from_s3.py \
            --prefix "${PULL_DATA_ON_START_PREFIX:-Kadatastr}" \
            --dst "${PULL_DATA_ON_START_DST:-data}" \
            && echo "[entrypoint] background data pull done" >&2 \
            || echo "[entrypoint] background data pull FAILED — see prior log" >&2
    ) &
    echo "[entrypoint] data pull PID=$! running in background" >&2
fi

# scripts/serve.py reads serve_host / serve_port from Settings, which
# Pydantic-settings populates from SERVE_HOST / SERVE_PORT env vars.
exec .venv/bin/python scripts/serve.py

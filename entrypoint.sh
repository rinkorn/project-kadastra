#!/bin/sh
set -e

if [ "${PULL_DATA_ON_START:-false}" = "true" ]; then
    echo "[entrypoint] PULL_DATA_ON_START=true → mirroring s3://${S3_BUCKET}/${PULL_DATA_ON_START_PREFIX:-Kadatastr}/ → ${PULL_DATA_ON_START_DST:-data}/" >&2
    .venv/bin/python scripts/download_dir_from_s3.py \
        --prefix "${PULL_DATA_ON_START_PREFIX:-Kadatastr}" \
        --dst "${PULL_DATA_ON_START_DST:-data}"
fi

# scripts/serve.py reads serve_host / serve_port from Settings, which
# Pydantic-settings populates from SERVE_HOST / SERVE_PORT env vars.
exec .venv/bin/python scripts/serve.py

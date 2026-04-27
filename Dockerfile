FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Geo stack: rasterio/shapely/pyproj/h3 need libgdal/libgeos/libproj at
# runtime; osmium-tool is invoked by scripts/extract_osm_polygons.py.
# curl stays in for the Docker healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        gdal-bin \
        osmium-tool \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS deps

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM deps AS runtime

COPY src/ src/
COPY scripts/ scripts/
# Only the kazan-agglomeration boundary file (3 KB) — the geoBoundaries
# tile under the same dir is 56 MB and isn't read at runtime.
COPY data/raw/regions/kazan-agglomeration.geojson data/raw/regions/
COPY README.md ./
COPY entrypoint.sh ./

RUN uv sync --frozen --no-dev && chmod +x entrypoint.sh

EXPOSE 15777

ENTRYPOINT ["./entrypoint.sh"]

FROM runtime AS scripts

ENTRYPOINT ["uv", "run", "python"]

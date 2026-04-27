from typing import Any

import h3
import polars as pl

from kadastra.etl.haversine import haversine_meters


def compute_road_features(coverage: pl.DataFrame, ways: list[dict[str, Any]]) -> pl.DataFrame:
    resolutions = coverage["resolution"].unique().to_list()
    if len(resolutions) != 1:
        raise ValueError(f"coverage must contain a single resolution, got {resolutions}")
    resolution = int(resolutions[0])

    seg_h3: list[str] = []
    seg_len: list[float] = []
    for way in ways:
        geom = way.get("geometry", []) or []
        for i in range(len(geom) - 1):
            lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
            lat2, lon2 = geom[i + 1]["lat"], geom[i + 1]["lon"]
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            seg_h3.append(h3.latlng_to_cell(mid_lat, mid_lon, resolution))
            seg_len.append(haversine_meters(lat1, lon1, lat2, lon2))

    if seg_h3:
        agg = (
            pl.DataFrame({"h3_index": seg_h3, "road_length_m": seg_len})
            .group_by("h3_index")
            .agg(pl.col("road_length_m").sum())
        )
    else:
        agg = pl.DataFrame(schema={"h3_index": pl.Utf8, "road_length_m": pl.Float64})

    return coverage.join(agg, on="h3_index", how="left").with_columns(pl.col("road_length_m").fill_null(0.0))

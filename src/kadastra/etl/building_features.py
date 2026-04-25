import h3
import polars as pl


def compute_building_features(coverage: pl.DataFrame, buildings: pl.DataFrame) -> pl.DataFrame:
    resolutions = coverage["resolution"].unique().to_list()
    if len(resolutions) != 1:
        raise ValueError(f"coverage must contain a single resolution, got {resolutions}")
    resolution = int(resolutions[0])

    if buildings.is_empty():
        agg = pl.DataFrame(
            schema={
                "h3_index": pl.Utf8,
                "building_count": pl.Int64,
                "building_count_apartments": pl.Int64,
                "building_count_detached": pl.Int64,
                "levels_mean": pl.Float64,
                "flats_total": pl.Int64,
            }
        )
    else:
        b = buildings.with_columns(
            [
                pl.col("levels").cast(pl.Float64, strict=False),
                pl.col("flats").cast(pl.Int64, strict=False),
            ]
        )
        h3_cells = [
            h3.latlng_to_cell(lat, lon, resolution)
            for lat, lon in zip(b["lat"].to_list(), b["lon"].to_list(), strict=True)
        ]
        b = b.with_columns(pl.Series("h3_index", h3_cells))

        agg = b.group_by("h3_index").agg(
            [
                pl.len().alias("building_count"),
                (pl.col("building") == "apartments").sum().alias("building_count_apartments"),
                (pl.col("building") == "detached").sum().alias("building_count_detached"),
                pl.col("levels").mean().alias("levels_mean"),
                pl.col("flats").sum().alias("flats_total"),
            ]
        )

    result = coverage.join(agg, on="h3_index", how="left")
    return result.with_columns(
        [
            pl.col("building_count").fill_null(0),
            pl.col("building_count_apartments").fill_null(0),
            pl.col("building_count_detached").fill_null(0),
            pl.col("flats_total").fill_null(0),
        ]
    )

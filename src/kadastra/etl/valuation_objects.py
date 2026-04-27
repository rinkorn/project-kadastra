import polars as pl

from kadastra.domain.asset_class import AssetClass

_OUTPUT_SCHEMA = {
    "object_id": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "levels": pl.Int64,
    "flats": pl.Int64,
}

_TAG_TO_CLASS: dict[str, str] = {
    "apartments": AssetClass.APARTMENT.value,
    "house": AssetClass.HOUSE.value,
    "detached": AssetClass.HOUSE.value,
    "semidetached_house": AssetClass.HOUSE.value,
    "terrace": AssetClass.HOUSE.value,
    "bungalow": AssetClass.HOUSE.value,
    "retail": AssetClass.COMMERCIAL.value,
    "commercial": AssetClass.COMMERCIAL.value,
    "supermarket": AssetClass.COMMERCIAL.value,
    "kiosk": AssetClass.COMMERCIAL.value,
    "shop": AssetClass.COMMERCIAL.value,
    "office": AssetClass.COMMERCIAL.value,
    "industrial": AssetClass.COMMERCIAL.value,
    "warehouse": AssetClass.COMMERCIAL.value,
}


def assemble_valuation_objects(buildings: pl.DataFrame) -> pl.DataFrame:
    if buildings.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    enriched = buildings.with_columns(
        [
            pl.col("building").str.to_lowercase().alias("_building_norm"),
        ]
    ).with_columns(
        [
            pl.col("_building_norm").replace_strict(_TAG_TO_CLASS, default=None).alias("asset_class"),
            (pl.col("osm_type") + "/" + pl.col("osm_id")).alias("object_id"),
            pl.col("levels").cast(pl.Int64, strict=False).alias("levels"),
            pl.col("flats").cast(pl.Int64, strict=False).alias("flats"),
        ]
    )

    valid = enriched.filter(
        pl.col("asset_class").is_not_null()
        & pl.col("lat").is_between(-90.0, 90.0)
        & pl.col("lon").is_between(-180.0, 180.0)
    )

    return valid.select(
        [
            pl.col("object_id"),
            pl.col("asset_class"),
            pl.col("lat"),
            pl.col("lon"),
            pl.col("levels"),
            pl.col("flats"),
        ]
    )

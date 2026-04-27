"""Read per-hex aggregates for the map UI.

Loads ``data/gold/hex_aggregates/region={REGION}/resolution={R}/data.parquet``,
filters by ``asset_class``, projects ``(h3_index, <feature>)`` and
returns it as ``[{"hex", "value"}, ...]``. The map UI's hex-mode
calls this for whatever (resolution, asset_class, feature) tuple is
selected.

Categorical features (``dominant_*``) are returned as strings; the
map UI colors them by category instead of by gradient.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Numeric metrics → linear/log gradient on the map.
# Order matters: groups close together so the dropdown reads
# «base / building / geo distance / share / count / age».
NUMERIC_FEATURES: tuple[str, ...] = (
    # Base
    "count",
    "median_target_rub_per_m2",
    "median_pred_oof_rub_per_m2",
    "median_residual_rub_per_m2",
    # Building / land descriptors
    "mean_levels",
    "mean_flats",
    "mean_area_m2",
    "mean_year_built",
    "mean_age_years",
    # ADR-0019 distance to nearest geometry — polygonal (dist_to_*)
    "mean_dist_to_water_m",
    "mean_dist_to_park_m",
    "mean_dist_to_forest_m",
    "mean_dist_to_industrial_m",
    "mean_dist_to_cemetery_m",
    "mean_dist_to_landfill_m",
    # ADR-0019 distance — linear
    "mean_dist_to_powerline_m",
    "mean_dist_to_railway_m",
    # ADR-0019 distance — point POIs
    "mean_dist_to_school_m",
    "mean_dist_to_kindergarten_m",
    "mean_dist_to_clinic_m",
    "mean_dist_to_hospital_m",
    "mean_dist_to_pharmacy_m",
    "mean_dist_to_supermarket_m",
    "mean_dist_to_cafe_m",
    "mean_dist_to_restaurant_m",
    "mean_dist_to_bus_stop_m",
    "mean_dist_to_tram_stop_m",
    "mean_dist_to_railway_station_m",
    # Pre-existing transport distances (silver-built, legacy naming)
    "mean_dist_metro_m",
    "mean_dist_entrance_m",
    # Polygonal share-in-buffer at the canonical 500 m radius
    "mean_water_share_500m",
    "mean_park_share_500m",
    "mean_forest_share_500m",
    "mean_industrial_share_500m",
    "mean_cemetery_share_500m",
    # Road density
    "mean_road_length_500m",
    # Zonal counts (pre-existing legacy + ADR-0019 per-POI within_500m)
    "mean_count_stations_1km",
    "mean_count_entrances_500m",
    "mean_count_apartments_500m",
    "mean_count_houses_500m",
    "mean_count_commercial_500m",
    "mean_school_within_500m",
    "mean_kindergarten_within_500m",
    "mean_clinic_within_500m",
    "mean_hospital_within_500m",
    "mean_pharmacy_within_500m",
    "mean_supermarket_within_500m",
    "mean_cafe_within_500m",
    "mean_restaurant_within_500m",
    "mean_bus_stop_within_500m",
    "mean_tram_stop_within_500m",
    "mean_railway_station_within_500m",
)
# Categorical metrics → categorical palette on the map.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "dominant_intra_city_raion",
    "dominant_mun_okrug_name",
    "dominant_settlement_name",
)
ASSET_CLASS_VALUES: tuple[str, ...] = (
    "all",
    "apartment",
    "house",
    "commercial",
    "landplot",
)


class GetHexAggregates:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def execute(
        self,
        region_code: str,
        resolution: int,
        asset_class: str,
        feature: str,
        *,
        model: str = "catboost",
    ) -> list[dict[str, object]]:
        path = (
            self._base_path / f"region={region_code}" / f"resolution={resolution}" / f"model={model}" / "data.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"hex aggregates not built for region={region_code} resolution={resolution}: {path}"
            )
        df = pl.read_parquet(path)
        if feature not in df.columns:
            available = sorted(c for c in df.columns if c not in {"h3_index", "resolution", "asset_class"})
            raise KeyError(f"feature {feature!r} not in hex_aggregates; available: {available}")

        filtered = df.filter(pl.col("asset_class") == asset_class)
        slim = filtered.select(["h3_index", pl.col(feature).alias("value")]).drop_nulls("value")
        return [{"hex": row["h3_index"], "value": row["value"]} for row in slim.iter_rows(named=True)]

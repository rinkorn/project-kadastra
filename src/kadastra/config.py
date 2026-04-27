from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    region_code: str = "RU-KAZAN-AGG"
    region_boundary_path: Path = Path("data/raw/regions/kazan-agglomeration.geojson")
    region_boundary_field: str = "shapeISO"
    coverage_store_path: Path = Path("data/silver/coverage")
    h3_resolutions: list[int] = [7, 8, 9, 10, 11]

    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"
    s3_addressing_style: str = "path"

    feature_store_path: Path = Path("data/silver/features")
    metro_stations_key: str = "Kadatastr/metro/metro_stations.csv"
    metro_entrances_key: str = "Kadatastr/metro/metro_entrances.csv"
    buildings_key: str = "Kadatastr/osm/osm_buildings_kazan_agglomeration.csv"
    roads_key: str = "Kadatastr/tatarstan_major_roads/tatarstan_major_roads.json"

    gold_store_path: Path = Path("data/gold/features")
    gold_feature_sets: list[str] = ["metro", "buildings", "roads"]

    synthetic_target_store_path: Path = Path("data/gold/targets")
    synthetic_target_seed: int = 42

    predictions_store_path: Path = Path("data/gold/predictions")
    valuation_object_store_path: Path = Path("data/gold/valuation_objects")
    object_predictions_store_path: Path = Path("data/gold/object_predictions")
    object_neighbor_radius_m: float = 500.0
    object_road_radius_m: float = 500.0
    poly_area_radii_m: list[int] = [100, 300, 500, 800]
    poly_area_layer_paths: dict[str, str] = {
        "water": "data/raw/osm/kazan-agg-water.geojsonseq",
        "park": "data/raw/osm/kazan-agg-park.geojsonseq",
        "forest": "data/raw/osm/kazan-agg-forest.geojsonseq",
        "industrial": "data/raw/osm/kazan-agg-industrial.geojsonseq",
        "cemetery": "data/raw/osm/kazan-agg-cemetery.geojsonseq",
    }
    # ADR-0019: distance to nearest geometry of each layer. The helper
    # is type-agnostic (Polygon / LineString / Point), so this dict can
    # mix polygonal layers, linear ones and point POIs over time — each
    # entry is just an OSM-extracted GeoJSON-seq file. Separate from
    # poly_area_layer_paths because the layer set may diverge over time
    # (e.g., landfill is meaningful as distance, less so as share).
    geom_distance_layer_paths: dict[str, str] = {
        # Polygonal — distance to nearest polygon edge (or 0 if inside).
        "water": "data/raw/osm/kazan-agg-water.geojsonseq",
        "park": "data/raw/osm/kazan-agg-park.geojsonseq",
        "forest": "data/raw/osm/kazan-agg-forest.geojsonseq",
        "industrial": "data/raw/osm/kazan-agg-industrial.geojsonseq",
        "cemetery": "data/raw/osm/kazan-agg-cemetery.geojsonseq",
        "landfill": "data/raw/osm/kazan-agg-landfill.geojsonseq",
        # Linear — distance to nearest line.
        "powerline": "data/raw/osm/kazan-agg-powerline.geojsonseq",
        "railway": "data/raw/osm/kazan-agg-railway.geojsonseq",
        # Point POIs — distance + (when in zonal_layer_names) counts.
        "school": "data/raw/osm/kazan-agg-school.geojsonseq",
        "kindergarten": "data/raw/osm/kazan-agg-kindergarten.geojsonseq",
        "clinic": "data/raw/osm/kazan-agg-clinic.geojsonseq",
        "hospital": "data/raw/osm/kazan-agg-hospital.geojsonseq",
        "pharmacy": "data/raw/osm/kazan-agg-pharmacy.geojsonseq",
        "supermarket": "data/raw/osm/kazan-agg-supermarket.geojsonseq",
        "cafe": "data/raw/osm/kazan-agg-cafe.geojsonseq",
        "restaurant": "data/raw/osm/kazan-agg-restaurant.geojsonseq",
        "bus_stop": "data/raw/osm/kazan-agg-bus_stop.geojsonseq",
        "tram_stop": "data/raw/osm/kazan-agg-tram_stop.geojsonseq",
        "railway_station": "data/raw/osm/kazan-agg-railway_station.geojsonseq",
    }
    zonal_radii_m: list[int] = [100, 300, 500, 800]
    zonal_layer_names: list[str] = [
        "stations",
        "entrances",
        "apartments",
        "houses",
        "commercial",
        # ADR-0019 point POIs — counts in radii alongside distance.
        # Polygonal/linear layers (water/park/.../powerline) are NOT
        # listed here on purpose: «share in buffer» covers area-density,
        # «distance to nearest» covers proximity, a count would just be
        # a noisier version of share.
        "school",
        "kindergarten",
        "clinic",
        "hospital",
        "pharmacy",
        "supermarket",
        "cafe",
        "restaurant",
        "bus_stop",
        "tram_stop",
        "railway_station",
    ]
    relative_feature_parent_resolutions: list[int] = [7, 8]
    relative_feature_columns: list[str] = [
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
        "road_length_500m",
        "count_apartments_500m",
        "count_houses_500m",
        "count_commercial_500m",
        "levels",
        "flats",
        "area_m2",
        "year_built",
    ]

    # ADR-0020 derived age features. Fixed by config (not
    # datetime.now()) so reruns are deterministic; bump once per year
    # at release time.
    current_year_for_age_features: int = 2026

    nspd_silver_store_path: Path = Path("data/silver/nspd")
    nspd_buildings_raw_dir: Path = Path("data/raw/nspd/buildings-kazan")
    nspd_landplots_raw_dir: Path = Path("data/raw/nspd/landplots-kazan")

    # ADR-0015: GAR-derived silver lookup tables for territorial features.
    gar_lookup_cadnum_index_path: Path = Path("data/silver/gar_lookup/cadnum_index.parquet")
    gar_lookup_mun_lookup_path: Path = Path("data/silver/gar_lookup/mun_lookup.parquet")
    # Pivoted PARAMS lookup keyed on objectid: oktmo_full / okato /
    # postal_index. Built alongside cadnum_index (same XMLs).
    gar_lookup_object_params_path: Path = Path("data/silver/gar_lookup/object_params.parquet")
    # ADR-0015 followup: OSM admin_level=9 polygons for intra_city_raion
    # spatial join (Kazan: 7 raions). Generated by
    # ``scripts/extract_osm_polygons.py --layers raions``.
    osm_raions_geojson_path: Path = Path("data/raw/osm/kazan-agg-raions.geojsonseq")

    road_graph_edges_path: Path = Path("data/silver/road_graph/edges.parquet")
    model_registry_path: Path = Path("data/models")
    # Per-hex aggregates (BuildHexAggregates output): consumed by the
    # map UI's hex-mode in addition to per-object scatter mode.
    hex_aggregates_base_path: Path = Path("data/gold/hex_aggregates")
    hex_aggregates_resolutions: list[int] = [7, 8, 9, 10]
    # ADR-0010 empirical anchor: ЕМИСС/Росстат silver. /api/market_reference
    # reads #61781 (apartments by center city, quarterly) for the inspector.
    emiss_silver_base_path: Path = Path("data/silver/emiss")
    emiss_market_reference_year: int = 2025
    catboost_iterations: int = 500
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    catboost_seed: int = 42
    train_n_splits: int = 5
    train_parent_resolution: int = 7

    # Quartet training perf knobs (S1+S2). When parallel_folds is True,
    # the n_splits per-fold fits are dispatched concurrently via joblib;
    # this collapses landplot wall time from hours to tens of minutes
    # at the cost of memory (n_splits × X). When skip_final_simplifier_fits
    # is True, the EBM/Grey/Naive full-data refits at the end of execute()
    # are skipped — no current consumer reads those *_model.pkl artifacts
    # (inspector reads OOFs only).
    quartet_parallel_folds: bool = True
    quartet_skip_final_simplifier_fits: bool = True

    # Block 5 (ADR-0016) — White Box (EBM) and Grey Box (Decision
    # Tree) hyperparameters. EBM defaults from interpret-ml; Grey
    # depth = 10 keeps the tree shallow enough to be useful as an
    # approximator of the Black Box rather than a competitor.
    ebm_max_bins: int = 256
    ebm_interactions: int = 10
    grey_tree_max_depth: int = 10

    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "kadastra-valuation"

    serve_host: str = "127.0.0.1"
    serve_port: int = 15777

    # Single shared bearer token; when set, BearerAuthMiddleware locks
    # everything except /health and /login/logout. None disables auth
    # entirely (local dev default).
    auth_token: str | None = None

    # Container entrypoint: when true, sync ``s3://{bucket}/{pull_data_on_start_prefix}/``
    # into the local data root before launching uvicorn. Lets the stage VM
    # cold-start without a manual data ship — local dev keeps it false so
    # `data/` stays under user control.
    pull_data_on_start: bool = False
    pull_data_on_start_prefix: str = "Kadatastr"
    pull_data_on_start_dst: Path = Path("data")

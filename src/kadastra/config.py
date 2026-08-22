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

    synthetic_target_seed: int = 42

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
    # ADR-0027: point-POI layers whose walking (graph) distance is computed
    # on the grid. Polygonal/linear layers (water, park, powerline, railway)
    # are excluded — graph distance is point-to-point.
    walk_dist_layer_names: list[str] = [
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

    # ADR-0027: when True, build_object_features joins location distances
    # from the cell grid (Слой 1) instead of computing them per object.
    # Requires the ``geom_distance`` feature set built by
    # BuildCellGeomDistanceFeatures first. Default False keeps the
    # per-object baseline until the A/B is run.
    cell_tsorf_enabled: bool = False
    cell_tsorf_resolution: int = 10
    # ADR-0027 §12: when True (and cell_tsorf_enabled), each object
    # blends the Слой 1 features of every res cell its footprint covers,
    # weighted by area share, instead of inheriting only its centroid
    # cell's values. Default True — the methodologically-correct path
    # (the single-cell fallback exists for A/B ablation). Only objects
    # without a usable polygon geometry fall back to centroid-by-point.
    cell_tsorf_overlap_weighted: bool = True

    # Эпик 001 этап 5: representativeness report (grid ЦОФ distribution
    # vs training-sample distribution) — parquet + markdown summary.
    representativeness_path: Path = Path("data/gold/representativeness")

    # ADR-0029: cell valuation layer — per-cell reference-object price and
    # EBM location score on the Слой 1 anchor grid. Output parquet layout
    # mirrors the valuation-object store (region=/asset_class= partitions).
    cell_valuation_store_path: Path = Path("data/gold/cell_valuation")
    landplot_vri_top_n: int = 5

    nspd_silver_store_path: Path = Path("data/silver/nspd")
    nspd_buildings_raw_dir: Path = Path("data/raw/nspd/buildings-kazan")
    nspd_landplots_raw_dir: Path = Path("data/raw/nspd/landplots-kazan")

    # ADR-0031: listings-as-target ETL (apartment, CIAN MVP-дамп).
    listings_mvp_parquet_path: Path = Path("data/silver/listings-mvp/all.parquet")
    listings_target_store_path: Path = Path("data/silver/listings_target")

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

    # ADR-0025 п. 1: per-region CBD anchor (lat, lon) for dist_to_cbd_m.
    # Kazan: Kremlin / пл. Свободы. Regions without an entry get no column.
    cbd_coords: dict[str, tuple[float, float]] = {
        "RU-KAZAN-AGG": (55.7975, 49.1066),
    }

    # ADR-0025 п. 2: cultural heritage (ОКН). The Минкульт open-data API
    # is unreachable from our network, so the source is an OSM extract
    # (backup on S3). scripts/build_heritage_silver.py parses the raw
    # GeoJSON-seq into heritage_silver_path; BuildObjectFeatures then
    # computes the 4 heritage features inline (the layer is tiny).
    heritage_raw_geojson_path: Path = Path("data/raw/osm/kazan-agg-heritage.geojsonseq")
    heritage_silver_path: Path = Path("data/silver/heritage")

    # ADR-0025 п. 3: ЗОУИТ — bulk dump of НСПД layer 36302 for
    # Tatarstan (177 663 zone polygons, EPSG:3857). The ADR's
    # hypothetical attrs.zouit_intersection field does not exist, so the
    # features come from a real spatial join against the zone polygons.
    # scripts/build_zouit_silver.py parses the page dump into
    # zouit_silver_path (bbox-filtered to the region); scripts/
    # build_zouit_features.py materializes the per-object table that
    # BuildObjectFeatures LEFT JOINs (ADR-0022/0023/0024 pattern).
    zouit_raw_dir: Path = Path("data/raw/nspd/zouit-tatarstan")
    zouit_silver_path: Path = Path("data/silver/zouit_zones")
    zouit_features_path: Path = Path("data/silver/zouit_per_object")

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

    # ADR-0022: macro-territorial EMISS/БД ПМО features. Yearly indicators
    # parsed to silver/emiss/{id}/ and joined wide into
    # silver/macro_oktmo_features by scripts/build_macro_oktmo_features.py.
    # BuildObjectFeatures joins them onto objects when macro_emiss_enabled
    # and the wide table exists for the region. bdmo_* — БД ПМО (tochno.st,
    # CC-BY); 43062 — fedstat уровень безработицы МОТ (субъектный разрез).
    emiss_indicators_yearly: list[str] = [
        "bdmo_8112027",  # численность населения на 1 января
        "bdmo_8213002",  # среднемесячная зарплата работников организаций ГО/МР
        "bdmo_8010001",  # ввод в действие жилых домов, м²
        "bdmo_8401003",  # оборот розничной торговли, тыс. руб
        "bdmo_8006001",  # общая площадь земель МО, га
        "43062",  # уровень безработицы по методологии МОТ (субъект РФ)
    ]
    cadastre_target_year: int = 2024
    macro_emiss_enabled: bool = True
    macro_oktmo_features_path: Path = Path("data/silver/macro_oktmo_features")

    # ADR-0023: topographic features from DEM. Raw Copernicus GLO-30
    # tiles (© DLR) in dem_raw_dir are merged/reprojected/derived into
    # three silver rasters by scripts/build_dem_silver.py;
    # BuildObjectFeatures samples them per object when the silver layers
    # exist for the region (else the step is skipped).
    dem_raw_dir: Path = Path("data/raw/dem")
    dem_silver_base_path: Path = Path("data/silver/dem")
    dem_relief_radius_m: float = 500.0
    dem_features_enabled: bool = True

    # ADR-0024: advanced road-network features. Group 1 (nearest road
    # class + per-class distances) is materialized per object by
    # scripts/build_nearest_road_features.py into
    # road_class_features_path; Group 2 (15-min walking isochrone
    # enrichment) is cached per res-11 hex by
    # scripts/build_isochrone_cache_per_hex.py into isochrone_cache_path.
    # BuildObjectFeatures LEFT JOINs both when the silver tables exist
    # for the region (else the steps are skipped). minor_road_ways_path
    # is the OSM extract of minor-class ways (service/footway/
    # residential/...) with coords_json — the existing road graph and
    # the major-roads raw carry no highway classes (ADR-0024 «Аудит»).
    minor_road_ways_path: Path = Path("data/raw/osm/kazan-agg-minor_road_ways.parquet")
    road_class_features_path: Path = Path("data/silver/road_class_per_object")
    isochrone_cache_path: Path = Path("data/silver/isochrone_cache")
    isochrone_walking_speed_m_per_min: float = 80.0
    isochrone_walking_time_min: int = 15
    isochrone_cache_resolution: int = 11
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
    # is True, the Grey/Naive full-data refits at the end of execute()
    # are skipped — no consumer reads their *_model.pkl artifacts. The
    # EBM (White Box) final fit is ALWAYS kept: the inspector's
    # explanation endpoint loads ebm_model.pkl.
    quartet_parallel_folds: bool = True
    quartet_skip_final_simplifier_fits: bool = True

    # Quartet crash recovery: per-stage checkpoints (per-fold pass1/pass2
    # results + final-fit blobs) under this dir, keyed by a data+params
    # fingerprint. A killed multi-hour run resumes finished stages
    # instead of starting from zero. quartet_resume=False (or
    # ``train_quartet.py --fresh``) ignores/rewrites checkpoints.
    quartet_checkpoint_dir: Path = Path("data/models/_checkpoints")
    quartet_resume: bool = True

    # Block 5 (ADR-0016) — White Box (EBM) and Grey Box (Decision
    # Tree) hyperparameters. EBM defaults from interpret-ml; Grey
    # depth = 10 keeps the tree shallow enough to be useful as an
    # approximator of the Black Box rather than a competitor.
    ebm_max_bins: int = 256
    # 5 pairs, not 10: the A/B on landplot fold-0 (50k rows) showed the
    # first 5 pairs (all dist_metro_m × area/geometry) buy ~0.7pp WAPE
    # and +0.003 Spearman for +24% fit time; pairs beyond 5 are
    # unmeasured, so we don't pay for them.
    ebm_interactions: int = 5
    # Features banned from EBM pair interactions (mains are kept).
    # High-cardinality categoricals explode the interaction tensor
    # (64 bins × 13k vri values ≈ 1M cells — hours per pair on landplot)
    # and overfit on ~12 rows per category anyway.
    ebm_interactions_exclude: list[str] = ["vri", "kadnum_quarter"]
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

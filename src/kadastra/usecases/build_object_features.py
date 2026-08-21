import io
import json
from pathlib import Path
from typing import Any, cast

import polars as pl
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.cell_overlap_weights import compute_overlap_weights
from kadastra.etl.h3_coverage import add_h3_index
from kadastra.etl.load_geometries import load_geojsonseq_geometries, load_geojsonseq_points
from kadastra.etl.object_age_features import compute_object_age_features
from kadastra.etl.object_cbd_distance import compute_cbd_distance
from kadastra.etl.object_dem_features import compute_object_dem_features
from kadastra.etl.object_geom_distance_features import (
    compute_object_geom_distance_features,
)
from kadastra.etl.object_geometry_features import compute_object_geometry_features
from kadastra.etl.object_heritage_features import compute_object_heritage_features
from kadastra.etl.object_isochrone_features import join_isochrone_features
from kadastra.etl.object_macro_features import compute_object_macro_features
from kadastra.etl.object_metro_features import compute_object_metro_features
from kadastra.etl.object_municipality_features import (
    compute_object_municipality_features,
)
from kadastra.etl.object_neighbor_features import compute_object_neighbor_features
from kadastra.etl.object_polygon_features import compute_object_polygon_features
from kadastra.etl.object_road_class_features import join_road_class_features
from kadastra.etl.object_road_features import compute_object_road_features
from kadastra.etl.object_zonal_features import compute_object_zonal_features
from kadastra.etl.relative_features import compute_relative_features
from kadastra.ports.dem_sampler import DemSamplerPort
from kadastra.ports.feature_reader import FeatureReaderPort
from kadastra.ports.raw_data import RawDataPort
from kadastra.ports.road_graph import RoadGraphPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort
from kadastra.usecases.assemble_nspd_valuation_objects import RAW_OBJECT_SCHEMA


class BuildObjectFeatures:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        store: ValuationObjectStorePort,
        raw_data: RawDataPort,
        stations_key: str,
        entrances_key: str,
        roads_key: str,
        neighbor_radius_m: float,
        road_radius_m: float,
        road_graph: RoadGraphPort | None,
        relative_feature_parent_resolutions: list[int],
        relative_feature_columns: list[str],
        zonal_radii_m: list[int],
        zonal_layer_names: list[str],
        poly_area_radii_m: list[int],
        poly_area_layer_paths: dict[str, str],
        geom_distance_layer_paths: dict[str, str] | None = None,
        gar_lookup_cadnum_index_path: Path | None = None,
        gar_lookup_mun_lookup_path: Path | None = None,
        gar_lookup_object_params_path: Path | None = None,
        osm_raions_geojson_path: Path | None = None,
        current_year_for_age_features: int = 2026,
        cell_geom_distance_reader: FeatureReaderPort | None = None,
        cell_polygon_reader: FeatureReaderPort | None = None,
        cell_zonal_reader: FeatureReaderPort | None = None,
        cell_road_reader: FeatureReaderPort | None = None,
        cell_metro_reader: FeatureReaderPort | None = None,
        cell_walk_dist_reader: FeatureReaderPort | None = None,
        cell_tsorf_resolution: int = 10,
        cell_tsorf_overlap_weighted: bool = True,
        macro_oktmo_features_path: Path | None = None,
        cadastre_target_year: int = 2024,
        dem_sampler: DemSamplerPort | None = None,
        road_class_features_path: Path | None = None,
        isochrone_cache_path: Path | None = None,
        isochrone_cache_resolution: int = 11,
        cbd_coords: dict[str, tuple[float, float]] | None = None,
        heritage_silver_path: Path | None = None,
    ) -> None:
        self._reader = reader
        self._store = store
        self._raw_data = raw_data
        self._stations_key = stations_key
        self._entrances_key = entrances_key
        self._roads_key = roads_key
        self._neighbor_radius_m = neighbor_radius_m
        self._road_radius_m = road_radius_m
        self._road_graph = road_graph
        self._relative_feature_parent_resolutions = relative_feature_parent_resolutions
        self._relative_feature_columns = relative_feature_columns
        self._zonal_radii_m = zonal_radii_m
        self._zonal_layer_names = zonal_layer_names
        self._poly_area_radii_m = poly_area_radii_m
        self._poly_area_layer_paths = poly_area_layer_paths
        self._geom_distance_layer_paths = geom_distance_layer_paths or {}
        self._gar_lookup_cadnum_index_path = gar_lookup_cadnum_index_path
        self._gar_lookup_mun_lookup_path = gar_lookup_mun_lookup_path
        self._gar_lookup_object_params_path = gar_lookup_object_params_path
        self._osm_raions_geojson_path = osm_raions_geojson_path
        self._current_year_for_age_features = current_year_for_age_features
        self._cell_geom_distance_reader = cell_geom_distance_reader
        self._cell_polygon_reader = cell_polygon_reader
        self._cell_zonal_reader = cell_zonal_reader
        self._cell_road_reader = cell_road_reader
        self._cell_metro_reader = cell_metro_reader
        self._cell_walk_dist_reader = cell_walk_dist_reader
        self._cell_tsorf_resolution = cell_tsorf_resolution
        self._cell_tsorf_overlap_weighted = cell_tsorf_overlap_weighted
        self._macro_oktmo_features_path = macro_oktmo_features_path
        self._cadastre_target_year = cadastre_target_year
        self._dem_sampler = dem_sampler
        self._road_class_features_path = road_class_features_path
        self._isochrone_cache_path = isochrone_cache_path
        self._isochrone_cache_resolution = isochrone_cache_resolution
        self._cbd_coords = cbd_coords or {}
        self._heritage_silver_path = heritage_silver_path

    def execute(self, region_code: str, asset_classes: list[AssetClass]) -> None:
        stations = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._stations_key)))
        entrances = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._entrances_key)))

        roads_payload = cast(dict[str, Any], json.loads(self._raw_data.read_bytes(self._roads_key)))
        elements = roads_payload.get("elements", []) or []
        ways = [e for e in elements if e.get("type") == "way" and e.get("geometry")]

        slices = {ac: self._reader.load(region_code, ac) for ac in asset_classes}
        non_empty = [df for df in slices.values() if not df.is_empty()]
        combined = pl.concat(non_empty, how="vertical_relaxed") if non_empty else next(iter(slices.values()))
        # Idempotency: the store is read-write (assemble writes raw here,
        # this usecase writes enriched back to the same path). A rerun
        # would otherwise read its own enriched output, and the grid
        # join (left join on h3_index) would then duplicate every
        # locational feature as ``*_right`` — corrupting the A/B (seen
        # in the first grid run: 326 feats, 108 ``_right`` dups).
        # Reduce to the raw schema so feature columns from a prior run
        # are dropped before recomputing.
        raw_cols = [c for c in RAW_OBJECT_SCHEMA if c in combined.columns]
        combined = combined.select(raw_cols)
        # CBD distance (ADR-0025 п. 1). Pure haversine on lat/lon with a
        # per-region constant anchor — recomputed after the RAW reset on
        # every run; skipped entirely for regions without a CBD anchor.
        if region_code in self._cbd_coords:
            cbd_lat, cbd_lon = self._cbd_coords[region_code]
            combined = compute_cbd_distance(combined, cbd_lat=cbd_lat, cbd_lon=cbd_lon)
        # ADR-0027 §12: overlap-weighted assignment of cell ЦОФ to each
        # object — a large footprint blending the features of every res
        # cell it covers, weighted by area share, instead of inheriting
        # only the centroid cell's values. Computed once and reused
        # across all six Слой 1 joins (metro/road/zonal/poly_area/
        # geom_distance/walk_dist). When disabled (or no cell readers
        # wired), falls back to single-cell-by-centroid in the join.
        cell_join_weights = (
            compute_overlap_weights(combined, resolution=self._cell_tsorf_resolution)
            if self._cell_tsorf_overlap_weighted and self._has_any_cell_reader()
            else None
        )

        if self._cell_metro_reader is not None:
            cell_metro = self._cell_metro_reader.load(region_code, self._cell_tsorf_resolution, "metro")
            enriched = self._join_cell_tsorf(combined, cell_metro, cell_join_weights)
        else:
            assert self._road_graph is not None, "road_graph is required when cell_metro_reader is not wired"
            enriched = compute_object_metro_features(combined, stations, entrances, road_graph=self._road_graph)
        if self._cell_road_reader is not None:
            cell_road = self._cell_road_reader.load(region_code, self._cell_tsorf_resolution, "road_density")
            enriched = self._join_cell_tsorf(enriched, cell_road, cell_join_weights)
        else:
            enriched = compute_object_road_features(enriched, ways, radius_m=self._road_radius_m)
        enriched = compute_object_neighbor_features(enriched, radius_m=self._neighbor_radius_m)
        # Zonal density at multiple radii (ADR-0013). Layers are built
        # from the same payload: stations/entrances are the loaded CSVs;
        # apartments/houses/commercial come from `enriched` itself,
        # filtered by asset_class with object_id preserved so the helper
        # excludes self-rows in the count.
        if self._cell_zonal_reader is not None:
            cell_zonal = self._cell_zonal_reader.load(region_code, self._cell_tsorf_resolution, "zonal")
            enriched = self._join_cell_tsorf(enriched, cell_zonal, cell_join_weights)
        else:
            zonal_layers = self._build_zonal_layers(enriched, stations, entrances)
            enriched = compute_object_zonal_features(enriched, layers=zonal_layers, radii_m=self._zonal_radii_m)
        # Poly-area buffer features (ADR-0014 → ADR-0027). When the cell
        # grid store is wired in, share comes from Слой 1 (computed at
        # cell centres); otherwise the per-object fallback applies.
        if self._cell_polygon_reader is not None:
            cell_share = self._cell_polygon_reader.load(region_code, self._cell_tsorf_resolution, "poly_area")
            enriched = self._join_cell_tsorf(enriched, cell_share, cell_join_weights)
        else:
            poly_layers = self._load_poly_area_layers()
            enriched = compute_object_polygon_features(
                enriched,
                polygons_by_layer=poly_layers,
                radii_m=self._poly_area_radii_m,
            )
        # Geom-distance features (ADR-0019). Each entry is a path to an
        # OSM-extracted GeoJSON-seq with arbitrary geometries (Polygon /
        # LineString / Point); the helper handles all three. Missing
        # files → empty layer → null dist column. Share + distance
        # blocks carry orthogonal signals; the model weights them.
        if self._cell_geom_distance_reader is not None:
            cell_dist = self._cell_geom_distance_reader.load(region_code, self._cell_tsorf_resolution, "geom_distance")
            enriched = self._join_cell_tsorf(enriched, cell_dist, cell_join_weights)
        elif self._geom_distance_layer_paths:
            distance_layers = self._load_layer_geometries(self._geom_distance_layer_paths)
            enriched = compute_object_geom_distance_features(enriched, geometries_by_layer=distance_layers)
        # ADR-0027: walking distance to point POIs — grid-only (graph is too
        # expensive per-object; methodology §17 «всё на сетке один раз»).
        if self._cell_walk_dist_reader is not None:
            cell_walk_dist = self._cell_walk_dist_reader.load(region_code, self._cell_tsorf_resolution, "walk_dist")
            enriched = self._join_cell_tsorf(enriched, cell_walk_dist, cell_join_weights)
        # Territorial / municipality features (ADR-0015). ГАР primary
        # via cad_num→objectid→mun_lookup; NSPD readable_address parse
        # fallback for the ~55–75 % unmatched rows. Skip if either
        # lookup is missing (treat block 4 as opt-in).
        if (
            self._gar_lookup_cadnum_index_path is not None
            and self._gar_lookup_mun_lookup_path is not None
            and self._gar_lookup_cadnum_index_path.is_file()
            and self._gar_lookup_mun_lookup_path.is_file()
        ):
            cadnum_ix = pl.read_parquet(self._gar_lookup_cadnum_index_path)
            mun_lookup = pl.read_parquet(self._gar_lookup_mun_lookup_path)
            # OSM admin_level=9 polygons: primary source for
            # intra_city_raion (address regex remains as fallback for
            # objects outside the polygon set or in regions where OSM
            # raions are not extracted yet).
            raion_polygons = self._load_intra_raion_polygons()
            # Settlement-level OKTMO + ОКАТО + postal_index from
            # AS_*_PARAMS pivoted lookup. Optional: if missing, the
            # municipality function emits null cells for those columns.
            object_params = self._load_object_params_lookup()
            enriched = compute_object_municipality_features(
                enriched,
                cadnum_index=cadnum_ix,
                mun_lookup=mun_lookup,
                object_params=object_params,
                intra_raion_polygons=raion_polygons,
            )
        # Macro-territorial EMISS features (ADR-0022). Left join on the
        # 8-digit municipal OKTMO prefix of GAR-derived ``oktmo_full``;
        # runs after the municipality block which produces that column.
        # Opt-in: skipped when no macro table path is wired or the wide
        # table was not built for the region yet.
        if self._macro_oktmo_features_path is not None:
            macro_table = self._load_macro_oktmo_features(region_code)
            if macro_table is not None:
                enriched = compute_object_macro_features(
                    enriched,
                    macro_table=macro_table,
                    target_year=self._cadastre_target_year,
                )
        # Object geometry features (ADR-0018). Reads polygon_wkt_3857
        # passthrough from ADR-0017 and derives 7 shape descriptors.
        # KeyError if the column is missing — that is an upstream
        # contract violation (silver→gold).
        enriched = compute_object_geometry_features(enriched)
        # Object age + era features (ADR-0020). Pure feature engineering
        # over the existing year_built column; current_year is fixed by
        # config so output stays deterministic across reruns.
        enriched = compute_object_age_features(
            enriched,
            current_year=self._current_year_for_age_features,
        )
        # Topographic DEM features (ADR-0023). Samples the silver DEM
        # rasters (built by scripts/build_dem_silver.py from GLO-30 raw)
        # at each object's (lat, lon). The columns are derived here,
        # AFTER the RAW_OBJECT_SCHEMA reset above, so they are not part
        # of the raw schema — a rerun recomputes them from the rasters.
        # Opt-in: skipped when no dem_sampler is wired (composition root
        # passes None when the silver layers are missing or the flag is
        # off).
        if self._dem_sampler is not None:
            enriched = compute_object_dem_features(enriched, dem_sampler=self._dem_sampler)
        # Advanced road-network features (ADR-0024). Both blocks are
        # LEFT JOINs from silver tables materialized by dedicated scripts
        # (build_nearest_road_features.py per object;
        # build_isochrone_cache_per_hex.py per res-11 hex). They run
        # after the RAW_OBJECT_SCHEMA reset, so a rerun recomputes them
        # from silver — the ADR-0022/0023 pattern. Opt-in: each join is
        # skipped when its silver partition does not exist for the
        # region.
        if self._road_class_features_path is not None:
            road_features = self._load_road_class_features(region_code)
            if road_features is not None:
                enriched = join_road_class_features(enriched, road_features)
        if self._isochrone_cache_path is not None:
            iso_cache = self._load_isochrone_cache(region_code)
            if iso_cache is not None:
                enriched = join_isochrone_features(
                    enriched,
                    iso_cache,
                    resolution=self._isochrone_cache_resolution,
                )
        # Heritage / ОКН features (ADR-0025 п. 2). Computed inline from
        # the silver ОКН layer (built by scripts/build_heritage_silver.py
        # from the OSM extract — Минкульт open-data API is unreachable
        # from our network). The layer is tiny (~200 objects), so no
        # per-object materialization is needed. Opt-in: skipped when the
        # silver partition does not exist for the region.
        if self._heritage_silver_path is not None:
            heritage = self._load_heritage_objects(region_code)
            if heritage is not None:
                enriched = compute_object_heritage_features(enriched, heritage=heritage)
        # Filter feature_columns to those present (allows configuring a
        # superset in Settings — missing ones are simply skipped, not
        # errors, so per-class slices with different schemas don't crash).
        present_relative_columns = [c for c in self._relative_feature_columns if c in enriched.columns]
        enriched = compute_relative_features(
            enriched,
            parent_resolutions=self._relative_feature_parent_resolutions,
            feature_columns=present_relative_columns,
        )

        for asset_class in asset_classes:
            slice_df = enriched.filter(pl.col("asset_class") == asset_class.value)
            self._store.save(region_code, asset_class, slice_df)

    def _load_heritage_objects(self, region_code: str) -> pl.DataFrame | None:
        """Load the silver ОКН layer for the region.

        Returns ``None`` when the partition does not exist, so the
        pipeline skips the heritage block entirely (same opt-in contract
        as the road-class loader).
        """
        base = self._heritage_silver_path
        if base is None:
            return None
        path = base / f"region={region_code}" / "data.parquet"
        if not path.is_file():
            return None
        return pl.read_parquet(path)

    def _load_road_class_features(self, region_code: str) -> pl.DataFrame | None:
        """Load the silver road-class-per-object table for the region.

        Returns ``None`` when the partition does not exist, so the
        pipeline skips the join entirely (same opt-in contract as the
        macro-OKTMO loader).
        """
        base = self._road_class_features_path
        if base is None:
            return None
        path = base / f"region={region_code}" / "data.parquet"
        if not path.is_file():
            return None
        return pl.read_parquet(path)

    def _load_isochrone_cache(self, region_code: str) -> pl.DataFrame | None:
        """Load the per-hex isochrone cache for the region/resolution.

        Returns ``None`` when the partition does not exist, so the
        pipeline skips the join entirely.
        """
        base = self._isochrone_cache_path
        if base is None:
            return None
        path = base / f"region={region_code}" / f"h3_p={self._isochrone_cache_resolution}" / "data.parquet"
        if not path.is_file():
            return None
        return pl.read_parquet(path)

    def _load_macro_oktmo_features(self, region_code: str) -> pl.DataFrame | None:
        """Load the wide per-(oktmo, year) EMISS macro table for the region.

        Reads every ``region={code}/year=*/data.parquet`` partition
        (ADR-0022 stores one wide row per oktmo per year; the year
        alignment to ``cadastre_target_year`` happens inside
        ``compute_object_macro_features``). Returns ``None`` when no
        partition exists, so the pipeline skips the join entirely.
        """
        base = self._macro_oktmo_features_path
        if base is None:
            return None
        paths = sorted(base.glob(f"region={region_code}/year=*/data.parquet"))
        if not paths:
            return None
        return pl.concat([pl.read_parquet(p) for p in paths])

    def _load_object_params_lookup(self) -> pl.DataFrame | None:
        """Load the per-OBJECTID PARAMS pivot if configured and present.

        Returns ``None`` (rather than an empty frame) when the lookup
        is missing, so downstream knows to skip the join entirely
        instead of producing all-null columns.
        """
        path = self._gar_lookup_object_params_path
        if path is None or not path.is_file():
            return None
        return pl.read_parquet(path)

    def _load_intra_raion_polygons(self) -> list[tuple[str, BaseGeometry]]:
        """Load (short_name, geometry) pairs from a GeoJSON-seq file.

        Each feature is expected to be a Polygon/MultiPolygon with at
        least a ``name`` property (e.g. "Советский район"). Returns an
        empty list if the file is not configured or not present, which
        downgrades the spatial-join step to a no-op (address regex
        still runs as fallback).
        """
        path = self._osm_raions_geojson_path
        if path is None or not path.is_file():
            return []
        named: list[tuple[str, BaseGeometry]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("\x1e"):
                    line = line.lstrip("\x1e").strip()
                    if not line:
                        continue
                feature = json.loads(line)
                geom = feature.get("geometry")
                props = feature.get("properties") or {}
                if geom is None:
                    continue
                full_name = (props.get("name") or "").strip()
                if not full_name:
                    continue
                # Drop trailing " район" (or " р-н") so the value
                # matches the short form produced by the address regex
                # path ("Советский район" → "Советский").
                short = full_name
                for suffix in (" район", " р-н"):
                    if short.endswith(suffix):
                        short = short[: -len(suffix)].strip()
                        break
                named.append((short, shape(geom)))
        return named

    def _load_poly_area_layers(self) -> dict[str, list[BaseGeometry]]:
        return self._load_layer_geometries(self._poly_area_layer_paths)

    def _load_layer_geometries(self, paths: dict[str, str]) -> dict[str, list[BaseGeometry]]:
        return load_geojsonseq_geometries(paths)

    def _has_any_cell_reader(self) -> bool:
        return any(
            reader is not None
            for reader in (
                self._cell_metro_reader,
                self._cell_road_reader,
                self._cell_zonal_reader,
                self._cell_polygon_reader,
                self._cell_geom_distance_reader,
                self._cell_walk_dist_reader,
            )
        )

    def _join_cell_tsorf(
        self,
        objects: pl.DataFrame,
        cell_tsorf: pl.DataFrame,
        weights_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Join objects to cell-level ЦОФ (ADR-0027, §12 overlap).

        Two modes:

        - **overlap-weighted** (``weights_df`` given): each object blends
          the features of every res cell its footprint covers, weighted
          by area share (``compute_overlap_weights``). A long
          ``(object_id, h3_index, weight)`` frame is joined to the cell
          feature set on ``h3_index``, then features are reduced per
          object via ``Σ value·weight`` (count columns get the same
          weighted mean — appropriate for densities; integer casts are
          re-applied). Objects with no coverage get nulls.
        - **single-cell** (``weights_df`` None): the legacy centroid-cell
          left-join — backward-compatible path used when overlap
          weighting is disabled or no cell reader is wired.
        """
        right = cell_tsorf.drop("resolution") if "resolution" in cell_tsorf.columns else cell_tsorf
        feature_cols = [c for c in right.columns if c != "h3_index"]
        if not feature_cols:
            return objects

        if weights_df is None:
            with_index = add_h3_index(objects, resolution=self._cell_tsorf_resolution)
            return with_index.join(right, on="h3_index", how="left").drop("h3_index")

        # Overlap-weighted: long (object_id, h3_index, weight) × cell features.
        joined = weights_df.join(right, on="h3_index", how="inner")
        if joined.is_empty():
            # No cell coverage at all — emit null feature columns.
            return objects.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in feature_cols])
        # Σ value·weight per object, keyed on object_id. Casts happen at
        # the end so count columns (stored Int64) re-become integers
        # only after the weighted mean; the intermediate is Float64.
        agg_exprs = [(pl.col(c).cast(pl.Float64) * pl.col("weight")).sum().alias(c) for c in feature_cols]
        weighted = joined.group_by("object_id").agg(agg_exprs)
        # Preserve original column dtypes (counts stay Int64).
        cast_exprs = []
        for c in feature_cols:
            src = right.schema.get(c, pl.Float64)
            if src != pl.Float64:
                cast_exprs.append(pl.col(c).cast(src))
            else:
                cast_exprs.append(pl.col(c))
        if cast_exprs:
            weighted = weighted.with_columns(cast_exprs)
        return objects.join(weighted, on="object_id", how="left")

    def _build_zonal_layers(
        self,
        enriched: pl.DataFrame,
        stations: pl.DataFrame,
        entrances: pl.DataFrame,
    ) -> dict[str, pl.DataFrame]:
        # Layer name → AssetClass for self-class slices.
        class_layer_map = {
            "apartments": AssetClass.APARTMENT.value,
            "houses": AssetClass.HOUSE.value,
            "commercial": AssetClass.COMMERCIAL.value,
            "landplots": AssetClass.LANDPLOT.value,
        }
        layers: dict[str, pl.DataFrame] = {}
        for name in self._zonal_layer_names:
            if name == "stations":
                layers[name] = stations.select(["lat", "lon"])
            elif name == "entrances":
                layers[name] = entrances.select(["lat", "lon"])
            elif name in class_layer_map:
                # Self-exclusion in compute_object_zonal_features kicks in
                # via object_id so the object's own row never counts.
                layers[name] = enriched.filter(pl.col("asset_class") == class_layer_map[name]).select(
                    ["object_id", "lat", "lon"]
                )
            elif name in self._geom_distance_layer_paths:
                # ADR-0019 part 4: OSM-extracted POI layer (school,
                # bus_stop, ...). Reuse the same GeoJSON-seq file that
                # geom-distance reads, centroiding non-Point features
                # so the count helper sees lat/lon points uniformly.
                layers[name] = self._load_zonal_poi_layer(self._geom_distance_layer_paths[name])
        return layers

    def _load_zonal_poi_layer(self, path_str: str) -> pl.DataFrame:
        return load_geojsonseq_points(path_str)

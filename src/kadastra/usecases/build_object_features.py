import io
import json
from pathlib import Path
from typing import Any, cast

import polars as pl
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.object_metro_features import compute_object_metro_features
from kadastra.etl.object_municipality_features import (
    compute_object_municipality_features,
)
from kadastra.etl.object_neighbor_features import compute_object_neighbor_features
from kadastra.etl.object_polygon_features import compute_object_polygon_features
from kadastra.etl.object_road_features import compute_object_road_features
from kadastra.etl.object_zonal_features import compute_object_zonal_features
from kadastra.etl.relative_features import compute_relative_features
from kadastra.ports.raw_data import RawDataPort
from kadastra.ports.road_graph import RoadGraphPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort


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
        road_graph: RoadGraphPort,
        relative_feature_parent_resolutions: list[int],
        relative_feature_columns: list[str],
        zonal_radii_m: list[int],
        zonal_layer_names: list[str],
        poly_area_radii_m: list[int],
        poly_area_layer_paths: dict[str, str],
        gar_lookup_cadnum_index_path: Path | None = None,
        gar_lookup_mun_lookup_path: Path | None = None,
        gar_lookup_object_params_path: Path | None = None,
        osm_raions_geojson_path: Path | None = None,
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
        self._gar_lookup_cadnum_index_path = gar_lookup_cadnum_index_path
        self._gar_lookup_mun_lookup_path = gar_lookup_mun_lookup_path
        self._gar_lookup_object_params_path = gar_lookup_object_params_path
        self._osm_raions_geojson_path = osm_raions_geojson_path

    def execute(self, region_code: str, asset_classes: list[AssetClass]) -> None:
        stations = pl.read_csv(
            io.BytesIO(self._raw_data.read_bytes(self._stations_key))
        )
        entrances = pl.read_csv(
            io.BytesIO(self._raw_data.read_bytes(self._entrances_key))
        )

        roads_payload = cast(
            dict[str, Any], json.loads(self._raw_data.read_bytes(self._roads_key))
        )
        elements = roads_payload.get("elements", []) or []
        ways = [e for e in elements if e.get("type") == "way" and e.get("geometry")]

        slices = {ac: self._reader.load(region_code, ac) for ac in asset_classes}
        non_empty = [df for df in slices.values() if not df.is_empty()]
        combined = (
            pl.concat(non_empty, how="vertical_relaxed")
            if non_empty
            else next(iter(slices.values()))
        )

        enriched = compute_object_metro_features(
            combined, stations, entrances, road_graph=self._road_graph
        )
        enriched = compute_object_road_features(
            enriched, ways, radius_m=self._road_radius_m
        )
        enriched = compute_object_neighbor_features(
            enriched, radius_m=self._neighbor_radius_m
        )
        # Zonal density at multiple radii (ADR-0013). Layers are built
        # from the same payload: stations/entrances are the loaded CSVs;
        # apartments/houses/commercial come from `enriched` itself,
        # filtered by asset_class with object_id preserved so the helper
        # excludes self-rows in the count.
        zonal_layers = self._build_zonal_layers(enriched, stations, entrances)
        enriched = compute_object_zonal_features(
            enriched, layers=zonal_layers, radii_m=self._zonal_radii_m
        )
        # Poly-area buffer features (ADR-0014). Layers are loaded once
        # from disk; missing files yield empty layers (zero share).
        poly_layers = self._load_poly_area_layers()
        enriched = compute_object_polygon_features(
            enriched,
            polygons_by_layer=poly_layers,
            radii_m=self._poly_area_radii_m,
        )
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
        # Filter feature_columns to those present (allows configuring a
        # superset in Settings — missing ones are simply skipped, not
        # errors, so per-class slices with different schemas don't crash).
        present_relative_columns = [
            c for c in self._relative_feature_columns if c in enriched.columns
        ]
        enriched = compute_relative_features(
            enriched,
            parent_resolutions=self._relative_feature_parent_resolutions,
            feature_columns=present_relative_columns,
        )

        for asset_class in asset_classes:
            slice_df = enriched.filter(pl.col("asset_class") == asset_class.value)
            self._store.save(region_code, asset_class, slice_df)

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
        layers: dict[str, list[BaseGeometry]] = {}
        for name, path_str in self._poly_area_layer_paths.items():
            path = Path(path_str)
            if not path.is_file():
                # Missing extraction — emit as empty layer; downstream will
                # produce zero-share columns. Keeps the pipeline composable
                # while OSM extractions are still being run.
                layers[name] = []
                continue
            polys: list[BaseGeometry] = []
            with path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("\x1e"):
                        # GeoJSON-seq lines may be RS-prefixed; skip if so.
                        line = line.lstrip("\x1e").strip()
                        if not line:
                            continue
                    feature = json.loads(line)
                    geom = feature.get("geometry")
                    if geom is None:
                        continue
                    polys.append(shape(geom))
            layers[name] = polys
        return layers

    def _build_zonal_layers(
        self,
        enriched: pl.DataFrame,
        stations: pl.DataFrame,
        entrances: pl.DataFrame,
    ) -> dict[str, pl.DataFrame]:
        # Layer name → AssetClass for self-class slices. Names that aren't
        # in this map are treated as external point layers.
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
                layers[name] = enriched.filter(
                    pl.col("asset_class") == class_layer_map[name]
                ).select(["object_id", "lat", "lon"])
        return layers

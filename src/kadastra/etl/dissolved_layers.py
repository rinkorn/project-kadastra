"""Per-run cache of dissolved (unioned) layer geometries.

Both polygon-share (``compute_object_polygon_features``) and
geom-distance (``compute_object_geom_distance_features``) blocks start
from the same expensive step per layer: project every WGS84 geometry to
EPSG:32639 (UTM-39N) and dissolve the layer with ``unary_union``. The
dissolved geometry depends only on the layer, not on the objects being
enriched, so within one pipeline run it must be computed once and
reused across feature blocks and asset-class slices.

``DissolvedLayers`` memoizes that step, keyed by the identity of the
layer's geometries list. The cache holds a strong reference to each
keyed list, so ``id()`` cannot be recycled while an entry is alive.
Consumers that don't share a cache (``None``) get a fresh instance per
call — behaviour identical to the previous inline pipeline.
"""

from __future__ import annotations

from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# UTM zone 39N — same projection used by the agglomeration boundary
# build script; minimal area distortion (≤ 0.1 %) at Kazan latitude.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


class DissolvedLayers:
    """Lazy memoizer for ``project → unary_union`` per layer."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[list[BaseGeometry], BaseGeometry]] = {}

    def dissolved(self, geometries: list[BaseGeometry]) -> BaseGeometry:
        """Return the dissolved EPSG:32639 geometry of the layer.

        The same list object always yields the same cached geometry
        without recomputing the union.
        """
        key = id(geometries)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is geometries:
            return hit[1]
        merged = unary_union([_project_lonlat(g) for g in geometries])
        self._cache[key] = (geometries, merged)
        return merged

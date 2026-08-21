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
"""

from __future__ import annotations

from shapely.geometry.base import BaseGeometry


class DissolvedLayers:
    """Lazy memoizer for ``project → unary_union`` per layer."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[list[BaseGeometry], BaseGeometry]] = {}

    def dissolved(self, geometries: list[BaseGeometry]) -> BaseGeometry:
        """Return the dissolved EPSG:32639 geometry of the layer.

        The same list object always yields the same cached geometry
        without recomputing the union.
        """
        raise NotImplementedError

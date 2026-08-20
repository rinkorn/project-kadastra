"""DEM sampler port — per-point topographic lookups (ADR-0023).

Silver DEM derivatives (elevation / slope / relative relief rasters)
are preprocessed once per region by ``BuildDemSilver``; use cases only
need point sampling at object coordinates. This port hides the raster
storage (rasterio adapter) so the ETL step works against fakes in
unit tests.

All coordinates are WGS84 (lat, lon). A sample outside the raster
extent or on a nodata pixel returns ``None`` — a legitimate state for
objects near the region boundary, not an error.
"""

from __future__ import annotations

from typing import Protocol


class DemSamplerPort(Protocol):
    def sample_elevation(self, *, lat: float, lon: float) -> float | None:
        """Absolute elevation in metres at (lat, lon), or None."""
        ...

    def sample_slope_deg(self, *, lat: float, lon: float) -> float | None:
        """Local slope in degrees at (lat, lon), or None."""
        ...

    def sample_relative_relief(self, *, lat: float, lon: float) -> float | None:
        """Relative relief (max − min elevation) in metres within the
        radius fixed at silver build time (``dem_relief_radius_m``),
        or None.

        Note: the ADR-0023 draft signature carried a ``radius_m``
        argument; the shipped contract drops it because the relief grid
        is precomputed at a single configured radius — sampling a
        different radius would silently read the wrong layer.
        """
        ...

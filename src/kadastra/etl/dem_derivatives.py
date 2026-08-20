"""Pure DEM derivative computations on a metric grid (ADR-0023).

Both functions assume the input elevation array is on a projected
(metre) grid — ``BuildDemSilver`` reprojects the merged raw tiles to
UTM first, per the repo rule «UTM-перепроекция для площадных
операций». NaN marks nodata; NaN propagates through the gradient and
is ignored by the (NaN-aware) rolling window, so holes in the source
DEM become holes in the derivatives instead of fake values.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def compute_slope_deg(
    elevation: np.ndarray,
    *,
    pixel_width_m: float,
    pixel_height_m: float,
) -> np.ndarray:
    """Local slope in degrees from central differences of elevation."""
    grad_y, grad_x = np.gradient(elevation, pixel_height_m, pixel_width_m)
    return np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)


def compute_relative_relief(
    elevation: np.ndarray,
    *,
    radius_m: float,
    pixel_width_m: float,
    pixel_height_m: float,
) -> np.ndarray:
    """max − min elevation within ``radius_m`` of each pixel, in metres.

    Separable rolling window: a (2r+1)² max/min is computed as a
    vertical pass followed by a horizontal pass, which keeps the one-off
    build at O(H·W·r) instead of O(H·W·r²). Edge pixels use the
    truncated window (NaN padding is ignored by nanmax/nanmin).
    """
    radius_rows = round(radius_m / pixel_height_m)
    radius_cols = round(radius_m / pixel_width_m)
    relief_max = _rolling_reduce(elevation, radius_rows, radius_cols, np.nanmax)
    relief_min = _rolling_reduce(elevation, radius_rows, radius_cols, np.nanmin)
    return (relief_max - relief_min).astype(np.float32)


def _rolling_reduce(
    arr: np.ndarray,
    radius_rows: int,
    radius_cols: int,
    func: Callable[..., np.ndarray],
) -> np.ndarray:
    out = arr.astype(np.float64, copy=False)
    for axis, radius in ((0, radius_rows), (1, radius_cols)):
        if radius <= 0:
            continue
        width = 2 * radius + 1
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="constant", constant_values=np.nan)
        view = sliding_window_view(padded, width, axis=axis)
        with warnings.catch_warnings():
            # All-NaN windows (outside the DEM coverage) → NaN, which is
            # exactly what we want; silence the RuntimeWarning.
            warnings.simplefilter("ignore", RuntimeWarning)
            out = func(view, axis=-1)
    return out

"""Per-object distance to the Central Business District (ADR-0025, п. 1).

CBD is a per-region manual constant (Kazan: Kremlin / пл. Свободы).
Pure haversine on (lat, lon) — no projection needed for a radial
distance signal (dev-rules: «haversine только для радиальных
дистанций»).
"""

from __future__ import annotations

import polars as pl


def compute_cbd_distance(
    objects: pl.DataFrame,
    *,
    cbd_lat: float,
    cbd_lon: float,
) -> pl.DataFrame:
    """Append ``dist_to_cbd_m`` — haversine metres from each object to the CBD.

    Objects with null coordinates get a null distance; an empty frame
    gets an empty Float64 column so the schema stays stable.
    """
    raise NotImplementedError

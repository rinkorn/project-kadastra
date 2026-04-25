from typing import Any

import polars as pl


def compute_road_features(coverage: pl.DataFrame, ways: list[dict[str, Any]]) -> pl.DataFrame:
    raise NotImplementedError

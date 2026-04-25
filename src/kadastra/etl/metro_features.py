import polars as pl


def compute_metro_features(
    coverage: pl.DataFrame,
    stations: pl.DataFrame,
    entrances: pl.DataFrame,
) -> pl.DataFrame:
    raise NotImplementedError

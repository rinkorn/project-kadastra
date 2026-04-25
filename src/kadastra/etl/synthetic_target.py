import polars as pl


def compute_synthetic_target(gold: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    raise NotImplementedError

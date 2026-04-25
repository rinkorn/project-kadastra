def spatial_kfold_split(
    h3_indices: list[str],
    *,
    n_splits: int,
    parent_resolution: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    raise NotImplementedError

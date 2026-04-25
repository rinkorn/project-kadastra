import random

import h3


def spatial_kfold_split(
    h3_indices: list[str],
    *,
    n_splits: int,
    parent_resolution: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    parents = [h3.cell_to_parent(c, parent_resolution) for c in h3_indices]
    unique_parents = sorted(set(parents))
    if n_splits > len(unique_parents):
        raise ValueError(
            f"n_splits={n_splits} exceeds number of unique parents ({len(unique_parents)}) "
            f"at resolution={parent_resolution}"
        )

    shuffled = list(unique_parents)
    random.Random(seed).shuffle(shuffled)

    parent_to_fold: dict[str, int] = {p: i % n_splits for i, p in enumerate(shuffled)}
    index_fold = [parent_to_fold[p] for p in parents]

    folds: list[tuple[list[int], list[int]]] = []
    for k in range(n_splits):
        train_idx = [i for i, f in enumerate(index_fold) if f != k]
        val_idx = [i for i, f in enumerate(index_fold) if f == k]
        folds.append((train_idx, val_idx))
    return folds

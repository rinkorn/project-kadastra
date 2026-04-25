import h3
import pytest

from kadastra.ml.spatial_kfold import spatial_kfold_split

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _cells_near_kazan(n: int, res: int) -> list[str]:
    return [h3.latlng_to_cell(KAZAN_LAT + 0.005 * i, KAZAN_LON, res) for i in range(n)]


def test_returns_requested_number_of_folds() -> None:
    cells = _cells_near_kazan(60, 8)

    folds = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=42)

    assert len(folds) == 5


def test_each_fold_covers_all_indices_without_overlap() -> None:
    cells = _cells_near_kazan(60, 8)

    folds = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=42)

    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == set(range(len(cells)))


def test_val_parents_disjoint_from_train_parents() -> None:
    cells = _cells_near_kazan(60, 8)

    folds = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=42)

    parents = [h3.cell_to_parent(c, 6) for c in cells]
    for train_idx, val_idx in folds:
        train_parents = {parents[i] for i in train_idx}
        val_parents = {parents[i] for i in val_idx}
        assert train_parents.isdisjoint(val_parents)


def test_every_index_appears_in_exactly_one_val_fold() -> None:
    cells = _cells_near_kazan(60, 8)

    folds = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=42)

    appearances: dict[int, int] = {}
    for _, val_idx in folds:
        for i in val_idx:
            appearances[i] = appearances.get(i, 0) + 1
    assert all(count == 1 for count in appearances.values())
    assert set(appearances.keys()) == set(range(len(cells)))


def test_deterministic_given_seed() -> None:
    cells = _cells_near_kazan(60, 8)

    f1 = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=42)
    f2 = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=42)

    assert f1 == f2


def test_different_seeds_produce_different_partitions() -> None:
    cells = _cells_near_kazan(60, 8)

    f1 = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=1)
    f2 = spatial_kfold_split(cells, n_splits=5, parent_resolution=6, seed=2)

    assert f1 != f2


def test_raises_when_n_splits_exceeds_unique_parents() -> None:
    # All cells under one parent → cannot split into 5
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    single_parent_cells = [cell] * 5

    with pytest.raises(ValueError, match="unique parents"):
        spatial_kfold_split(single_parent_cells, n_splits=5, parent_resolution=6, seed=42)

import h3
import polars as pl
import pytest

from kadastra.etl.synthetic_target import compute_synthetic_target

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _gold(
    h3_indices: list[str],
    resolution: int,
    *,
    building_count: list[int] | None = None,
    count_stations_1km: list[int] | None = None,
) -> pl.DataFrame:
    n = len(h3_indices)
    return pl.DataFrame(
        {
            "h3_index": h3_indices,
            "resolution": [resolution] * n,
            "building_count": building_count if building_count is not None else [0] * n,
            "count_stations_1km": count_stations_1km if count_stations_1km is not None else [0] * n,
        }
    )


def test_compute_synthetic_target_adds_expected_columns() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold = _gold([cell], 8)

    result = compute_synthetic_target(gold, seed=42)

    assert "kazan_distance_km" in result.columns
    assert "synthetic_target_rub_per_m2" in result.columns


def test_compute_synthetic_target_kazan_distance_is_small_at_kazan_center() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold = _gold([cell], 8)

    result = compute_synthetic_target(gold, seed=42)

    # Center of Kazan: hex center is within hex size (~0.3 km for res=8)
    assert result["kazan_distance_km"][0] < 1.0


def test_compute_synthetic_target_distance_grows_for_far_hex() -> None:
    near = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    far = h3.latlng_to_cell(53.2, 50.0, 8)  # ~300 km south of Kazan
    gold = _gold([near, far], 8)

    result = compute_synthetic_target(gold, seed=42)

    by_hex = {row["h3_index"]: row["kazan_distance_km"] for row in result.iter_rows(named=True)}
    assert by_hex[near] < 1.0
    assert by_hex[far] > 200.0


def test_compute_synthetic_target_target_is_non_negative() -> None:
    cells = [
        h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8),
        h3.latlng_to_cell(56.5, 52.0, 8),
        h3.latlng_to_cell(54.0, 48.0, 8),
    ]
    gold = _gold(cells, 8)

    result = compute_synthetic_target(gold, seed=1)

    targets = result["synthetic_target_rub_per_m2"].to_list()
    assert all(t >= 0 for t in targets)


def test_compute_synthetic_target_is_deterministic_given_seed() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold = _gold([cell], 8)

    r1 = compute_synthetic_target(gold, seed=42)
    r2 = compute_synthetic_target(gold, seed=42)

    assert r1["synthetic_target_rub_per_m2"].to_list() == r2["synthetic_target_rub_per_m2"].to_list()


def test_compute_synthetic_target_differs_for_different_seeds() -> None:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(20)]
    gold = _gold(cells, 8)

    r1 = compute_synthetic_target(gold, seed=1)
    r2 = compute_synthetic_target(gold, seed=2)

    assert r1["synthetic_target_rub_per_m2"].to_list() != r2["synthetic_target_rub_per_m2"].to_list()


def test_near_kazan_has_higher_mean_target_than_far() -> None:
    # Small cluster near Kazan + cluster far away
    near_cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(50)]
    far_cells = [h3.latlng_to_cell(53.0 + 0.01 * i, 50.0, 8) for i in range(50)]
    gold = _gold(near_cells + far_cells, 8)

    result = compute_synthetic_target(gold, seed=42)

    by_hex = {row["h3_index"]: row["synthetic_target_rub_per_m2"] for row in result.iter_rows(named=True)}
    near_mean = sum(by_hex[c] for c in near_cells) / len(near_cells)
    far_mean = sum(by_hex[c] for c in far_cells) / len(far_cells)
    assert near_mean > far_mean * 10  # exponential decay; near should be >>10x larger


def test_metro_presence_boosts_target() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold_no_metro = _gold([cell], 8, count_stations_1km=[0])
    gold_with_metro = _gold([cell], 8, count_stations_1km=[1])

    r_no = compute_synthetic_target(gold_no_metro, seed=42)
    r_with = compute_synthetic_target(gold_with_metro, seed=42)

    assert r_with["synthetic_target_rub_per_m2"][0] > r_no["synthetic_target_rub_per_m2"][0]


def test_raises_on_missing_required_columns() -> None:
    df = pl.DataFrame({"h3_index": ["abc"], "resolution": [8]})  # missing building_count, count_stations_1km

    with pytest.raises(KeyError, match="required"):
        compute_synthetic_target(df, seed=42)

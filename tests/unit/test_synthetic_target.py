import h3
import polars as pl
import pytest

from kadastra.etl.synthetic_target import compute_synthetic_target

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _gold(
    h3_indices: list[str],
    resolution: int,
    *,
    building_count_apartments: list[int] | None = None,
    count_stations_1km: list[int] | None = None,
    road_length_m: list[float] | None = None,
) -> pl.DataFrame:
    n = len(h3_indices)
    return pl.DataFrame(
        {
            "h3_index": h3_indices,
            "resolution": [resolution] * n,
            "building_count_apartments": (
                building_count_apartments if building_count_apartments is not None else [0] * n
            ),
            "count_stations_1km": (
                count_stations_1km if count_stations_1km is not None else [0] * n
            ),
            "road_length_m": (
                road_length_m if road_length_m is not None else [0.0] * n
            ),
        }
    )


def test_compute_synthetic_target_adds_target_column() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold = _gold([cell], 8)

    result = compute_synthetic_target(gold, seed=42)

    assert "synthetic_target_rub_per_m2" in result.columns


def test_compute_synthetic_target_does_not_emit_geographic_columns() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold = _gold([cell], 8)

    result = compute_synthetic_target(gold, seed=42)

    assert "kazan_distance_km" not in result.columns


def test_target_is_non_negative_for_empty_hex() -> None:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(20)]
    gold = _gold(cells, 8)  # all features zero

    result = compute_synthetic_target(gold, seed=1)

    assert all(t >= 0 for t in result["synthetic_target_rub_per_m2"].to_list())


def test_deterministic_given_seed() -> None:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(10)]
    gold = _gold(cells, 8, building_count_apartments=list(range(10)))

    r1 = compute_synthetic_target(gold, seed=42)
    r2 = compute_synthetic_target(gold, seed=42)

    assert r1["synthetic_target_rub_per_m2"].to_list() == r2["synthetic_target_rub_per_m2"].to_list()


def test_different_seeds_produce_different_targets() -> None:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(20)]
    gold = _gold(cells, 8, building_count_apartments=[5] * 20)

    r1 = compute_synthetic_target(gold, seed=1)
    r2 = compute_synthetic_target(gold, seed=2)

    assert r1["synthetic_target_rub_per_m2"].to_list() != r2["synthetic_target_rub_per_m2"].to_list()


def test_more_apartment_buildings_yield_higher_mean_target() -> None:
    cells_low = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(50)]
    cells_high = [h3.latlng_to_cell(KAZAN_LAT - 0.01 * i, KAZAN_LON, 8) for i in range(50)]
    low = _gold(cells_low, 8, building_count_apartments=[1] * 50)
    high = _gold(cells_high, 8, building_count_apartments=[100] * 50)

    r_low = compute_synthetic_target(low, seed=42)
    r_high = compute_synthetic_target(high, seed=42)

    assert r_high["synthetic_target_rub_per_m2"].mean() > r_low["synthetic_target_rub_per_m2"].mean()  # type: ignore[operator]


def test_more_road_length_yields_higher_mean_target() -> None:
    cells_a = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(50)]
    cells_b = [h3.latlng_to_cell(KAZAN_LAT - 0.01 * i, KAZAN_LON, 8) for i in range(50)]
    no_roads = _gold(cells_a, 8, building_count_apartments=[5] * 50, road_length_m=[0.0] * 50)
    many_roads = _gold(cells_b, 8, building_count_apartments=[5] * 50, road_length_m=[5000.0] * 50)

    r_no = compute_synthetic_target(no_roads, seed=42)
    r_many = compute_synthetic_target(many_roads, seed=42)

    assert r_many["synthetic_target_rub_per_m2"].mean() > r_no["synthetic_target_rub_per_m2"].mean()  # type: ignore[operator]


def test_metro_presence_boosts_target() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    gold_no_metro = _gold([cell], 8, building_count_apartments=[10], count_stations_1km=[0])
    gold_with_metro = _gold([cell], 8, building_count_apartments=[10], count_stations_1km=[1])

    r_no = compute_synthetic_target(gold_no_metro, seed=42)
    r_with = compute_synthetic_target(gold_with_metro, seed=42)

    assert r_with["synthetic_target_rub_per_m2"][0] > r_no["synthetic_target_rub_per_m2"][0]


def test_target_has_floor_for_zero_features() -> None:
    """Even with zero features, mean target should sit above zero (rural baseline)."""
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(200)]
    gold = _gold(cells, 8)

    result = compute_synthetic_target(gold, seed=42)

    assert result["synthetic_target_rub_per_m2"].mean() > 0  # type: ignore[operator]


def test_raises_on_missing_required_columns() -> None:
    df = pl.DataFrame({"h3_index": ["abc"], "resolution": [8]})

    with pytest.raises(KeyError, match="required"):
        compute_synthetic_target(df, seed=42)

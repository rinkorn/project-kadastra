import polars as pl
import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.object_synthetic_target import compute_object_synthetic_target

_REQUIRED_COLS = {
    "object_id": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "levels": pl.Int64,
    "flats": pl.Int64,
    "dist_metro_m": pl.Float64,
    "count_stations_1km": pl.Int64,
    "count_apartments_500m": pl.Int64,
    "count_houses_500m": pl.Int64,
    "count_commercial_500m": pl.Int64,
    "road_length_500m": pl.Float64,
}


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_REQUIRED_COLS)


def _row(
    *,
    oid: str,
    ac: AssetClass,
    dist_metro_m: float,
    count_stations_1km: int = 0,
    count_apartments_500m: int = 0,
    count_houses_500m: int = 0,
    count_commercial_500m: int = 0,
    road_length_500m: float = 0.0,
    levels: int | None = None,
    flats: int | None = None,
) -> dict[str, object]:
    return {
        "object_id": oid,
        "asset_class": ac.value,
        "lat": 55.78,
        "lon": 49.12,
        "levels": levels,
        "flats": flats,
        "dist_metro_m": dist_metro_m,
        "count_stations_1km": count_stations_1km,
        "count_apartments_500m": count_apartments_500m,
        "count_houses_500m": count_houses_500m,
        "count_commercial_500m": count_commercial_500m,
        "road_length_500m": road_length_500m,
    }


def test_appends_target_column() -> None:
    df = _frame([_row(oid="way/1", ac=AssetClass.APARTMENT, dist_metro_m=500.0)])

    out = compute_object_synthetic_target(df, seed=42)

    assert "synthetic_target_rub_per_m2" in out.columns
    assert out["synthetic_target_rub_per_m2"][0] >= 0


def test_target_is_deterministic_for_same_seed() -> None:
    df = _frame([_row(oid="way/1", ac=AssetClass.APARTMENT, dist_metro_m=500.0)])

    a = compute_object_synthetic_target(df, seed=42)
    b = compute_object_synthetic_target(df, seed=42)

    assert a["synthetic_target_rub_per_m2"][0] == b["synthetic_target_rub_per_m2"][0]


def test_target_changes_with_seed() -> None:
    df = _frame([_row(oid="way/1", ac=AssetClass.APARTMENT, dist_metro_m=500.0)])

    a = compute_object_synthetic_target(df, seed=1)
    b = compute_object_synthetic_target(df, seed=2)

    assert a["synthetic_target_rub_per_m2"][0] != b["synthetic_target_rub_per_m2"][0]


def test_apartment_class_responds_to_metro_proximity() -> None:
    df = _frame(
        [
            _row(oid="close", ac=AssetClass.APARTMENT, dist_metro_m=200.0, count_stations_1km=2),
            _row(oid="far", ac=AssetClass.APARTMENT, dist_metro_m=5000.0, count_stations_1km=0),
        ]
    )

    out = compute_object_synthetic_target(df, seed=42)
    close = out.filter(pl.col("object_id") == "close")["synthetic_target_rub_per_m2"][0]
    far = out.filter(pl.col("object_id") == "far")["synthetic_target_rub_per_m2"][0]

    assert close > far


def test_house_class_average_below_apartment_at_same_features() -> None:
    rows: list[dict[str, object]] = []
    for i in range(50):
        rows.append(
            _row(
                oid=f"apt-{i}",
                ac=AssetClass.APARTMENT,
                dist_metro_m=1000.0,
                count_apartments_500m=10,
            )
        )
        rows.append(
            _row(
                oid=f"hs-{i}",
                ac=AssetClass.HOUSE,
                dist_metro_m=1000.0,
                count_apartments_500m=10,
            )
        )
    df = _frame(rows)

    out = compute_object_synthetic_target(df, seed=42)
    apt_mean = out.filter(pl.col("asset_class") == "apartment")["synthetic_target_rub_per_m2"].mean()
    house_mean = out.filter(pl.col("asset_class") == "house")["synthetic_target_rub_per_m2"].mean()

    assert isinstance(apt_mean, float) and isinstance(house_mean, float)
    assert apt_mean > house_mean


def test_commercial_responds_to_foot_traffic() -> None:
    df = _frame(
        [
            _row(
                oid="busy",
                ac=AssetClass.COMMERCIAL,
                dist_metro_m=400.0,
                count_apartments_500m=40,
                count_commercial_500m=20,
            ),
            _row(
                oid="quiet",
                ac=AssetClass.COMMERCIAL,
                dist_metro_m=400.0,
                count_apartments_500m=0,
                count_commercial_500m=0,
            ),
        ]
    )

    out = compute_object_synthetic_target(df, seed=42)
    busy = out.filter(pl.col("object_id") == "busy")["synthetic_target_rub_per_m2"][0]
    quiet = out.filter(pl.col("object_id") == "quiet")["synthetic_target_rub_per_m2"][0]

    assert busy > quiet


def test_targets_are_non_negative() -> None:
    rows = [
        _row(
            oid=f"obj-{i}",
            ac=AssetClass.APARTMENT,
            dist_metro_m=10_000.0,
        )
        for i in range(100)
    ]
    df = _frame(rows)

    out = compute_object_synthetic_target(df, seed=42)

    assert (out["synthetic_target_rub_per_m2"] >= 0).all()


def test_missing_required_column_raises() -> None:
    schema = {k: v for k, v in _REQUIRED_COLS.items() if k != "dist_metro_m"}
    df = pl.DataFrame(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": 55.78,
                "lon": 49.12,
                "levels": None,
                "flats": None,
                "count_stations_1km": 0,
                "count_apartments_500m": 0,
                "count_houses_500m": 0,
                "count_commercial_500m": 0,
                "road_length_500m": 0.0,
            }
        ],
        schema=schema,
    )

    with pytest.raises(KeyError, match="dist_metro_m"):
        compute_object_synthetic_target(df, seed=42)


def test_unknown_asset_class_raises() -> None:
    df = _frame([_row(oid="way/1", ac=AssetClass.APARTMENT, dist_metro_m=500.0)])
    df = df.with_columns(pl.lit("land_plot").alias("asset_class"))

    with pytest.raises(ValueError, match="land_plot"):
        compute_object_synthetic_target(df, seed=42)

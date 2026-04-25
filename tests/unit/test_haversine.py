import math

import pytest

from kadastra.etl.haversine import haversine_meters

MOSCOW = (55.7558, 37.6173)
SPB = (59.9343, 30.3351)
KAZAN = (55.7887, 49.1221)


def test_haversine_zero_distance_for_same_point() -> None:
    assert haversine_meters(*MOSCOW, *MOSCOW) == 0.0


def test_haversine_moscow_spb_is_about_635_km() -> None:
    distance = haversine_meters(*MOSCOW, *SPB)

    assert 630_000 < distance < 640_000


def test_haversine_is_symmetric() -> None:
    forward = haversine_meters(*MOSCOW, *KAZAN)
    backward = haversine_meters(*KAZAN, *MOSCOW)

    assert forward == pytest.approx(backward, rel=1e-12)


def test_haversine_antipodes_is_half_earth_circumference() -> None:
    distance = haversine_meters(0.0, 0.0, 0.0, 180.0)

    assert distance == pytest.approx(math.pi * 6_371_000.0, rel=1e-9)

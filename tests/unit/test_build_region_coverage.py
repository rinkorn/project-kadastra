from collections.abc import Iterable

import h3
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from kadastra.usecases.build_region_coverage import BuildRegionCoverage

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class FakeBoundary:
    def __init__(self, geometry: BaseGeometry) -> None:
        self._geometry = geometry
        self.calls: list[str] = []

    def get_boundary(self, region_code: str) -> BaseGeometry:
        self.calls.append(region_code)
        return self._geometry


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, list[tuple[str, int]]]] = []

    def save(self, region_code: str, cells: Iterable[tuple[str, int]]) -> None:
        self.saved.append((region_code, list(cells)))


def _kazan_polygon() -> BaseGeometry:
    return Point(KAZAN_LON, KAZAN_LAT).buffer(0.01)


def test_build_region_coverage_fetches_boundary_for_given_region() -> None:
    boundary = FakeBoundary(_kazan_polygon())
    store = FakeStore()
    usecase = BuildRegionCoverage(boundary, store)

    usecase.execute("RU-TA", resolutions=[8])

    assert boundary.calls == ["RU-TA"]


def test_build_region_coverage_stores_cells_for_each_resolution() -> None:
    boundary = FakeBoundary(_kazan_polygon())
    store = FakeStore()
    usecase = BuildRegionCoverage(boundary, store)

    usecase.execute("RU-TA", resolutions=[7, 8])

    assert len(store.saved) == 1
    region_code, cells = store.saved[0]
    assert region_code == "RU-TA"
    resolutions_seen = {res for _, res in cells}
    assert resolutions_seen == {7, 8}


def test_build_region_coverage_cells_match_requested_resolutions() -> None:
    boundary = FakeBoundary(_kazan_polygon())
    store = FakeStore()
    usecase = BuildRegionCoverage(boundary, store)

    usecase.execute("RU-TA", resolutions=[8, 10])

    _, cells = store.saved[0]
    for cell, declared_res in cells:
        assert h3.get_resolution(cell) == declared_res


def test_build_region_coverage_includes_centroid_cell() -> None:
    boundary = FakeBoundary(_kazan_polygon())
    store = FakeStore()
    usecase = BuildRegionCoverage(boundary, store)

    usecase.execute("RU-TA", resolutions=[8])

    _, cells = store.saved[0]
    centroid = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    assert (centroid, 8) in cells

import numpy as np
import polars as pl

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.object_metro_features import compute_object_metro_features
from kadastra.ports.road_graph import RoadGraphPort

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class _HaversineRoadGraph(RoadGraphPort):
    """A road graph that ignores topology and reports haversine distance.

    Used by tests whose assertions don't care about graph routing
    (only that some 'distance' is computed). The point is that
    compute_object_metro_features no longer hard-codes haversine —
    it takes a RoadGraphPort and asks it for distances. This fake
    is the simplest port impl that lets the existing tests' outcomes
    survive the refactor.
    """

    def distance_matrix_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        out = np.empty((len(from_coords), len(to_coords)), dtype=np.float64)
        for i, (la1, lo1) in enumerate(from_coords):
            for j, (la2, lo2) in enumerate(to_coords):
                out[i, j] = haversine_meters(la1, lo1, la2, lo2)
        return out


_FAKE_GRAPH = _HaversineRoadGraph()


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "asset_class": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "levels": pl.Int64,
        "flats": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def _points(rows: list[dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"lat": pl.Float64, "lon": pl.Float64})


def test_appends_distance_and_count_columns() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
            }
        ]
    )
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances, road_graph=_FAKE_GRAPH)

    expected_extra = {
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    }
    assert expected_extra.issubset(set(result.columns))
    assert result["dist_metro_m"][0] < 1.0
    assert result["count_stations_1km"][0] == 1
    assert result["count_entrances_500m"][0] == 1


def test_distant_object_has_high_distance_and_zero_counts() -> None:
    far_lat, far_lon = KAZAN_LAT + 0.5, KAZAN_LON + 0.5  # ~50 km away
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "house",
                "lat": far_lat,
                "lon": far_lon,
                "levels": None,
                "flats": None,
            }
        ]
    )
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances, road_graph=_FAKE_GRAPH)

    assert result["dist_metro_m"][0] > 10_000
    assert result["count_stations_1km"][0] == 0
    assert result["count_entrances_500m"][0] == 0


def test_preserves_object_columns_and_row_order() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
            },
            {
                "object_id": "way/2",
                "asset_class": "house",
                "lat": KAZAN_LAT + 0.01,
                "lon": KAZAN_LON,
                "levels": 1,
                "flats": None,
            },
        ]
    )
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances, road_graph=_FAKE_GRAPH)

    assert result["object_id"].to_list() == ["way/1", "way/2"]
    assert {"object_id", "asset_class", "lat", "lon", "levels", "flats"}.issubset(
        set(result.columns)
    )


def test_empty_objects_returns_empty_with_feature_columns() -> None:
    objects = _objects([])
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances, road_graph=_FAKE_GRAPH)

    assert result.is_empty()
    for col in (
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    ):
        assert col in result.columns


def test_uses_graph_distance_not_euclidean_when_graph_routing_is_longer() -> None:
    """The function must consult the road graph; euclidean shortcut is
    not allowed. Construct a graph where the only path from A to B
    detours through an intermediate node, so graph distance is
    strictly larger than the haversine straight line. The recorded
    dist_metro_m must be the graph distance, and count_stations_1km
    must reflect the same routing (not the euclidean shortcut).
    """
    from kadastra.adapters.networkx_road_graph import NetworkxRoadGraph

    object_coord = (KAZAN_LAT, KAZAN_LON)
    detour_coord = (KAZAN_LAT + 0.0040, KAZAN_LON)  # ~445 m north
    station_coord = (KAZAN_LAT, KAZAN_LON + 0.0050)  # ~313 m east

    graph = NetworkxRoadGraph.from_edges(
        [
            (
                object_coord,
                detour_coord,
                haversine_meters(*object_coord, *detour_coord),
            ),
            (
                detour_coord,
                station_coord,
                haversine_meters(*detour_coord, *station_coord),
            ),
        ]
    )

    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": object_coord[0],
                "lon": object_coord[1],
                "levels": 9,
                "flats": 72,
            }
        ]
    )
    stations = _points([{"lat": station_coord[0], "lon": station_coord[1]}])

    result = compute_object_metro_features(
        objects, stations, _points([]), road_graph=graph
    )

    euclidean = haversine_meters(*object_coord, *station_coord)
    expected_graph = haversine_meters(*object_coord, *detour_coord) + haversine_meters(
        *detour_coord, *station_coord
    )

    # Graph routing detours through node B → must be > euclidean.
    assert result["dist_metro_m"][0] > euclidean + 1.0
    # And matches the explicit graph path length.
    assert result["dist_metro_m"][0] == np.float64(expected_graph) or abs(
        result["dist_metro_m"][0] - expected_graph
    ) < 1.0
    # Station is ~313 m euclidean (under 1 km) but graph distance is
    # ~890 m which is also under 1 km — so the graph-based count
    # still includes it. We check the count is computed from graph
    # values by replacing the station with one whose graph distance
    # exceeds 1 km below.
    assert result["count_stations_1km"][0] == 1


def test_count_stations_1km_uses_graph_threshold_not_euclidean() -> None:
    """A station that is within 1 km euclidean but >1 km by graph
    must NOT be counted in count_stations_1km. This isolates the
    rule from the previous test by forcing the detour to push graph
    distance over the threshold.
    """
    from kadastra.adapters.networkx_road_graph import NetworkxRoadGraph

    object_coord = (KAZAN_LAT, KAZAN_LON)
    far_detour = (KAZAN_LAT + 0.0080, KAZAN_LON)  # ~890 m north
    station_coord = (KAZAN_LAT, KAZAN_LON + 0.0080)  # ~500 m east (euclidean)

    # Path goes object -> far_detour -> station. Total ≈ 890 + 1300 ≈ 2200 m.
    graph = NetworkxRoadGraph.from_edges(
        [
            (object_coord, far_detour, haversine_meters(*object_coord, *far_detour)),
            (far_detour, station_coord, haversine_meters(*far_detour, *station_coord)),
        ]
    )

    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": object_coord[0],
                "lon": object_coord[1],
                "levels": 9,
                "flats": 72,
            }
        ]
    )
    stations = _points([{"lat": station_coord[0], "lon": station_coord[1]}])

    result = compute_object_metro_features(
        objects, stations, _points([]), road_graph=graph
    )

    euclidean = haversine_meters(*object_coord, *station_coord)
    assert euclidean < 1000.0  # sanity: euclidean would have counted it
    assert result["dist_metro_m"][0] > 1500.0
    # Counted by graph distance, not euclidean → not within 1 km.
    assert result["count_stations_1km"][0] == 0


def test_no_stations_yields_inf_distance_and_zero_counts() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": None,
                "flats": None,
            }
        ]
    )
    empty_points = _points([])

    result = compute_object_metro_features(objects, empty_points, empty_points, road_graph=_FAKE_GRAPH)

    assert result["count_stations_1km"][0] == 0
    assert result["count_entrances_500m"][0] == 0
    # No reference points → use a sentinel that signals "unknown / never close":
    # finite but very large, so downstream models treat it as "far".
    assert result["dist_metro_m"][0] > 1e6
    assert result["dist_entrance_m"][0] > 1e6


def test_disconnected_object_yields_far_sentinel_not_inf() -> None:
    """If an object's nearest graph component has no path to any station,
    the road graph returns inf for that pair. CatBoost's split logic
    handles NaN cleanly but not inf — and we already use a finite
    'far sentinel' (1e9 m) for the empty-stations branch. Treating
    a disconnected object the same way keeps the downstream contract
    consistent: dist_metro_m is always finite (NaN only when we say
    so explicitly).
    """

    class _DisconnectedGraph(RoadGraphPort):
        """Returns inf for every pair — simulates total disconnect."""

        def distance_matrix_m(
            self,
            from_coords: list[tuple[float, float]],
            to_coords: list[tuple[float, float]],
        ) -> np.ndarray:
            return np.full(
                (len(from_coords), len(to_coords)), np.inf, dtype=np.float64
            )

    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "house",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": None,
                "flats": None,
            }
        ]
    )
    stations = _points([{"lat": KAZAN_LAT + 0.5, "lon": KAZAN_LON + 0.5}])
    entrances = _points([{"lat": KAZAN_LAT + 0.5, "lon": KAZAN_LON + 0.5}])

    result = compute_object_metro_features(
        objects, stations, entrances, road_graph=_DisconnectedGraph()
    )

    assert np.isfinite(result["dist_metro_m"][0])
    assert np.isfinite(result["dist_entrance_m"][0])
    assert result["dist_metro_m"][0] > 1e6
    assert result["dist_entrance_m"][0] > 1e6
    assert result["count_stations_1km"][0] == 0
    assert result["count_entrances_500m"][0] == 0

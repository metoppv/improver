# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the DistanceTo plugin."""

import os

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon

from improver.generate_ancillaries.generate_distance_to_feature import DistanceToFeature
from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)


@pytest.fixture()
def geometry_point_latlon():
    """Create a geometry containing points on a latitude, longitude grid.
    The location of the points is identical to geometry_point_laea, but in a different
    CRS.

    The points locations look like:
             x        x


             x        x
    """
    data = [
        Point(-1.393298, 49.538352),
        Point(-1.395401, 49.547221),
        Point(-1.379621, 49.539742),
        Point(-1.381721, 49.548611),
    ]

    return GeoDataFrame(geometry=data, crs="EPSG:4326")


@pytest.fixture()
def geometry_point_laea():
    """Create a geometry containing points on a Lambert azimuthal equal-area grid.
    The location of the points is identical to geometry_point_latlon, but in a
    different CRS.

    The points locations look like:
             x        x


             x        x
    """
    data = [
        Point(3500000, 3000000),
        Point(3500000, 3001000),
        Point(3501000, 3000000),
        Point(3501000, 3001000),
    ]
    return GeoDataFrame(geometry=data, crs="EPSG:3035")


@pytest.fixture()
def geometry_line_latlon():
    """Create a simple line geometry on a latitude, longitude grid.
    The line defined is identical to geometry_line_laea, but in a different CRS.

    The line looks like:
             x-------x
             |       |
             |       |
             |       |
             x-------x
    """
    data = [
        LineString(
            [
                [-1.393298, 49.538352],
                [-1.395401, 49.547221],
                [-1.381721, 49.548611],
                [-1.379621, 49.539742],
                [-1.393298, 49.538352],
            ]
        )
    ]

    return GeoDataFrame(geometry=data, crs="EPSG:4326")


@pytest.fixture()
def geometry_line_laea():
    """Create a simple line geometry on a Lambert azimuthal equal area grid.
    The line defined is identical to geometry_point_latlon, but in a different CRS.

    The line looks like:
             x-------x
             |       |
             |       |
             |       |
             x-------x
    """
    data = [
        LineString(
            [
                [3500000, 3000000],
                [3500000, 3001000],
                [3501000, 3001000],
                [3501000, 3000000],
                [3500000, 3000000],
            ]
        )
    ]
    return GeoDataFrame(geometry=data, crs="EPSG:3035")


@pytest.fixture()
def geometry_polygon_latlon():
    """Create a simple polygon geometry on a latitude, longitude grid.
    The polygon defined is identical to geometry_polygon_laea, but in a different CRS.

    The polygon looks like:
             x-------x
             ---------
             ---------
             x-------x
    """
    data = [
        Polygon(
            [
                [-1.393298, 49.538352],
                [-1.395401, 49.547221],
                [-1.381721, 49.548611],
                [-1.379621, 49.539742],
                [-1.393298, 49.538352],
            ]
        )
    ]
    return GeoDataFrame(geometry=data, crs="EPSG:4326")


@pytest.fixture()
def geometry_polygon_laea():
    """Create a simple polygon geometry on a Lambert azimuthal equal area grid.
    The polygon defined is identical to geometry_polygon_latlon, but in a different CRS.

    The polygon looks like:
             x-------x
             ---------
             ---------
             x-------x
    """
    data = [
        Polygon(
            [
                [3500000, 3000000],
                [3500000, 3001000],
                [3501000, 3001000],
                [3501000, 3000000],
                [3500000, 3000000],
            ]
        )
    ]
    return GeoDataFrame(geometry=data, crs="EPSG:3035")


def make_site_cube(latitudes, longitudes, name="rain_rate"):
    """Make a site cube."""

    prob_cube = set_up_spot_variable_cube(
        np.repeat(-9999, len(latitudes)),
        name="rain_rate",
        units="1",
        altitudes=np.repeat(-9999, len(latitudes)),
        wmo_ids=[f"{i:05d}" for i in range(len(latitudes))],
        latitudes=np.asarray(latitudes, np.dtypes.Float32DType),
        longitudes=np.asarray(longitudes, np.dtypes.Float32DType),
    )
    return prob_cube


def make_grid_cube(domain_corner, name="rain_rate"):
    """Make a site cube."""

    prob_cube = set_up_variable_cube(
        np.repeat(-9999, 1).reshape(1, 1),
        name="rain_rate",
        units="1",
        domain_corner=domain_corner,
        spatial_grid="equalarea",
    )
    return prob_cube


@pytest.fixture()
def single_site_cube():
    """Set up a site cube for a single site."""

    return make_site_cube(
        latitudes=[49.539047274],
        longitudes=[-1.386459578],
    )


@pytest.fixture()
def multiple_site_cube():
    """Set up a site cube containing data at multiple sites."""

    return make_site_cube(
        latitudes=[49.538352, 49.539047274, 49.543481633, 49.552350289],
        longitudes=[-1.393298, -1.386459578, -1.387510304, -1.389612479],
    )


@pytest.fixture()
def coastline():
    """Create a GeoDataFrame representing a simple coastline.
    x-------x
    |       |
    |       |
    |       |
    x-------x
    """

    data = [
        LineString(
            [
                [3500000, 3000000],
                [3500000, 3001000],
                [3501000, 3001000],
                [3501000, 3000000],
                [3500000, 3000000],
            ]
        )
    ]
    return GeoDataFrame(geometry=data, crs="EPSG:3035")


@pytest.fixture()
def land():
    """Create a simple polygon representing a land area surrounded
    by the coastline defined in the coastline fixture.

    The polygon looks like:
             x-------x
             ---------
             ---------
             x-------x
    """
    data = [
        Polygon(
            [
                [3500000, 3000000],
                [3500000, 3001000],
                [3501000, 3001000],
                [3501000, 3000000],
                [3500000, 3000000],
            ]
        )
    ]
    return GeoDataFrame(geometry=data, crs="EPSG:3035")


@pytest.fixture()
def site_locations():
    return make_site_cube(
        latitudes=[49.543481633, 49.551655272],
        longitudes=[-1.387510304, -1.3964531],
        name="site_locations",
    )


@pytest.fixture()
def single_site_at_point():
    return make_site_cube(
        latitudes=[49.538352],
        longitudes=[-1.393298],
        name="site_locations",
    )


@pytest.fixture()
def single_site_at_halfway_point():
    return make_site_cube(
        latitudes=[49.539047274],
        longitudes=[-1.386459578],
        name="site_locations",
    )


@pytest.fixture()
def single_grid_point_at_halfway_point():
    return make_grid_cube(domain_corner=(-595702.831, 80673.488))


@pytest.fixture()
def single_site_at_centre_point():
    return make_site_cube(
        latitudes=[49.543481633],
        longitudes=[-1.387510304],
        name="site_locations",
    )


@pytest.fixture()
def single_site_outside_points():
    return make_site_cube(
        latitudes=[49.551655272],
        longitudes=[-1.3964531],
        name="site_locations",
    )


@pytest.mark.parametrize(
    "target_projection, cube_fixture_name, expected_distance",
    [
        (3035, "single_site_at_point", 0),
        (3035, "single_site_at_halfway_point", 500),
        (3035, "single_grid_point_at_halfway_point", 500),
        (3035, "single_site_at_centre_point", 707),
        # Test a conic projection over Europe as well, which yields difference distances
        # for the non-zero distance cases.
        (9001, "single_site_at_point", 0),
        (9001, "single_site_at_halfway_point", 498),
        (9001, "single_site_at_centre_point", 603),
    ],
)
@pytest.mark.parametrize(
    "shape_file_crs", ["geometry_point_laea", "geometry_point_latlon"]
)
def test_distance_to_with_points_geometry(
    cube_fixture_name,
    shape_file_crs,
    target_projection,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin with a single site and a geometry
    of points."""

    geometry = request.getfixturevalue(shape_file_crs)

    single_site_cube = request.getfixturevalue(cube_fixture_name)

    output_cube = DistanceToFeature(target_projection, "distance_to_thing")(
        single_site_cube, geometry
    )
    assert output_cube.name() == "distance_to_thing"
    assert output_cube.units == "m"
    assert output_cube.coord(axis="x").points == single_site_cube.coord(axis="x").points
    assert output_cube.coord(axis="y").points == single_site_cube.coord(axis="y").points
    assert output_cube.data == expected_distance


@pytest.mark.parametrize(
    "target_projection, cube_fixture_name, expected_distance",
    [
        (
            3035,
            "single_site_at_point",
            0,
        ),  # site is the same location as a corner of the line
        (
            3035,
            "single_site_at_halfway_point",
            0,
        ),  # site is halfways between two points on the line
        (
            3035,
            "single_site_at_centre_point",
            500,
        ),  # Site is at the exact centre of the square formed by the line
        # Test a conic projection over Europe as well, which yields difference distances
        # for the non-zero distance cases.
        (
            9001,
            "single_site_at_point",
            0,
        ),  # site is the same location as a corner of the line
        (
            9001,
            "single_site_at_halfway_point",
            0,
        ),  # site is halfways between two points on the line
        (
            9001,
            "single_site_at_centre_point",
            381,
        ),  # Site is at the exact centre of the square formed by the line
    ],
)
@pytest.mark.parametrize("geometry_crs", ["geometry_line_laea", "geometry_line_latlon"])
def test_distance_to_with_line_geometry(
    cube_fixture_name,
    geometry_crs,
    target_projection,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin with a single site and a
    single line geometry."""

    geometry = request.getfixturevalue(geometry_crs)

    single_site_cube = request.getfixturevalue(cube_fixture_name)

    output_cube = DistanceToFeature(target_projection, "distance_to_thing")(
        single_site_cube, geometry
    )
    assert output_cube.name() == "distance_to_thing"
    assert output_cube.units == "m"
    assert (
        output_cube.coord("latitude").points
        == single_site_cube.coord("latitude").points
    )
    assert (
        output_cube.coord("longitude").points
        == single_site_cube.coord("longitude").points
    )
    assert output_cube.data == expected_distance


@pytest.mark.parametrize(
    "target_projection, cube_fixture_name, expected_distance",
    [
        (
            3035,
            "single_site_at_point",
            0,
        ),  # site is the same location as a corner of the polygon
        (
            3035,
            "single_site_at_halfway_point",
            0,
        ),  # site is halfways between two points on the edge of the polygon
        (
            3035,
            "single_site_at_centre_point",
            0,
        ),  # Site is at the exact centre of the polygon
        (3035, "single_site_outside_points", 500),  # Site is outside the polygon
        # Test a conic projection over Europe as well, which yields difference distances
        # for the non-zero distance cases.
        (
            9001,
            "single_site_at_point",
            0,
        ),  # site is the same location as a corner of the polygon
        (
            9001,
            "single_site_at_halfway_point",
            0,
        ),  # site is halfways between two points on the edge of the polygon
        (
            9001,
            "single_site_at_centre_point",
            0,
        ),  # Site is at the exact centre of the polygon
        (9001, "single_site_outside_points", 383),  # Site is outside the polygon
    ],
)
@pytest.mark.parametrize(
    "geometry_crs", ["geometry_polygon_laea", "geometry_polygon_latlon"]
)
def test_distance_to_with_polygon_geometry(
    cube_fixture_name,
    geometry_crs,
    target_projection,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin with a single site and a simple polygon geometry."""

    geometry = request.getfixturevalue(geometry_crs)

    single_site_cube = request.getfixturevalue(cube_fixture_name)

    output_cube = DistanceToFeature(target_projection, "distance_to_thing")(
        single_site_cube, geometry
    )
    assert output_cube.name() == "distance_to_thing"
    assert output_cube.units == "m"
    assert (
        output_cube.coord("latitude").points
        == single_site_cube.coord("latitude").points
    )
    assert (
        output_cube.coord("longitude").points
        == single_site_cube.coord("longitude").points
    )
    assert output_cube.data == expected_distance


def test_distance_to_feature_with_exclusion(site_locations, coastline, land):
    """Test the DistanceToFeature class calculates distance to feature with exclusion
    correctly."""

    # Instantiate the DistanceToFeature plugin
    calculator = DistanceToFeature(
        epsg_projection=3035,
        new_name="distance_to_ocean",
    )

    # Generate the distance to feature with exclusion
    distance_to_feature = calculator.process(
        site_locations,
        coastline,
        exclude_outside_of=land,
        exclusion_buffer=10,
    )

    # Ensure the cube has the correct metadata
    assert distance_to_feature.name() == "distance_to_ocean"
    assert distance_to_feature.units == "m"
    from numpy.testing import assert_array_equal

    assert_array_equal(distance_to_feature.data, [500, 0])


@pytest.mark.parametrize(
    "geometry_type,if_parallel,expected_distance",
    [
        ("point", False, [0, 500, 707, 707]),
        ("line", False, [0, 0, 500, 500]),
        ("polygon", False, [0, 0, 0, 500]),
        ("point", True, [0, 500, 707, 707]),
        ("line", True, [0, 0, 500, 500]),
        ("polygon", True, [0, 0, 0, 500]),
    ],
)
@pytest.mark.parametrize("geometry_crs", ("laea",))
def test_distance_to_with_multiple_sites(
    multiple_site_cube,
    geometry_type,
    geometry_crs,
    expected_distance,
    if_parallel,
    request,
):
    """Test the DistanceTo plugin works when provided a site cube with multiple sites
    and different types of geometry"""

    geometry = request.getfixturevalue(f"geometry_{geometry_type}_{geometry_crs}")

    n_jobs = 1
    if len(os.sched_getaffinity(0)) > 1:
        n_jobs = 2

    plugin = DistanceToFeature(
        3035, new_name="distance_to_thing", parallel=if_parallel, n_parallel_jobs=n_jobs
    )

    assert plugin.parallel == if_parallel
    assert plugin.n_parallel_jobs == n_jobs
    output_cube = plugin(multiple_site_cube, geometry)
    assert output_cube.name() == "distance_to_thing"
    assert output_cube.units == "m"

    np.testing.assert_allclose(output_cube.data, expected_distance)


def test_distance_to_with_new_name(single_site_at_halfway_point, geometry_point_laea):
    """Test the DistanceTo plugin correctly sets a new name."""

    output_cube = DistanceToFeature(3035, new_name="distance_to_river")(
        single_site_at_halfway_point, geometry_point_laea
    )
    assert output_cube.name() == "distance_to_river"
    assert output_cube.units == "m"
    assert (
        output_cube.coord("latitude").points
        == single_site_at_halfway_point.coord("latitude").points
    )
    assert (
        output_cube.coord("longitude").points
        == single_site_at_halfway_point.coord("longitude").points
    )


@pytest.mark.parametrize(
    "clip, buffer, expected",
    [(True, 100, [100, 800]), (True, 3000, [100, 200]), (False, None, [100, 200])],
)
def test_distance_to_clipping_loss_of_data(
    multiple_site_cube,
    clip,
    buffer,
    expected,
):
    """Test the DistanceTo plugin with clipping and buffer. The test involves two sites
    (o) and two features (x) configured as follows:

        (-100) (0)     (800) (1000)
            o   x        o    x

    The numbers represent their relative distance to each other in metres.
    """
    site_cubes = multiple_site_cube[0:2].copy()  # Use only the first two sites
    site_cubes.coord("latitude").points = [49.537465617, 49.545447324]
    site_cubes.coord("longitude").points = [-1.393088146, -1.394980629]

    data = [
        Point(3500000, 3000000),
        Point(3500000, 3001000),
    ]
    geometry = GeoDataFrame(geometry=data, crs="EPSG:3035")

    if clip:
        output_cube = DistanceToFeature(
            3035, new_name="distance_to_thing", clip_geometry=True, buffer=buffer
        )(site_cubes, geometry)
    else:
        output_cube = DistanceToFeature(
            3035, new_name="distance_to_thing", clip_geometry=False
        )(site_cubes, geometry)

    assert output_cube.name() == "distance_to_thing"
    assert output_cube.units == "m"
    np.testing.assert_allclose(output_cube.data, expected)


def test_distance_to_with_empty_geometry(single_site_cube, geometry_point_laea):
    """Test the DistanceTo plugin raises a ValueError when clipping leads to an empty
    geometry."""

    with pytest.raises(
        ValueError, match="Clipping the geometry with a buffer size of 100m"
    ):
        DistanceToFeature(
            3035, new_name="distance_to_thing", clip_geometry=True, buffer=100
        )(single_site_cube, geometry_point_laea)


def test_distance_to_with_unsuitable_projection(single_site_cube, geometry_point_laea):
    """
    Test the DistanceTo plugin raises a ValueError when the projection is unsuitable.
    """

    msg = (
        "The provided projection defined by EPSG code 3112 is not suitable "
        "for the site / grid locations provided. Limits of this domain are: x: 112.85 "
        "to 153.69, y: -43.7 to -9.86, whilst the site / grid locations are bounded by "
        "x: -1.3864595890045166 to -1.3864595890045166, y: 49.53904724121094 to 49.53904724121094."
    )
    with pytest.raises(ValueError, match=msg):
        DistanceToFeature(3112, new_name="distance_to_thing")(
            single_site_cube, geometry_point_laea
        )

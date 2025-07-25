# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the DistanceTo plugin."""

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon

from improver.generate_ancillaries.generate_distance_to_feature import DistanceTo
from improver.spotdata.build_spotdata_cube import build_spotdata_cube


@pytest.fixture()
def geometry_point_latlong():
    """Create a single point geometry on a latitude, longitude grid.
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
    """Create a single point geometry on a Lambert azimuthal equal-area grid.
    The location of the points is identical to geometry_point_latlong, but in a
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
def geometry_line_latlong():
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
    The line defined is identical to geometry_point_latlong, but in a different CRS.

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
def geometry_polygon_latlong():
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
    The polygon defined is identical to geometry_polygon_latlong, but in a different CRS.

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
def single_site_cube():
    """Set up a site cube for a single site."""

    latitude = 49.539047274  # This value is overridden in the test functions.
    longitude = -1.386459578  # This value is overridden in the test functions.

    altitude = -99999  # This value is not used but is required for cube creation.
    data = -99999  # This value is not used but is required for cube creation.
    wmo_id = ["00000"]  # This value is not used but is required for cube creation.

    prob_cube = build_spotdata_cube(
        data,
        name="rain_rate",
        units="1",
        altitude=altitude,
        wmo_id=wmo_id,
        latitude=latitude,
        longitude=longitude,
    )
    return prob_cube


@pytest.fixture()
def multiple_site_cube():
    """Set up a site cube containing data at multiple sites."""

    latitude = np.array([49.538352, 49.539047274, 49.543481633, 49.552350289])
    longitude = np.array([-1.393298, -1.386459578, -1.387510304, -1.389612479])

    altitude = np.array(
        [-99999, -99999, -99999, -99999]
    )  # These values are not used but are required for cube creation.
    data = np.array(
        [-99999, -99999, -99999, -99999]
    )  # These values are not used but are required for cube creation.
    wmo_id = [
        "00000",
        "00001",
        "00002",
        "00003",
    ]  # These values are not used but are required for cube creation.

    prob_cube = build_spotdata_cube(
        data,
        name="rain_rate",
        units="1",
        altitude=altitude,
        wmo_id=wmo_id,
        latitude=latitude,
        longitude=longitude,
    )
    return prob_cube


@pytest.mark.parametrize(
    "site_latitude, site_longitude, expected_distance",
    [
        (49.538352, -1.393298, 0),  # site is the same location as a point
        (49.539047274, -1.386459578, 500),  # site is halfways between two points
        (
            49.543481633,
            -1.387510304,
            707,
        ),  # Site is at the exact centre of the 4 points
    ],
)
@pytest.mark.parametrize(
    "shape_file_crs", ["geometry_point_laea", "geometry_point_latlong"]
)
def test_distance_to_with_points_geometry(
    single_site_cube,
    shape_file_crs,
    site_latitude,
    site_longitude,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin with a single site and a geometry
    of points."""

    geometry = request.getfixturevalue(shape_file_crs)

    single_site_cube.coord("latitude").points = site_latitude
    single_site_cube.coord("longitude").points = site_longitude

    output_cube = DistanceTo()(single_site_cube, geometry)
    assert output_cube.name() == "rain_rate"
    assert output_cube.units == "m"
    assert output_cube.coord("latitude").points == site_latitude
    assert output_cube.coord("longitude").points == site_longitude
    assert output_cube.data == expected_distance


@pytest.mark.parametrize(
    "site_latitude, site_longitude, expected_distance",
    [
        (49.538352, -1.393298, 0),  # site is the same location as a corner of the line
        (
            49.539047274,
            -1.386459578,
            0,
        ),  # site is halfways between two points on the line
        (
            49.543481633,
            -1.387510304,
            500,
        ),  # Site is at the exact centre of the square formed by the line
    ],
)
@pytest.mark.parametrize(
    "geometry_crs", ["geometry_line_laea", "geometry_line_latlong"]
)
def test_distance_to_with_line_geometry(
    single_site_cube,
    geometry_crs,
    site_latitude,
    site_longitude,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin with a single site and a
    single line geometry."""

    geometry = request.getfixturevalue(geometry_crs)

    single_site_cube.coord("latitude").points = site_latitude
    single_site_cube.coord("longitude").points = site_longitude

    output_cube = DistanceTo()(single_site_cube, geometry)
    assert output_cube.name() == "rain_rate"
    assert output_cube.units == "m"
    assert output_cube.coord("latitude").points == site_latitude
    assert output_cube.coord("longitude").points == site_longitude
    assert output_cube.data == expected_distance


@pytest.mark.parametrize(
    "site_latitude, site_longitude, expected_distance",
    [
        (
            49.538352,
            -1.393298,
            0,
        ),  # site is the same location as a corner of the polygon
        (
            49.539047274,
            -1.386459578,
            0,
        ),  # site is halfways between two points on the edge of the polygon
        (49.543481633, -1.387510304, 0),  # Site is at the exact centre of the polygon
        (49.551655272, -1.3964531, 500),  # Site is outside the polygon
    ],
)
@pytest.mark.parametrize(
    "geometry_crs", ["geometry_polygon_laea", "geometry_polygon_latlong"]
)
def test_distance_to_with_polygon_geometry(
    single_site_cube,
    geometry_crs,
    site_latitude,
    site_longitude,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin with a single site and a simple polygon geometry."""

    geometry = request.getfixturevalue(geometry_crs)

    single_site_cube.coord("latitude").points = site_latitude
    single_site_cube.coord("longitude").points = site_longitude

    output_cube = DistanceTo()(single_site_cube, geometry)
    assert output_cube.name() == "rain_rate"
    assert output_cube.units == "m"
    assert output_cube.coord("latitude").points == site_latitude
    assert output_cube.coord("longitude").points == site_longitude
    assert output_cube.data == expected_distance


@pytest.mark.parametrize(
    "geometry_type,expected_distance",
    [
        ("point", [0, 500, 707, 707]),
        ("line", [0, 0, 500, 500]),
        ("polygon", [0, 0, 0, 500]),
    ],
)
@pytest.mark.parametrize("geometry_crs", ("laea", "latlong"))
def test_distance_to_with_multiple_sites(
    multiple_site_cube,
    geometry_type,
    geometry_crs,
    expected_distance,
    request,
):
    """Test the DistanceTo plugin works when provided a site cube with multiple sites and
    different types of geometry"""

    geometry = request.getfixturevalue(f"geometry_{geometry_type}_{geometry_crs}")

    output_cube = DistanceTo()(multiple_site_cube, geometry)
    assert output_cube.name() == "rain_rate"
    assert output_cube.units == "m"

    np.testing.assert_allclose(output_cube.data, expected_distance)


def test_distance_to_with_new_name(single_site_cube, geometry_point_laea):
    """Test the DistanceTo plugin correctly sets a new name."""

    single_site_cube.coord("latitude").points = 49.539047274
    single_site_cube.coord("longitude").points = -1.386459578

    output_cube = DistanceTo(new_name="distance_to_river")(
        single_site_cube, geometry_point_laea
    )
    assert output_cube.name() == "distance_to_river"
    assert output_cube.units == "m"
    assert output_cube.coord("latitude").points == 49.539047274
    assert output_cube.coord("longitude").points == -1.386459578


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
        output_cube = DistanceTo(clip=True, buffer=buffer)(site_cubes, geometry)
    else:
        output_cube = DistanceTo(clip=False)(site_cubes, geometry)

    assert output_cube.name() == "rain_rate"
    assert output_cube.units == "m"
    np.testing.assert_allclose(output_cube.data, expected)


def test_distance_to_with_empty_geometry(single_site_cube, geometry_point_laea):
    """Test the DistanceTo plugin raises a ValueError when clipping leads to an empty
    geometry."""

    with pytest.raises(
        ValueError, match="Clipping the geometry with a buffer size of 100m"
    ):
        DistanceTo(clip=True, buffer=100)(single_site_cube, geometry_point_laea)

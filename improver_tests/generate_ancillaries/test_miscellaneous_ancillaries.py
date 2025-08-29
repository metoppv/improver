# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the the miscellaneous ancillary generation functions.
"""

import cartopy.crs as ccrs
import numpy as np
import pytest
from geopandas import GeoDataFrame
from iris.cube import CubeList
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shapely.geometry import LineString, Polygon

from improver.generate_ancillaries.generate_miscellaneous_ancillaries import (
    generate_distance_to_ocean,
    generate_distance_to_water,
    generate_land_area_fraction_at_sites,
    generate_roughness_length_at_sites,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture()
def distance_cube_template():
    """Set up a site cube containing data at multiple sites."""

    latitude = np.array([48, 49, 50, 51])
    longitude = np.array([-1, 0, -1, 0])

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
        name="distance_to_something",
        units="m",
        altitude=altitude,
        wmo_id=wmo_id,
        latitude=latitude,
        longitude=longitude,
    )
    return prob_cube


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
    """Set up a site cube containing data at multiple sites."""
    latitude = np.array([49.543481633, 49.551655272])
    longitude = np.array([-1.387510304, -1.3964531])

    altitude = np.array(
        [-99999, -99999]
    )  # These values are not used but are required for cube creation.
    data = np.array(
        [-99999, -99999]
    )  # These values are not used but are required for cube creation.
    wmo_id = [
        "00000",
        "00001",
    ]  # These values are not used but are required for cube creation.
    site_cube = build_spotdata_cube(
        data,
        name="site_locations",
        units="m",
        altitude=altitude,
        wmo_id=wmo_id,
        latitude=latitude,
        longitude=longitude,
    )
    return site_cube


def neighbour_cube(neighbours, altitudes, latitudes, longitudes, wmo_ids):
    """Set up a neighbour cube with a simple grid of neighbours."""

    grid_attributes = ["x_index", "y_index", "vertical_displacement"]
    neighbour_methods = ["nearest", "nearest_land"]
    neighbour_cube = build_spotdata_cube(
        neighbours,
        "grid_neighbours",
        1,
        altitudes,
        latitudes,
        longitudes,
        wmo_ids,
        grid_attributes=grid_attributes,
        neighbour_methods=neighbour_methods,
    )
    neighbour_cube.attributes["model_grid_hash"] = (
        "e5b78c90234ed2f4a17f4109abce231c3826577bd738940af0e227c9e2892069"
    )
    return neighbour_cube


@pytest.fixture()
def default_neighbour_cube():
    """Set up a neighbour cube with a simple grid of neighbours."""

    neighbours = np.array(
        [
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, -1.0, 0.0, 1.0]],
            [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [-1.0, 0.0, 0.0, 1.0]],
        ]
    )

    altitudes = np.array([0, 1, 3, 2])
    latitudes = np.array([10, 10, 20, 20])
    longitudes = np.array([10, 10, 20, 20])
    wmo_ids = np.arange(4)

    return neighbour_cube(neighbours, altitudes, latitudes, longitudes, wmo_ids)


@pytest.fixture()
def corine_land_cover():
    """Set up a dummy CORINE land cover."""
    data = np.array(
        [
            [44, 44, 44, 44],
            [41, 41, 6, 7],
            [41, 9, 10, 11],
            [41, 13, 14, 15],
        ]
    )

    attributes = {
        "something_unhelpful": "remove_this",
        "conventions": "CF-1.7",
    }

    corine_cube = set_up_variable_cube(
        data,
        name="corine_land_cover",
        units="1",
        spatial_grid="equalarea",
        x_grid_spacing=2500,
        y_grid_spacing=2500,
        attributes=attributes,
    )

    return corine_cube


@pytest.fixture()
def gridded_template_cube():
    """Set up a gridded template cube with a simple grid."""
    data = np.array([[0, 1, 1], [2, 3, 3], [4, 5, 5]])
    cube = set_up_variable_cube(
        data,
        name="gridded_template",
        units="1",
        spatial_grid="equalarea",
        x_grid_spacing=2500,
        y_grid_spacing=2500,
    )
    return cube


def test_distance_to_water(distance_cube_template):
    """Test the distance to water ancillary is generated correctly."""

    river_cube = distance_cube_template.copy()
    river_cube.data = np.array([100, 200, 300, 400])
    lake_cube = distance_cube_template.copy()
    lake_cube.data = np.array([400, 300, 200, 100])
    ocean_cube = distance_cube_template.copy()
    ocean_cube.data = np.array([200, 200, 200, 10])
    water_cubes = CubeList([river_cube, lake_cube, ocean_cube])

    output_cube = generate_distance_to_water(water_cubes)

    assert output_cube.name() == "distance_to_water"
    assert output_cube.units == "m"
    assert_array_equal(output_cube.data, [100, 200, 200, 10])


def test_distance_to_ocean(site_locations, coastline, land):
    """Test the distance to ocean ancillary is generated correctly."""

    # Generate the distance to ocean ancillary
    distance_to_ocean = generate_distance_to_ocean(
        3035, coastline, land, site_locations
    )

    # Ensure the cube has the correct metadata
    assert distance_to_ocean.name() == "distance_to_ocean"
    assert distance_to_ocean.units == "m"
    assert_array_equal(distance_to_ocean.data, [500, 0])


@pytest.mark.parametrize(
    "radius, expected",
    (
        (2500, [0.0, 1.0 / 3.0, 8.0 / 9.0, 1.0]),
        (5000, [1.0 / 3.0, 0.5, 0.5, 8.0 / 9.0]),
        (20000, [0.5, 0.5, 0.5, 0.5]),
    ),
)
def test_land_area_fraction_ancillary(corine_land_cover, radius, expected):
    """Test that the land area fraction ancillary is generated correctly.
    A site list is defined that traverses the corine land cover cube from
    top left to bottom right across the grid for nearest neighbours.

    The corine land cover grid looks like this once translated into a
    land sea mask:

        [[0 0 0 0]
         [0 0 1 1]
         [0 1 1 1]
         [0 1 1 1]]

    The cell spacing is 2500m.
    With a radius of 2500m the top left point gets a 2x2 neighbourhood full
    of sea (0) and is classified as 100% sea (0). The next point down and to
    the right has a 3x3 neighbourhood that is 1/3 land (1) and 2/3 sea (0),
    giving a land area fraction of 1/3. The next point down and to the right
    has a 3x3 neighbourhood that is 8/9 land and 1/9 sea, giving a land area
    fraction of 8/9. The final point in the bottom right has a 2x2
    neighbourhood that is 100% land, giving a land area fraction of 1.

    Larger radii are tested which capture more of the domain and give different
    land fractions. The last test has a radius large enough to capture the
    entire domain for all sites, giving a land area fraction of 0.5 for all
    sites.
    """
    x_points = corine_land_cover.coord(axis="x").points
    y_points = corine_land_cover.coord(axis="y").points
    target_crs = corine_land_cover.coord(axis="x").coord_system.as_cartopy_crs()
    longitudes, latitudes = (
        ccrs.PlateCarree().transform_points(target_crs, x_points, y_points)[:, 0:2].T
    )

    # The grid point neighbour indices are not used, so all set to 1.
    neighbours = neighbour_cube(
        np.ones((2, 3, 4)), np.ones(4), latitudes, longitudes, np.arange(4)
    )

    land_area_fraction = generate_land_area_fraction_at_sites(
        corine_land_cover,
        neighbours,
        radius=radius,
    )

    assert land_area_fraction.name() == "land_area_fraction"
    assert land_area_fraction.units == "1"
    assert_array_almost_equal(land_area_fraction.data, expected)


def test_roughness_length_ancillary(default_neighbour_cube, gridded_template_cube):
    """Test that the roughness length ancillary is generated correctly."""

    # Create a gridded roughness length cube
    roughness_cube = gridded_template_cube.copy()
    roughness_cube.rename("roughness_length")
    roughness_cube.units = "m"
    roughness_cube.data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Generate the roughness length ancillary
    result = generate_roughness_length_at_sites(roughness_cube, default_neighbour_cube)

    # Ensure the cube has the correct metadata
    assert result.name() == "roughness_length"
    assert result.units == "m"
    assert_array_almost_equal(result.data, [0.1, 0.1, 0.9, 0.9])

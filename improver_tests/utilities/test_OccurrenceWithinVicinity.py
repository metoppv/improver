# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the utilities.OccurrenceWithinVicinity plugin."""

import datetime
from typing import Tuple

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube
from numpy import ndarray

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.spatial import OccurrenceWithinVicinity

RADIUS = 2000
GRID_POINT_RADIUS = 1


def land_mask_cube_generator(shape: Tuple[int, int] = (5, 5)) -> Cube:
    """Creates a land-mask cube for use in these tests"""
    mask = np.zeros(shape, dtype=np.int8)
    mask[:, 3:] = 1
    return set_up_variable_cube(
        mask,
        name="land_binary_mask",
        units="1",
        spatial_grid="equalarea",
        x_grid_spacing=2000.0,
        y_grid_spacing=2000.0,
        domain_corner=(0.0, 0.0),
    )


@pytest.fixture
def all_land_cube() -> Cube:
    cube = land_mask_cube_generator((4, 4))
    cube.data = np.zeros_like(cube.data)
    return cube


@pytest.fixture
def land_mask_cube() -> Cube:
    cube = land_mask_cube_generator()
    return cube


@pytest.fixture
def binary_expected() -> ndarray:
    return np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def cube() -> Cube:
    """Sets up a cube for testing"""
    return set_up_variable_cube(
        np.zeros((5, 5), dtype=np.float32),
        spatial_grid="equalarea",
        x_grid_spacing=2000.0,
        y_grid_spacing=2000.0,
        domain_corner=(0.0, 0.0),
    )


@pytest.fixture
def latlon_cube() -> Cube:
    """Sets up a lat-lon cube for testing"""
    return set_up_variable_cube(
        np.zeros((5, 5), dtype=np.float32),
        spatial_grid="latlon",
        x_grid_spacing=1.0,
        y_grid_spacing=1.0,
        domain_corner=(0.0, 0.0),
    )


@pytest.fixture
def radius_coord() -> AuxCoord:
    return AuxCoord(
        np.array([RADIUS], dtype=np.float32), units="m", long_name="radius_of_vicinity"
    )


@pytest.fixture
def grid_point_radius_coord() -> AuxCoord:
    return AuxCoord(
        np.array([GRID_POINT_RADIUS], dtype=np.float32),
        units="1",
        long_name="radius_of_vicinity",
        attributes={
            "comment": "Units of 1 indicate radius of vicinity is defined "
            "in grid points rather than physical distance"
        },
    )


@pytest.mark.parametrize(
    "kwargs", ({"radii": [RADIUS]}, {"grid_point_radii": [GRID_POINT_RADIUS]})
)
def test_basic(cube, binary_expected, kwargs):
    """Test for binary events to determine where there is an occurrence
    within the vicinity."""

    cube.data[0, 1] = 1.0
    cube.data[2, 3] = 1.0
    result = OccurrenceWithinVicinity(**kwargs).process(cube)
    assert isinstance(result, Cube)
    assert np.allclose(result.data, binary_expected)


@pytest.mark.parametrize(
    "kwargs, expected_coord",
    (
        ({"radii": [RADIUS]}, "radius_coord"),
        ({"grid_point_radii": [GRID_POINT_RADIUS]}, "grid_point_radius_coord"),
        ({"radii": [RADIUS], "apply_cell_method": False}, "radius_coord"),
    ),
)
def test_metadata(request, cube, kwargs, expected_coord):
    """Test that the metadata on the cube reflects the data it contains
    following the application of vicinity processing.

    Parameterisation tests this using a radius defined as a distance or as
    a number of grid points."""

    expected_coord = request.getfixturevalue(expected_coord)

    plugin = OccurrenceWithinVicinity(**kwargs)
    # repeat the test with the same plugin instance to ensure self variables
    # have not been modified.
    for _ in range(2):
        result = plugin.process(cube)
        assert isinstance(result, Cube)
        assert result.coord("radius_of_vicinity") == expected_coord
        assert "in_vicinity" in result.name()
        if "apply_cell_method" in kwargs:
            assert result.cell_methods == ()
        else:
            assert "area" in result.cell_methods[0].coord_names
            assert result.cell_methods[0].method == "maximum"


def test_basic_latlon(latlon_cube, binary_expected):
    """Test for occurrence in vicinity calculation on a lat-lon (non equal
    area) grid using a grid_point_radius."""

    latlon_cube.data[0, 1] = 1.0
    latlon_cube.data[2, 3] = 1.0
    result = OccurrenceWithinVicinity(grid_point_radii=[GRID_POINT_RADIUS]).process(
        latlon_cube
    )
    assert isinstance(result, Cube)
    assert np.allclose(result.data, binary_expected)


def test_fuzzy(cube):
    """Test for non-binary events to determine where there is an occurrence
    within the vicinity."""
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    cube.data[0, 1] = 1.0
    cube.data[2, 3] = 0.5
    result = OccurrenceWithinVicinity(radii=[RADIUS]).process(cube)
    assert isinstance(result, Cube)
    assert np.allclose(result.data, expected)


@pytest.mark.parametrize(
    "kwargs", ({"radii": [2 * RADIUS]}, {"grid_point_radii": [2 * GRID_POINT_RADIUS]})
)
def test_different_distance(cube, kwargs):
    """Test for binary events to determine where there is an occurrence
    within the vicinity for an alternative distance."""
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    cube.data[0, 1] = 1.0
    cube.data[2, 3] = 1.0
    result = OccurrenceWithinVicinity(**kwargs).process(cube)
    assert isinstance(result, Cube)
    assert np.allclose(result.data, expected)


def test_masked_data(cube):
    """Test masked values are ignored in OccurrenceWithinVicinity."""
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    cube.data[0, 1] = 1.0
    cube.data[2, 3] = 1.0
    cube.data[0, 4] = 10.0
    mask = np.zeros((5, 5))
    mask[0, 4] = 1
    cube.data = np.ma.array(cube.data, mask=mask)
    result = OccurrenceWithinVicinity(radii=[RADIUS]).process(cube)
    assert isinstance(result, Cube)
    assert isinstance(result.data, np.ma.core.MaskedArray)
    assert np.allclose(result.data.data, expected)
    assert np.allclose(result.data.mask, mask)


def test_with_land_mask(cube, land_mask_cube):
    """Test that a land mask is used correctly."""
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    cube.data[0, 1] = 1.0  # would not cross mask
    cube.data[2, 3] = 1.0  # would cross mask
    cube.data[0, 4] = 10.0  # would not cross mask
    result = OccurrenceWithinVicinity(
        radii=[RADIUS], land_mask_cube=land_mask_cube
    ).process(cube)
    assert isinstance(result, Cube)
    assert ~isinstance(result.data, np.ma.core.MaskedArray)
    assert np.allclose(result.data, expected)


def test_with_land_mask_and_mask(cube, land_mask_cube):
    """Test that a land mask is used correctly when cube also has a mask."""
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    cube.data[0, 1] = 1.0  # would not cross mask
    cube.data[2, 3] = 1.0  # would cross mask
    cube.data[0, 4] = 10.0  # is masked in input
    mask = np.zeros((5, 5))
    mask[0, 4] = 1
    cube.data = np.ma.array(cube.data, mask=mask)
    result = OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=land_mask_cube)(
        cube
    )
    assert isinstance(result, Cube)
    assert isinstance(result.data, np.ma.core.MaskedArray)
    assert np.allclose(result.data.data, expected)
    assert np.allclose(result.data.mask, mask)


def test_with_invalid_land_mask_name(land_mask_cube):
    """Test that a mis-named land mask is rejected correctly."""
    bad_mask_cube = land_mask_cube.copy()
    bad_mask_cube.rename("kittens")
    with pytest.raises(
        ValueError,
        match="Expected land_mask_cube to be called land_binary_mask, not kittens",
    ):
        OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=bad_mask_cube)


def test_with_invalid_land_mask_coords(cube, land_mask_cube):
    """Test that a spatially mis-matched land mask is rejected correctly."""
    bad_mask_cube = land_mask_cube.copy()
    bad_points = np.array(bad_mask_cube.coord(axis="x").points)
    bad_points[0] += 1
    bad_mask_cube.coord(axis="x").points = bad_points
    with pytest.raises(
        ValueError,
        match="Supplied cube do not have the same spatial coordinates and land mask",
    ):
        OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=bad_mask_cube)(cube)


@pytest.fixture(name="cube_with_realizations")
def cube_with_realizations_fixture() -> Cube:
    return set_up_variable_cube(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="lwe_precipitation_rate",
        units="m s-1",
        spatial_grid="equalarea",
        x_grid_spacing=2000.0,
        y_grid_spacing=2000.0,
        domain_corner=(0.0, 0.0),
    )


TIMESTEPS = [datetime.datetime(2017, 11, 9, 12), datetime.datetime(2017, 11, 9, 15)]


@pytest.mark.parametrize("land_fixture", [None, "all_land_cube"])
def test_with_multiple_realizations_and_times(
    request, cube_with_realizations, land_fixture
):
    """Test for multiple realizations and times, so that multiple
    iterations will be required within the process method."""
    cube = cube_with_realizations
    land = request.getfixturevalue(land_fixture) if land_fixture else None
    expected = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ],
        ]
    )
    cube = add_coordinate(cube, TIMESTEPS, "time", order=[1, 0, 2, 3], is_datetime=True)
    cube.data[0, 0, 2, 1] = 1.0
    cube.data[1, 1, 1, 3] = 1.0
    orig_shape = cube.data.copy().shape
    result = OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=land)(cube)
    assert isinstance(result, Cube)
    assert result.data.shape == orig_shape
    assert np.allclose(result.data, expected)


@pytest.mark.parametrize(
    "kwargs", (({"radii": [2000, 5000]}), ({"grid_point_radii": [2, 3, 4, 5]}))
)
def test_coordinate_order_with_multiple_realizations_and_times(
    cube_with_realizations, kwargs
):
    """Test the output coordinate order for input cubes with multiple
    realizations and times using multiple vicinity radii."""

    cube = cube_with_realizations
    cube = add_coordinate(cube, TIMESTEPS, "time", order=[1, 0, 2, 3], is_datetime=True)

    # Add the expected radius_of_vicinity coordinate dimension size
    orig_shape = list(cube.data.copy().shape)
    orig_shape.insert(-2, len(list(kwargs.values())[0]))

    # Add the expected radius_of_vicinity coordinate dimension name
    orig_order = [crd.name() for crd in cube.coords(dim_coords=True)]
    orig_order.insert(-2, "radius_of_vicinity")

    result = OccurrenceWithinVicinity(**kwargs)(cube)
    result_order = [crd.name() for crd in result.coords(dim_coords=True)]

    assert list(result.data.shape) == orig_shape
    assert result_order == orig_order


@pytest.mark.parametrize("land_fixture", [None, "all_land_cube"])
def test_with_multiple_realizations(request, cube_with_realizations, land_fixture):
    """Test for multiple realizations, so that multiple
    iterations will be required within the process method."""
    cube = cube_with_realizations
    land = request.getfixturevalue(land_fixture) if land_fixture else None
    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    cube.data[0, 2, 1] = 1.0
    cube.data[1, 1, 3] = 1.0
    result = OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=land)(cube)
    assert isinstance(result, Cube)
    assert np.allclose(result.data, expected)


@pytest.mark.parametrize("land_fixture", [None, "all_land_cube"])
def test_with_multiple_times(request, cube_with_realizations, land_fixture):
    """Test for multiple times, so that multiple
    iterations will be required within the process method."""
    cube = cube_with_realizations
    land = request.getfixturevalue(land_fixture) if land_fixture else None
    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    cube = cube[0]
    cube = add_coordinate(cube, TIMESTEPS, "time", is_datetime=True)
    cube.data[0, 2, 1] = 1.0
    cube.data[1, 1, 3] = 1.0
    orig_shape = cube.data.shape
    result = OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=land)(cube)
    assert isinstance(result, Cube)
    assert result.data.shape == orig_shape
    assert np.allclose(result.data, expected)


@pytest.mark.parametrize("land_fixture", [None, "all_land_cube"])
def test_no_realization_or_time(request, cube_with_realizations, land_fixture):
    """Test for no realizations and no times, so that the iterations
    will not require slicing cubes within the process method."""
    cube = cube_with_realizations
    land = request.getfixturevalue(land_fixture) if land_fixture else None
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    cube = cube[0]
    cube.data[2, 1] = 1.0
    orig_shape = cube.data.shape
    result = OccurrenceWithinVicinity(radii=[RADIUS], land_mask_cube=land)(cube)
    assert isinstance(result, Cube)
    assert result.data.shape == orig_shape
    assert np.allclose(result.data, expected)


@pytest.mark.parametrize("radius", [-1, 10, 2000])
def test_two_radii_types_provided_exception(cube, radius):
    """Test an exception is raised if both radii and grid_point_radii are
    provided as non-zero arguments."""

    expected = (
        "Vicinity processing requires that only one of radii or "
        "grid_point_radii should be set"
    )

    with pytest.raises(ValueError, match=expected):
        OccurrenceWithinVicinity(radii=[radius], grid_point_radii=2)


@pytest.mark.parametrize("kwargs", ({"radii": None, "grid_point_radii": None}, {}))
def test_no_radii_provided_exception(cube, kwargs):
    """Test an exception is raised if neither radii nor grid_point_radii are
    set to non-zero values."""

    expected = (
        "Vicinity processing requires that one of radii or "
        "grid_point_radii should be set to a non-zero value"
    )

    with pytest.raises(ValueError, match=expected):
        OccurrenceWithinVicinity(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        ({"radii": [-2000], "grid_point_radii": None}),
        ({"radii": None, "grid_point_radii": [-1]}),
    ),
)
def test_negative_radii_provided_exception(cube, kwargs):
    """Test an exception is raised if the radius provided in either form is
    a negative value."""

    expected = "Vicinity processing requires only positive vicinity radii"

    with pytest.raises(ValueError, match=expected):
        OccurrenceWithinVicinity(**kwargs).get_grid_point_radius(cube)


@pytest.mark.parametrize(
    "kwargs",
    (
        ({"radii": [2000], "grid_point_radii": None}),
        ({"radii": [5000], "grid_point_radii": None}),
        ({"radii": [2000, 5000], "grid_point_radii": None}),
        ({"radii": None, "grid_point_radii": [1]}),
        ({"radii": None, "grid_point_radii": [3]}),
        ({"radii": None, "grid_point_radii": [1, 3]}),
    ),
)
def test_radius_set_correctly(cube, kwargs):
    """Test the radius is set correctly within the plugin."""

    (expected,) = [value for value in kwargs.values() if value is not None]
    plugin = OccurrenceWithinVicinity(**kwargs)
    assert plugin.radii == expected

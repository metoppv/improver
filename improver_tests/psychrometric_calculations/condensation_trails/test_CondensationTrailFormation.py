# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CondensationTrailFormation plugin"""

from typing import List, Optional, Tuple

import iris
import numpy as np
import pytest
from iris.cube import Cube

from improver.psychrometric_calculations.condensation_trails import (
    CondensationTrailFormation,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


def cube_with_pressure_factory(
    data: np.ndarray, name: str, pressure_levels: np.ndarray
) -> Cube:
    """
    Create an Iris cube with a specified pressure DimCoord as the
    leading dimension.

    Args:
        data (np.ndarray): The data array for the cube, with the first
            dimension matching the length of pressure_levels.
        name (str): The name of the variable (e.g., "air_temperature").
        pressure_levels (np.ndarray): 1D array of pressure levels to use
            as the leading dimension coordinate.

    Returns:
        Cube: An Iris cube with the specified data, name, and pressure
            coordinate.
    """
    cube = set_up_variable_cube(
        data,
        name=name,
        units="K" if "temperature" in name else "%" if "humidity" in name else "m",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    pressure_coord = iris.coords.DimCoord(
        pressure_levels, units="Pa", var_name="pressure"
    )
    # Remove first dim coord if present
    if cube.coords(dim_coords=True):
        cube.remove_coord(cube.coords(dim_coords=True)[0].name())
    cube.add_dim_coord(pressure_coord, 0)
    return cube


@pytest.mark.parametrize(
    "provided_engine_contrail_factors, expected_factors",
    [
        (None, np.array([3e-5, 3.4e-5, 3.9e-5], dtype=np.float32)),  # Test default
        ([1e-5, 1.4e-5, 1.9e-5], np.array([1e-5, 1.4e-5, 1.9e-5], dtype=np.float32)),
        ([2e-5, 2.4e-5, 2.9e-5], np.array([2e-5, 2.4e-5, 2.9e-5], dtype=np.float32)),
        ([3e-5, 3.4e-5, 3.9e-5], np.array([3e-5, 3.4e-5, 3.9e-5], dtype=np.float32)),
    ],
)
def test_initialisation_with_arguments_and_defaults(
    provided_engine_contrail_factors: Optional[List[float]],
    expected_factors: np.ndarray,
) -> None:
    """
    Test that the CondensationTrailFormation plugin can be initialised
    with custom or default engine contrail factors.

    This test checks that when a custom list of engine contrail factors
    is provided to the plugin, or when no list is provided (using
    defaults), the internal _engine_contrail_factors attribute is
    correctly set as a numpy array with the expected values.

    Args:
        provided_engine_contrail_factors (Optional[List[float]]): List
            of engine contrail factors to initialise the plugin with,
            or None for defaults.
        expected_factors (np.ndarray): The expected numpy array of
            engine contrail factors.
    """
    if provided_engine_contrail_factors is not None:
        plugin = CondensationTrailFormation(
            engine_contrail_factors=provided_engine_contrail_factors
        )
    else:
        plugin = CondensationTrailFormation()
    assert isinstance(plugin._engine_contrail_factors, np.ndarray)
    np.testing.assert_array_equal(plugin._engine_contrail_factors, expected_factors)


@pytest.mark.parametrize(
    "pressure_levels",
    [
        (np.array([100000], dtype=np.float32)),
        (np.array([100000, 90000], dtype=np.float32)),
        (np.array([100000, 90000, 80000], dtype=np.float32)),
    ],
)
def test_pressure_levels(
    pressure_levels: np.ndarray,
) -> None:
    """
    Test that the CondensationTrailFormation plugin correctly sets the
    pressure_levels attribute after processing cubes with various
    pressure level arrays.

    This test creates temperature, humidity, and height cubes with the
    given pressure levels, runs the CondensationTrailFormation plugin,
    and checks that the plugin's pressure_levels attribute matches the
    input pressure levels.

    Args:
        pressure_levels (np.ndarray): Array of pressure levels to use
            for the cubes.
    """
    shape = (len(pressure_levels), 3, 2)
    temperature_cube = cube_with_pressure_factory(
        np.full(shape, 250, dtype=np.float32), "air_temperature", pressure_levels
    )
    humidity_cube = cube_with_pressure_factory(
        np.full(shape, 50, dtype=np.float32), "relative_humidity", pressure_levels
    )
    height_cube = cube_with_pressure_factory(
        np.full(shape, 10000, dtype=np.float32), "geopotential_height", pressure_levels
    )

    plugin = CondensationTrailFormation()
    plugin.process(temperature_cube, humidity_cube, height_cube)

    # Check that pressure_levels attribute is set correctly
    np.testing.assert_array_equal(plugin.pressure_levels, pressure_levels)


@pytest.mark.parametrize(
    "pressure_levels, expected_shape, expected_mixing_ratios",
    [
        (
            np.array([100000], dtype=np.float32),
            (3, 1),
            np.array([[4.823306], [5.466414], [6.2702975]], dtype=np.float32),
        ),
        (
            np.array([100000, 90000], dtype=np.float32),
            (3, 2),
            np.array(
                [[4.823306, 4.3409758], [5.466414, 4.919772], [6.2702975, 5.643268]],
                dtype=np.float32,
            ),
        ),
        (
            np.array([100000, 90000, 80000], dtype=np.float32),
            (3, 3),
            np.array(
                [
                    [4.823306, 4.3409758, 3.8586447],
                    [5.466414, 4.919772, 4.373131],
                    [6.2702975, 5.643268, 5.016238],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_engine_mixing_ratio(
    pressure_levels: np.ndarray,
    expected_shape: Tuple,
    expected_mixing_ratios: np.ndarray,
) -> None:
    """
    Test that the engine mixing ratios are calculated correctly for
    various pressure level arrays.

    This test creates temperature, humidity, and height cubes with the
    given pressure levels, runs the CondensationTrailFormation plugin,
    and checks that the calculated engine mixing ratios match the
    expected values and shapes.

    Args:
        pressure_levels (np.ndarray): Array of pressure levels to use
            for the cubes.
        expected_shape (tuple): Expected shape of the mixing ratios
            output.
        expected_mixing_ratios (np.ndarray): Expected mixing ratios for
            the given input.
    """
    shape = (len(pressure_levels), 3, 2)
    temperature_cube = cube_with_pressure_factory(
        np.full(shape, 250, dtype=np.float32), "air_temperature", pressure_levels
    )
    humidity_cube = cube_with_pressure_factory(
        np.full(shape, 50, dtype=np.float32), "relative_humidity", pressure_levels
    )
    height_cube = cube_with_pressure_factory(
        np.full(shape, 10000, dtype=np.float32), "geopotential_height", pressure_levels
    )

    plugin = CondensationTrailFormation()
    plugin.process(temperature_cube, humidity_cube, height_cube)

    # Check that calculate_engine_mixing_ratios works after process
    mixing_ratios = plugin.calculate_engine_mixing_ratios()
    np.testing.assert_array_equal(mixing_ratios, expected_mixing_ratios)

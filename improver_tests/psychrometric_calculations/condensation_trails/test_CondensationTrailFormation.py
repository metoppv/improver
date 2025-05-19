# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CondensationTrailFormation plugin"""

import iris
import numpy as np
import pytest

from improver.psychrometric_calculations.condensation_trails import (
    CondensationTrailFormation,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture
def cube_with_pressure_factory():
    def _make_cube(data, name, pressure_levels):
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

    return _make_cube


def test_CondensationTrailFormation_initialisation():
    """Check that the plugin is initialised correctly."""
    plugin = CondensationTrailFormation()
    assert isinstance(plugin._engine_contrail_factors, np.ndarray)
    #! The values here are placeholders and should be replaced with the actual expected values
    np.testing.assert_array_equal(
        plugin._engine_contrail_factors, np.array([1, 2, 3], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "pressure_levels",
    [
        (np.array([100000], dtype=np.float32)),
        (np.array([100000, 90000], dtype=np.float32)),
        (np.array([100000, 90000, 80000], dtype=np.float32)),
    ],
)
def test_pressure_levels(cube_with_pressure_factory, pressure_levels):
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
            np.array([[160776.87385446], [321553.74770893], [482330.62156339]]),
        ),
        (
            np.array([100000, 90000], dtype=np.float32),
            (3, 2),
            np.array(
                [
                    [160776.87385446, 144699.18646902],
                    [321553.74770893, 289398.37293804],
                    [482330.62156339, 434097.55940705],
                ]
            ),
        ),
        (
            np.array([100000, 90000, 80000], dtype=np.float32),
            (3, 3),
            np.array(
                [
                    [160776.87385446, 144699.18646902, 128621.49908357],
                    [321553.74770893, 289398.37293804, 257242.99816714],
                    [482330.62156339, 434097.55940705, 385864.49725072],
                ]
            ),
        ),
    ],
)
def test_engine_mixing_ratio(
    cube_with_pressure_factory, pressure_levels, expected_shape, expected_mixing_ratios
):
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
    assert mixing_ratios.shape == expected_shape

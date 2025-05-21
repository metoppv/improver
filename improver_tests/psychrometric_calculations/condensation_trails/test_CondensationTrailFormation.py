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
    np.testing.assert_array_equal(
        plugin._engine_contrail_factors,
        np.array([3e-5, 3.4e-5, 3.9e-5], dtype=np.float32),
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
    print(repr(mixing_ratios))
    assert mixing_ratios.shape == expected_shape
    np.testing.assert_array_equal(mixing_ratios, expected_mixing_ratios)

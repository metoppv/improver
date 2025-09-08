# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the PrecipitableWater plugin"""

import iris
import iris.coords
import iris.cube
import numpy as np
import pytest

from improver.psychrometric_calculations.total_precipitable_water import (
    PrecipitableWater,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture
def mock_cube():
    """
    Fixture to create a mock humidity mixing ratio cube with pressure levels.
    """
    pressure_levels = np.array(
        [100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000],
        dtype=np.float32,
    )  # Pa
    data = np.random.uniform(0.001, 0.01, size=(len(pressure_levels), 5, 5)).astype(
        np.float32
    )  # realistic variation

    # Create a basic cube
    cube = set_up_variable_cube(
        data, name="specific_humidity", units="kg kg-1", spatial_grid="latlon"
    )

    # Replace the vertical coordinate with pressure
    pressure_coord = iris.coords.DimCoord(
        pressure_levels,
        standard_name="air_pressure",
        units="Pa",
        long_name="pressure",
        var_name="pressure",
    )
    cube.remove_coord("realization")  # if present
    cube.add_dim_coord(pressure_coord, 0)

    return cube


def test_precipitable_water_basic(mock_cube):
    plugin = PrecipitableWater(model_type="global")
    result = plugin.process(mock_cube)
    assert result.units == "m"
    assert result.standard_name == "lwe_thickness_of_precipitation_amount"
    assert result.attributes["least_significant_digit"] == 3
    assert (
        result.attributes["title"]
        == "Global Enhanced Model Forecast on Global 10 km Standard Grid"
    )
    assert result.data.shape == mock_cube.data.shape[1:]


def test_metadata_for_ukv_model(mock_cube):
    plugin = PrecipitableWater(model_type="ukv")
    result = plugin.process(mock_cube)
    assert (
        result.attributes["title"]
        == "UKV Enhanced Model Forecast on UK 2 km Standard Grid"
    )


def test_pressure_conversion_and_shape(mock_cube):
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    # Expect shape to match horizontal dimensions only
    expected_shape = mock_cube.data.shape[1:]  # (5, 5)
    assert result.data.shape == expected_shape


def test_invalid_input_type():
    plugin = PrecipitableWater()
    with pytest.raises(AttributeError):
        plugin.process("not_a_cube")


def test_least_significant_digit_attribute(mock_cube):
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    assert "least_significant_digit" in result.attributes
    assert result.attributes["least_significant_digit"] == 3


def test_output_units_and_standard_name(mock_cube):
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    assert result.units == "m"
    assert result.standard_name == "lwe_thickness_of_precipitation_amount"

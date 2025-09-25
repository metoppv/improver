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
        data,
        name="specific_humidity",
        units="kg kg-1",
        spatial_grid="latlon",
        vertical_levels=pressure_levels,
        pressure=True,
        attributes={
            "title": "Global Enhanced Model Forecast on Global 10 km Standard Grid"
        },
    )

    return cube


def test_precipitable_water_basic(mock_cube):
    """
    Test basic functionality and metadata of the Precipitable Water plugin.
    """
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    assert result.units == "m"
    assert result.standard_name == "lwe_thickness_of_precipitation_amount"
    assert (
        result.attributes["title"]
        == "Global Enhanced Model Forecast on Global 10 km Standard Grid"
    )
    assert result.data.shape == mock_cube.data.shape


def test_metadata_for_ukv_model(mock_cube):
    """
    Check that the metadata is preserved correctly for the UKV model input.
    """
    # Override title of mock_cube for UKV model test
    mock_cube.attributes["title"] = (
        "UKV Enhanced Model Forecast on UK 2 km Standard Grid"
    )
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    assert (
        result.attributes["title"]
        == "UKV Enhanced Model Forecast on UK 2 km Standard Grid"
    )


def test_pressure_conversion_and_shape(mock_cube):
    """
    Verify output shape matches input cube's horizontal dimensions.
    """
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    # Expect shape to match horizontal dimensions only
    expected_shape = mock_cube.data.shape
    assert result.data.shape == expected_shape


def test_invalid_input_type():
    """
    Ensure plugin raises error when input is not a cube.
    """
    plugin = PrecipitableWater()
    with pytest.raises(AttributeError):
        plugin.process("not_a_cube")


def test_output_units_and_standard_name(mock_cube):
    """
    Confirm output units and standard names are correctly set.
    """
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    assert result.units == "m"
    assert result.standard_name == "lwe_thickness_of_precipitation_amount"


def test_precipitable_water_known_values():
    """
    Test the plugin with a cube with known input humidity values and pressure levels to verify output.
    """
    pressure_levels = np.array([100000, 90000], dtype=np.float32)  # Pa
    data = np.full((2, 2, 2), 0.005, dtype=np.float32)  # uniform humidity

    cube = set_up_variable_cube(
        data,
        name="specific_humidity",
        units="kg kg-1",
        spatial_grid="latlon",
        vertical_levels=pressure_levels,
        pressure=True,
        attributes={
            "title": "Global Enhanced Model Forecast on Global 10 km Standard Grid"
        },
    )

    plugin = PrecipitableWater()
    result = plugin.process(cube)

    # Expected calculation:
    # delta_p = abs(100000 - 90000) = 10000 Pa
    # TPW = 0.005 * 10000 / (9.80665 * 1000) ≈ 0.005096 m
    expected_value = 0.005096
    expected_array = np.full((2, 2, 2), expected_value, dtype=np.float32)

    np.testing.assert_allclose(result.data, expected_array, rtol=1e-4, atol=1e-6)


def test_process_with_pressure_coord(mock_cube):
    """
    Ensure plugin retrieves 'pressure' coordinate successfully when present.
    """
    plugin = PrecipitableWater()
    result = plugin.process(mock_cube)
    assert isinstance(result, iris.cube.Cube)
    assert result.coord("pressure") is not None


def test_process_raises_if_pressure_coord_missing(mock_cube):
    """
    Raise ValueError if cube does not contain a 'pressure' coordinate.
    """
    cube = mock_cube.copy()
    cube.remove_coord("pressure")

    plugin = PrecipitableWater()
    with pytest.raises(ValueError, match="Cube must have a 'pressure' coordinate."):
        plugin.process(cube)


def test_plugin_converts_hpa_to_pa(mock_cube):
    """
    Modify the mock_cube to have pressure levels in hPa and test that the plugin converts them to Pa.
    """
    # Simulate the cube originally having pressure levels in hPa
    pressure_coord = mock_cube.coord("pressure")
    pressure_coord.convert_units("hPa")

    assert pressure_coord.units == "hPa"

    plugin = PrecipitableWater()
    result_cube = plugin.process(mock_cube)

    pressure_coord_converted = result_cube.coord("pressure")
    assert (
        pressure_coord_converted.units == "Pa"
    ), "Pressure units were not converted to Pa"

    expected_pa = np.array(
        [100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(pressure_coord_converted.points, expected_pa)

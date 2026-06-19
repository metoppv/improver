# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the VirtualTemperatureFromSpecificHumidity plugin"""

import re

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.exceptions import ConstraintMismatchError

from improver.temperature.virtual_temperature import (
    VirtualTemperatureFromSpecificHumidity,
)

# Set up fixtures for test cubes


@pytest.fixture(name="temperature_cube")
def temperature_cube_fixture() -> Cube:
    """
    Set up a temperature cube for use in tests.

    Returns:
        A valid cube names "air_temperature" with 2 realizations, 7 latitudes
          spanning from 60S to 60N and 3 longitudes.
    """
    coord = AuxCoord(0, standard_name="forecast_period")
    data = np.full((2, 7, 3), dtype=np.float32, fill_value=273.15)
    cube = Cube(
        data,
        standard_name="air_temperature",
        units="K",
    )
    cube.add_aux_coord(coord)
    return cube


@pytest.fixture(name="specific_humidity_cube")
def specific_humidity_cube_fixture() -> Cube:
    """
    Set up a cube representing specific humidity for use in tests.

    Returns:
        Cube: named "specific_humidity"
    """
    coord = AuxCoord(0, standard_name="forecast_period")
    data = np.full((1, 2, 3), dtype=np.int64, fill_value=1)
    cube = Cube(data, standard_name="specific_humidity", units="kg kg-1")
    cube.add_aux_coord(coord)
    return cube


@pytest.fixture(name="cloud_liquid_water_mixing_ratio")
def cloud_liquid_water_mixing_ratio_cube_fixture() -> Cube:
    """
    Set up a cube representing cloud_liquid_water_mixing_ratio for use in tests.

    Returns:
        Cube: named "cloud_water_mixing_ratio"
    """
    coord = AuxCoord(0, standard_name="forecast_period")
    data = np.full((1, 2, 3), dtype=np.int64, fill_value=1)
    cube = Cube(data, standard_name="cloud_liquid_water_mixing_ratio", units="kg kg-1")
    cube.add_aux_coord(coord)
    return cube


@pytest.fixture(name="cloud_ice_mixing_ratio")
def cloud_ice_mixing_ratio_cube_fixture() -> Cube:
    """
    Set up a cube representing cloud_ice_mixing_ratio for use in tests.

    Returns:
        Cube: named "cloud_ice_mixing_ratio"
    """
    coord = AuxCoord(0, standard_name="forecast_period")
    data = np.full((1, 2, 3), dtype=np.int64, fill_value=1)
    cube = Cube(data, standard_name="cloud_ice_mixing_ratio", units="kg kg-1")
    cube.add_aux_coord(coord)
    return cube


@pytest.fixture(name="incorrect_cube")
def incorrect_cube_fixture() -> Cube:
    """
    Set up a cube representing an incorrect input for use in tests.

    Returns:
        Cube: named "humidity_mixing_ratio", which is incorrect for this plugin.
    """
    coord = AuxCoord(0, standard_name="forecast_period")
    data = np.full((1, 2, 3), dtype=np.int64, fill_value=1)
    cube = Cube(data, standard_name="humidity_mixing_ratio", units="kg kg-1")
    cube.add_aux_coord(coord)
    return cube


class TestInitialisation:
    """
    Test the plugin is set up with the required attributes.
    """

    def test_init_for_virtual_temperature_specific_humidity(self):
        """Test the empty set up."""
        result = VirtualTemperatureFromSpecificHumidity()
        assert result.temperature is None
        assert result.specific_humidity is None
        assert result.cloud_water_mixing_ratio is None
        assert result.cloud_ice_mixing_ratio is None


class TestProcess:
    def test_process_with_virt_temp_spec_hum_temp_and_spec_hum(
        self, temperature_cube, specific_humidity_cube
    ):
        """Test the plugin with valid required inputs only."""
        expected_data = np.ndarray(shape=(2, 7, 3), dtype=np.float64)
        cubes = [temperature_cube, specific_humidity_cube]
        plugin = VirtualTemperatureFromSpecificHumidity()
        result = plugin.process(cubes)
        assert result.name() == "virtual_temperature"
        assert result.units == "K"
        assert type(result.data) == type(expected_data)
        assert result.data.shape == expected_data.shape
        assert result.data.dtype == expected_data.dtype
        assert result.data[0][0][0] == np.float64(439.1620211204289)

    def test_process_with_condensates(
        self,
        temperature_cube,
        specific_humidity_cube,
        cloud_liquid_water_mixing_ratio,
        cloud_ice_mixing_ratio,
    ):
        """Test the plugin with valid required and both optional inputs."""
        expected_data = np.ndarray(shape=(2, 7, 3), dtype=np.float64)
        cubes = [
            temperature_cube,
            specific_humidity_cube,
            cloud_liquid_water_mixing_ratio,
            cloud_ice_mixing_ratio,
        ]
        plugin = VirtualTemperatureFromSpecificHumidity()
        result = plugin.process(cubes)
        assert result.name() == "virtual_temperature"
        assert result.units == "K"
        assert type(result.data) == type(expected_data)
        assert result.data.shape == expected_data.shape
        assert result.data.dtype == expected_data.dtype
        assert result.data[0][0][0] == np.float64(439.1620211204289)

    def test_process_with_water_condensate(
        self,
        temperature_cube,
        specific_humidity_cube,
        cloud_liquid_water_mixing_ratio,
    ):
        """Test the plugin with only water as an optional input."""
        expected_data = np.ndarray(shape=(2, 7, 3), dtype=np.float64)
        cubes = [
            temperature_cube,
            specific_humidity_cube,
            cloud_liquid_water_mixing_ratio,
        ]
        plugin = VirtualTemperatureFromSpecificHumidity()
        result = plugin.process(cubes)
        assert result.name() == "virtual_temperature"
        assert result.units == "K"
        assert type(result.data) == type(expected_data)
        assert result.data.shape == expected_data.shape
        assert result.data.dtype == expected_data.dtype
        assert result.data[0][0][0] == np.float64(712.3120150169133)

    def test_process_with_only_ice_condensate(
        self,
        temperature_cube,
        specific_humidity_cube,
        cloud_ice_mixing_ratio,
    ):
        """Test the plugin with only ice as an optional input."""
        expected_data = np.ndarray(shape=(2, 7, 3), dtype=np.float64)
        cubes = [
            temperature_cube,
            specific_humidity_cube,
            cloud_ice_mixing_ratio,
        ]
        plugin = VirtualTemperatureFromSpecificHumidity()
        result = plugin.process(cubes)
        assert result.name() == "virtual_temperature"
        assert result.units == "K"
        assert type(result.data) == type(expected_data)
        assert result.data.shape == expected_data.shape
        assert result.data.dtype == expected_data.dtype
        assert result.data[0][0][0] == np.float64(712.3120150169133)

    def test_process_cubes_incorrect(self, temperature_cube, incorrect_cube):
        """Test the plugin does not calculate the result if given the incorrect input."""
        plugin = VirtualTemperatureFromSpecificHumidity()
        cubes = [temperature_cube, incorrect_cube]
        plugin = VirtualTemperatureFromSpecificHumidity()
        with pytest.raises(
            ConstraintMismatchError,
            match=re.escape(
                "Got 0 cubes for constraint Constraint(name='specific_humidity'), expecting 1."
            ),
        ):
            plugin.process(cubes)

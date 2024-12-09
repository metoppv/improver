# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the VirtualTemperature plugin"""

from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.cube import Cube

from improver.virtual_temperature import VirtualTemperature
from improver_tests.utilities.copy_metadata.test_CopyMetadata import HaltExecution


@pytest.fixture(name="temperature_cube")
def temperature_cube_fixture() -> Cube:
    """
    Set up a temperature cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """
    data = np.full((2, 7, 3), dtype=np.float32, fill_value=273.15)
    cube = Cube(
        data,
        standard_name="air_temperature",
        units="K",
    )
    return cube


@pytest.fixture(name="humidity_mixing_ratio_cube")
def humidity_mixing_ratio_cube_fixture() -> Cube:
    """
    Set up a humidity mixing ratio cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """
    data = np.full((2, 7, 3), dtype=np.float32, fill_value=0.01)
    cube = Cube(
        data,
        standard_name="humidity_mixing_ratio",
        units="1",
    )
    return cube


@pytest.fixture(name="virtual_temperature_cube")
def virtual_temperature_cube_fixture() -> Cube:
    """
    Set up a virtual temperature cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """
    data = np.full((2, 7, 3), dtype=np.float32, fill_value=274.81622)
    cube = Cube(
        data,
        standard_name="virtual_temperature",
        units="K",
        attributes={"units_metadata": "on-scale"},
    )
    return cube


@patch("improver.virtual_temperature.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    """Test that the as_cubelist function is called with the input cubes"""
    mock_as_cubelist.side_effect = HaltExecution
    try:
        VirtualTemperature()(sentinel.cube1, sentinel.cube2)
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(sentinel.cube1, sentinel.cube2)


def test_VirtualTemperature_get_virtual_temperature(
    temperature_cube, humidity_mixing_ratio_cube, virtual_temperature_cube
):
    """Test the get_virtual_temperature method produces a virtual temperature cube"""
    result = VirtualTemperature()(temperature_cube, humidity_mixing_ratio_cube)
    assert result == virtual_temperature_cube

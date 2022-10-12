# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Unit tests for the ExtractPressureLevel plugin"""

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_extraction import ExtractPressureLevel

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}
pytest.importorskip("stratify")


@pytest.fixture
def temperature_on_pressure_levels() -> Cube:
    """Set up a r, p, y, x cube of temperature of atmosphere on pressure levels"""
    temperatures = np.array([300, 286, 280, 274, 267, 262, 257, 245], dtype=np.float32)
    data = np.broadcast_to(
        temperatures.reshape((1, len(temperatures), 1, 1)), (2, len(temperatures), 3, 2)
    )
    t_cube = set_up_variable_cube(
        data,
        pressure=True,
        height_levels=np.arange(100000, 29999, -10000),
        name="temperature_on_pressure_levels",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return t_cube


def metadata_check(pressure_slice_cube: Cube, value: float, units: str):
    """Checks the cube produced by plugin has the expected metadata."""
    assert pressure_slice_cube.long_name == f"pressure_of_atmosphere_at_{value}{units}"
    assert pressure_slice_cube.units == "Pa"
    assert pressure_slice_cube.attributes == {
        "title": "unit test data",
        "source": "unit test",
        "institution": "somewhere",
    }


def cube_shape_check_with_realizations(pressure_slice_cube):
    """Checks cube coordinates and dimensions when two realizations are present"""
    coord_names = [coord.name() for coord in pressure_slice_cube.coords()]
    assert coord_names == [
        "realization",
        "latitude",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "time",
    ]
    assert pressure_slice_cube.shape == (2, 3, 2)


def cube_shape_check_without_realizations(pressure_slice_cube):
    """Checks cube coordinates and dimensions when realization is a scalar coord"""
    coord_names = [coord.name() for coord in pressure_slice_cube.coords()]
    assert coord_names == [
        "latitude",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "realization",
        "time",
    ]
    assert pressure_slice_cube.shape == (3, 2)


@pytest.mark.parametrize("with_realization", (True, False))
@pytest.mark.parametrize(
    "temperature,expected",
    (
        (280, 80000),  # Exactly matches a pressure value
        (277, 75000),  # Half way between pressure values
        (301, 100000),  # Temperature above max snaps to max
        (244, 30000),  # Temperature below min snaps to min
    ),
)
def test_basic(
    temperature, temperature_on_pressure_levels, expected, with_realization,
):
    """Tests the ExtractPressureLevel plugin with values for temperature and
    temperature on pressure levels to check for expected result.
    Also checks the metadata of the output cube"""
    expected_data = np.full_like(
        temperature_on_pressure_levels.data[:, 0, ...], expected
    )
    if not with_realization:
        temperature_on_pressure_levels = temperature_on_pressure_levels[0]
        expected_data = expected_data[0]
    result = ExtractPressureLevel(value_of_pressure_level=temperature)(
        temperature_on_pressure_levels
    )
    np.testing.assert_array_almost_equal(result.data, expected_data)
    metadata_check(result, temperature, temperature_on_pressure_levels.units)
    if with_realization:
        cube_shape_check_with_realizations(result)
    else:
        cube_shape_check_without_realizations(result)

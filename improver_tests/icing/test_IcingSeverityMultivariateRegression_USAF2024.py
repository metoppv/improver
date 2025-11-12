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
"""Test methods in icing.IcingSeverityMultivariateRegression_USAF2024"""

from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.icing import IcingSeverityMultivariateRegression_USAF2024
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.spatial import create_vicinity_coord


@pytest.fixture(name="temp_cube")
def temp_cube_fixture() -> Cube:
    """
    Set up an air temperature cube for use in tests over a variety of conditions.
    """

    data = np.array([[270, 272], [266, 280]], dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="rh_cube")
def rh_cube_fixture() -> Cube:
    """
    Set up a relative humidity cube for use in tests over a variety of conditions.
    """

    data = np.array([[65, 75], [90, 80]], dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="relative_humidity",
        units="%",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube




@pytest.fixture(name="expected_cube")
def expected_cube_fixture() -> Cube:
    """
    Set up the Icing cube that we expect to get from the plugin.
    """

    data = np.array([[0, 45.593143], [84.660645, 0 ]], dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="aircraft_icing_severity_index",
        units="1",
        time=datetime(2017, 11, 10, 3, 0),
        attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    

    return cube


def test_basic(temp_cube, rh_cube, expected_cube):
    """Run the plugin and check the result cube matches the expected_cube"""
    result = IcingSeverityMultivariateRegression_USAF2024()(
        CubeList([temp_cube, rh_cube]), None
    )
    assert result.xml().splitlines(keepends=True) == expected_cube.xml().splitlines(
        keepends=True
    )
    assert np.allclose(result.data, expected_cube.data)


def break_temp_name(temp_cube, rh_cube):
    """Modifies temp_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    temp_cube.rename("TEMP")
    return r"No cube named air_temperature found in .*"


def break_rh_name(temp_cube, rh_cube):
    """Modifies precip_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    rh_cube.rename("RH")
    return r"No cube named relative_humidity found in .*"

def break_reference_time(temp_cube, rh_cube):
    """Modifies temp_cube time points to be incremented by 1 second and
    returns the error message this will trigger"""
    temp_cube.coord("forecast_reference_time").points = (
        temp_cube.coord("forecast_reference_time").points + 1
    )
    return r".* and .* do not have the same forecast reference time"


def break_units(temp_cube, rh_cube):
    """Modifies rh_cube units to be incorrect as K and
    returns the error message this will trigger"""
    rh_cube.units = "K"
    return r"The .* units are incorrect, expected units as .* but received .*"


def break_coordinates(temp_cube, rh_cube):
    """Modifies the first latitude point on the temp_cube (adds one degree)
    and returns the error message this will trigger"""
    points = list(temp_cube.coord("latitude").points)
    points[0] = points[0] + 1
    temp_cube.coord("latitude").points = points
    return ".* and .* do not have the same spatial coordinates"

@pytest.mark.parametrize(
    "breaking_function",
    (
        break_temp_name,
        break_rh_name,
        break_reference_time,
        break_units,
        break_coordinates,
    ),
)
def test_exceptions(
    temp_cube, rh_cube, breaking_function
):
    """Tests that a suitable exception is raised when the  cube meta-data does
    not match what is expected"""
    error_msg = breaking_function(
        temp_cube, rh_cube
    )
    with pytest.raises(ValueError, match=error_msg):
        IcingSeverityMultivariateRegression_USAF2024()(
            CubeList([temp_cube, rh_cube])
        )

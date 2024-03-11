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
"""Test methods in lightning.LightningUSAF"""

from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.lightning import LightningMultivariateProbability
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture(name="cape_cube")
def cape_cube_fixture() -> Cube:
    """
    Set up a CAPE cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=4000)
    cube = set_up_variable_cube(
        data,
        name="atmosphere_specific_convective_available_potential_energy",
        units="J kg-1",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        grid_spacing=20,
    )
    return cube


@pytest.fixture(name="cin_cube")
def cin_cube_fixture() -> Cube:
    """
    Set up a liftidx cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """
    data = np.full((2, 7, 3), dtype=np.float32, fill_value=-0.25)
    cube = set_up_variable_cube(
        data,
        name="atmosphere_specific_convective_inhibition",
        units="J kg-1",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        grid_spacing=20,
    )
    return cube


@pytest.fixture(name="liftidx_cube")
def liftidx_cube_fixture() -> Cube:
    """
    Set up a liftidx cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=10)
    cube = set_up_variable_cube(
        data,
        name="temperature_difference_between_ambient_air_and_air_lifted_adiabatically",
        units="K",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        grid_spacing=20,
    )
    return cube


@pytest.fixture(name="pwat_cube")
def pwat_cube_fixture() -> Cube:
    """
    Set up a pwat cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=3)
    cube = set_up_variable_cube(
        data,
        name="precipitable_water",
        units="kg m-2",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        grid_spacing=20,
    )
    return cube


@pytest.fixture(name="apcp_cube")
def apcp_cube_fixture() -> Cube:
    """
    Set up a apcp cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=6)
    cube = set_up_variable_cube(
        data,
        name="precipitation_amount",
        units="kg m-2",
        time=datetime(2017, 11, 10, 6, 0),
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 6, 0)),
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        grid_spacing=20,
    )
    return cube


@pytest.fixture(name="expected_cube")
def expected_cube_fixture() -> Cube:
    """
    Set up the Lightning cube that we expect to get from the plugin
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=14.111012)
    cube = set_up_variable_cube(
        data,
        name="20_km_lightning_probability_over_the_valid_time_of_the_accumulated_precipitation",
        units="1",
        time=datetime(2017, 11, 10, 4, 30),
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 6, 0)),
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        grid_spacing=20,
    )
    return cube


def test_basic(cape_cube, liftidx_cube, pwat_cube, cin_cube, apcp_cube, expected_cube):
    """Run the plugin and check the result cube matches the expected_cube"""
    result = LightningMultivariateProbability()(
        CubeList([cape_cube, liftidx_cube, pwat_cube, cin_cube, apcp_cube]), None
    )
    assert np.allclose(result.data, expected_cube.data)

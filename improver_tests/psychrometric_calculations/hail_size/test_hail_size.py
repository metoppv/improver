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


import numpy as np
import pytest
from iris.cube import Cube

from improver.psychrometric_calculations.hail_size import HailSize
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture(name="ccl_temperature")
def ccl_temperature() -> Cube:
    """Set up a r, y, x cube of Cloud condensation level data"""
    data = np.full((2, 3, 2), fill_value=250, dtype=np.float32)
    ccl_temperature_cube = set_up_variable_cube(
        data,
        name="temperature_at_cloud_condensation_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return ccl_temperature_cube


@pytest.fixture(name="ccl_pressure")
def ccl_pressure() -> Cube:
    """Set up a r, y, x cube of Cloud condensation level data"""
    data = np.full((2, 3, 2), fill_value=85000, dtype=np.float32)
    ccl_pressure_cube = set_up_variable_cube(
        data,
        name="pressure_at_cloud_condensation_level",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return ccl_pressure_cube


@pytest.fixture(name="temperature_on_pressure_levels")
def t_cube_fixture() -> Cube:
    """Set up a r, p, y, x cube of Temperature on pressure level data"""
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


@pytest.fixture(name="humidity_mixing_ratio_on_pressure")
def humidity_cube_fixture() -> Cube:
    """Set up a r, p, y, x cube of Temperature on pressure level data"""
    temperatures = np.array(
        [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float32
    )
    data = np.broadcast_to(
        temperatures.reshape((1, len(temperatures), 1, 1)), (2, len(temperatures), 3, 2)
    )
    humidity_cube = set_up_variable_cube(
        data,
        pressure=True,
        height_levels=np.arange(100000, 29999, -10000),
        name="humidity_mixing_ratio_on_pressure_levels",
        units="kg/kg",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )

    return humidity_cube



@pytest.mark.parametrize(
    "ccl_p,ccl_t,humidity,expected",
    (
    (75000,290,0.001,20),# values approximate from literiture tephigram (https://journals.ametsoc.org/view/journals/bams/34/6/1520-0477-34_6_235.xml?tab_body=pdf)
    (94000,300,0.001,0), #vertical value negative
    (1000,360,0.001,0), #horizontal value negative
    (65000,300,0.001,15), #vertical grreater than table
    (150000,290,1.0,120),  #horizontal greater than table
    (80000,250,0.001,0), # ccl temperature below 268.15
    )
)
def test_basic_hail_size(
    ccl_pressure,
    ccl_temperature,
    temperature_on_pressure_levels,
    humidity_mixing_ratio_on_pressure,
    ccl_p,
    ccl_t,
    humidity,
    expected
):
    ccl_pressure.data=np.full_like(ccl_pressure.data,ccl_p)
    ccl_temperature.data=np.full_like(ccl_temperature.data,ccl_t)
    humidity_mixing_ratio_on_pressure.data=np.full_like(humidity_mixing_ratio_on_pressure.data,humidity)
    
    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        humidity_mixing_ratio_on_pressure,
    )

    np.testing.assert_allclose(result.data,expected)

    



# following values lead to iteration error

# @pytest.mark.parametrize(
#     "ccl_p,ccl_t",
#     (75000,330),
# )
# def test_error_hail_size(ccl_pressure,ccl_temperature,temperature_on_pressure_levels,humidity_mixing_ratio_on_pressure,ccl_p,ccl_t):

#     ccl_pressure.data=np.full_like(ccl_pressure.data,ccl_p)
#     ccl_temperature.data=np.full_like(ccl_temperature.data,ccl_t)

#     result = HailSize()(
#         ccl_temperature,
#         ccl_pressure,
#         temperature_on_pressure_levels,
#         humidity_mixing_ratio_on_pressure,
#     )

#     print(result.data)








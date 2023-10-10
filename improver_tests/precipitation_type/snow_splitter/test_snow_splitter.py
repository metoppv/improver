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
"""Tests for the SnowSplitter plugin"""
import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.precipitation_type.snow_splitter import SnowSplitter
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture()
def snow_cube() -> Cube:
    """Set up a r, y, x cube of probability of snow at surface"""
    data = np.full((2, 2, 2), fill_value=1, dtype=np.float32)
    snow_phase_prob_cube = set_up_variable_cube(
        data,
        name="probability_of_snow_at_surface",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return snow_phase_prob_cube


@pytest.fixture()
def rain_cube() -> Cube:
    """Set up a r, y, x cube of probability of rain at surface"""
    data = np.full((2, 2, 2), fill_value=0, dtype=np.float32)
    rain_phase_prob_cube = set_up_variable_cube(
        data,
        name="probability_of_rain_at_surface",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return rain_phase_prob_cube


@pytest.fixture()
def precip_rate_cube() -> Cube:
    """Set up a r, y, x cube of precipitation rate"""
    data = np.full((2, 2, 2), fill_value=1, dtype=np.float32)
    precip_cube = set_up_variable_cube(
        data,
        name="precipitation_rate",
        units="m/s",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return precip_cube


@pytest.mark.parametrize(
    "output_is_rain,cube_name", ((True, "rain_rate"), (False, "snow_rate"))
)
@pytest.mark.parametrize(
    "rain_value,snow_value,expected",
    ((1, 1, 0.5), (0, 1, "dependent"), (1, 0, "dependent")),
)
def test_basic(
    snow_cube,
    rain_cube,
    precip_rate_cube,
    snow_value,
    rain_value,
    output_is_rain,
    cube_name,
    expected,
):
    """Check that for all possible combinations of rain and snow probabilities the correct
    rain/snow rate is returned. The correct output will sometimes depend on what output_variable
    is requested. Also check the name of the returned cube has been updated correctly"""
    rain_cube.data = np.full_like(rain_cube.data, rain_value)
    snow_cube.data = np.full_like(snow_cube.data, snow_value)

    result = SnowSplitter(output_is_rain=output_is_rain)(
        CubeList([snow_cube, rain_cube, precip_rate_cube])
    )

    if expected == "dependent":
        expected = rain_value if output_is_rain else snow_value

    assert np.isclose(result.data, expected).all()
    assert result.name() == cube_name
    assert result.units == "m/s"
    assert result.attributes==LOCAL_MANDATORY_ATTRIBUTES


def test_both_phases_0(snow_cube, rain_cube, precip_rate_cube):
    """Test an error is raised if both snow and rain_cube have a probability of
    0"""

    rain_cube.data = np.full_like(rain_cube.data, 0)
    snow_cube.data = np.full_like(snow_cube.data, 0)
    with pytest.raises(ValueError, match="1 grid square where the probability of snow"):
        SnowSplitter(output_is_rain=False)(
            CubeList([snow_cube, rain_cube, precip_rate_cube])
        )
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
"""Tests for the CloudTopTemperature plugin"""

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations.cloud_top_temperature import (
    CloudTopTemperature,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture(name="ccl")
def ccl_cube_fixture() -> Cube:
    """Set up a r, y, x cube of Cloud condensation level data with pressure coordinate"""
    data = np.zeros((2, 2, 2), dtype=np.float32)
    ccl_cube = set_up_variable_cube(
        data,
        name="temperature_at_cloud_condensation_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    ccl_cube.add_aux_coord(
        AuxCoord(standard_name="air_pressure", points=data.copy(), units="Pa"),
        (0, 1, 2),
    )
    return ccl_cube


@pytest.fixture(name="temperature")
def t_cube_fixture(profile_shift) -> Cube:
    """Set up a r, p, y, x cube of Temperature on pressure level data"""
    temperatures = np.array([300, 286, 280, 274, 267, 262, 257, 245], dtype=np.float32)
    data = np.broadcast_to(
        temperatures.reshape((1, len(temperatures), 1, 1)), (2, len(temperatures), 2, 2)
    )
    t_cube = set_up_variable_cube(
        data + profile_shift,
        pressure=True,
        height_levels=np.arange(100000, 29999, -10000),
        name="temperature_on_pressure_levels",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return t_cube


def metadata_ok(cct: Cube, baseline: Cube, model_id_attr=None) -> None:
    """
    Checks convective cloud top temperature Cube long_name, units and dtype are as expected.
    Compares with baseline to make sure everything else matches.

    Args:
        cct: Result of ConvectiveCloudTop plugin
        baseline: A cube with the expected coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    assert cct.long_name == "temperature_at_convective_cloud_top"
    assert cct.units == "K"
    assert cct.dtype == np.float32
    for coord in cct.coords():
        base_coord = baseline.coord(coord.name())
        assert cct.coord_dims(coord) == baseline.coord_dims(base_coord)
        assert coord == base_coord
    for attr in MANDATORY_ATTRIBUTES:
        assert cct.attributes[attr] == baseline.attributes[attr]
    all_attr_keys = list(cct.attributes.keys())
    if model_id_attr:
        assert cct.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
    else:
        mandatory_attr_keys = all_attr_keys
    assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


@pytest.mark.parametrize(
    "ccl_t, ccl_p, profile_shift, expected",
    (
        (290, 95000, 0, 264.575),
        (288.12, 90000, 0, 264.575),
        (290, 95000, -4, 254.698),
        (288.17, 90000, -4, 254.698),
        (286, 91000, 0, False),
    ),
)
def test_basic(ccl, temperature, ccl_t, ccl_p, expected):
    """
    When the profile is shifted to be colder, the same saturated ascents
    reach a higher, and therefore colder, level.
    The different CCL values show that starting at a higher point on the same
    profile has no impact on the final result.
    The last test is for a case where convection does not occur and the result is masked.
    """
    ccl.data = np.full_like(ccl.data, ccl_t)
    ccl.coord("air_pressure").points = np.full_like(
        ccl.coord("air_pressure").points, fill_value=ccl_p
    )
    result = CloudTopTemperature()(ccl, temperature)
    metadata_ok(result, ccl)
    if expected:
        assert not result.data.mask.any()
        np.testing.assert_allclose(result.data, expected, atol=1e-2)
    else:
        assert result.data.mask.all()


@pytest.mark.parametrize("profile_shift", (0,))
@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(ccl, temperature, model_id_attr):
    """Check that tests pass if model_id_attr is set on inputs and is applied or not"""
    ccl.data = np.full_like(ccl.data, 290)
    ccl.coord("air_pressure").points = np.full_like(
        ccl.coord("air_pressure").points, fill_value=95000
    )
    ccl.attributes["mosg__model_configuration"] = "gl_ens"
    temperature.attributes["mosg__model_configuration"] = "gl_ens"
    result = CloudTopTemperature(model_id_attr=model_id_attr)(ccl, temperature)
    metadata_ok(result, ccl, model_id_attr=model_id_attr)

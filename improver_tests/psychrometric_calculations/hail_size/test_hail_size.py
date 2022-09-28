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
from iris.cube import Cube, CubeList

from improver.psychrometric_calculations.hail_size import HailSize
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture
def ccl_temperature() -> Cube:
    """Set up a r, y, x cube of cloud condensation level temperature data"""
    data = np.full((2, 3, 2), fill_value=330, dtype=np.float32)
    ccl_temperature_cube = set_up_variable_cube(
        data,
        name="temperature_at_cloud_condensation_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return ccl_temperature_cube


@pytest.fixture
def ccl_pressure() -> Cube:
    """Set up a r, y, x cube of cloud condensation level pressure data"""
    data = np.full((2, 3, 2), fill_value=120000, dtype=np.float32)
    ccl_pressure_cube = set_up_variable_cube(
        data,
        name="pressure_at_cloud_condensation_level",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return ccl_pressure_cube

@pytest.fixture
def wet_bulb_freezing() -> Cube:
    """Set up a r, y, x cube of wet bulb freezing height data"""
    data = np.full((2, 3, 2), fill_value=2000, dtype=np.float32)
    ccl_pressure_cube = set_up_variable_cube(
        data,
        name="wet_bulb_freezing_level_altitude",
        units="m",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return ccl_pressure_cube


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


@pytest.fixture
def relative_humidity_on_pressure() -> Cube:
    """Set up a r, p, y, x cube of relative_humidity on pressure levels"""
    humidity = np.repeat(0.1, 8).astype("float32")
    data = np.broadcast_to(
        humidity.reshape((1, len(humidity), 1, 1)), (2, len(humidity), 3, 2)
    )
    humidity_cube = set_up_variable_cube(
        data,
        pressure=True,
        height_levels=np.arange(100000, 29999, -10000),
        name="relative_humidity_on_pressure_levels",
        units="kg/kg",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )

    return humidity_cube


def metadata_check(hail_cube):
    """Checks the hail cube produced by plugin has the expected metadata."""
    assert hail_cube.long_name == "diameter_of_hail_stones"
    assert hail_cube.units == "m"

    attributes = [attr for attr in hail_cube.attributes]

    if "mosg__model_configuration" in attributes:
        assert hail_cube.attributes == {
            "title": "unit test data",
            "source": "unit test",
            "institution": "somewhere",
            "mosg__model_configuration": "gl_ens",
        }
    else:
        assert hail_cube.attributes == {
            "title": "unit test data",
            "source": "unit test",
            "institution": "somewhere",
        }


def cube_shape_check(hail_cube):
    """Checks cube coordinates and dimensions"""
    coord_names = [coord.name() for coord in hail_cube.coords()]
    assert coord_names == [
        "realization",
        "latitude",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "time",
    ]
    assert hail_cube.shape == (2, 3, 2)


"""literature tephigram link
Fawbush, E.J., and R.C. Miller. 1953.
“A method for forecasting hailstone size at the earth’s surface.”
Bulletin of the American Meteorological Society 34: 235-244.
https://doi.org/10.1175/1520-0477-34.6.235
"""


@pytest.mark.parametrize(
    "ccl_p,ccl_t,humidity,wbz,expected",
    (
        (75000, 290, 0.001,2200, 0.02),  # values approx from tephigram in literature
        (75000, 290, 0.001,5000, 0), #wet bulb zero height above 4400m
        (75000, 290, 0.001,3400, 0.015), #wet bulb zero height above 3350m but less than 4400m
        (94000, 300, 0.001,2200, 0),  # vertical value negative
        (1000, 360, 0.001,2200, 0),  # horizontal value negative
        (95000, 330, 0.001,2200, 0.08),  # vertical greater than length of table
        (150000, 350, 0.1,2200, 0.025),  # horizontal greater than length of table
        (75000, 265, 0.001,2200, 0),  # ccl temperature below 268.15
    ),
)
def test_basic_hail_size(
    ccl_pressure,
    ccl_temperature,
    temperature_on_pressure_levels,
    relative_humidity_on_pressure,
    wet_bulb_freezing,
    ccl_p,
    ccl_t,
    humidity,
    wbz,
    expected,
):
    """Tests the hail_size plugin with values for ccl temperature, ccl pressure,
    wet_bulb_freezing_height and relative humidity to check for expected result.
    Also checks the metadata of the produced hail_size cube"""
    ccl_pressure.data = np.full_like(ccl_pressure.data, ccl_p)
    ccl_temperature.data = np.full_like(ccl_temperature.data, ccl_t)
    relative_humidity_on_pressure.data = np.full_like(
        relative_humidity_on_pressure.data, humidity
    )
    wet_bulb_freezing.data = np.full_like(wet_bulb_freezing.data, wbz)

    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        relative_humidity_on_pressure,
        wet_bulb_freezing,
    )
    np.testing.assert_array_almost_equal(result.data, expected)
    metadata_check(result)
    cube_shape_check(result)


def test_temperature_too_high(
    temperature_on_pressure_levels,
    ccl_pressure,
    ccl_temperature,
    relative_humidity_on_pressure,
    wet_bulb_freezing,
):
    """Tests for the case where there are grid squares where the temperature
    doesn't drop below 268.15K at any pressure. At these points hail size
    should be set to zero"""
    temperature_on_pressure_levels.data = np.full_like(
        temperature_on_pressure_levels.data, 260
    )
    temperature_on_pressure_levels.data[:, :, 1] = 300
    expected = [
        [[0.035, 0.035], [0, 0], [0.035, 0.035]],
        [[0.035, 0.035], [0, 0], [0.035, 0.035]],
    ]

    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        relative_humidity_on_pressure,
        wet_bulb_freezing,
    )
    np.testing.assert_array_almost_equal(result.data, expected)
    metadata_check(result)
    cube_shape_check(result)


@pytest.mark.parametrize(
    "variable",
    (
        "temperature_on_pressure_levels",
        "ccl_temperature",
        "ccl_pressure",
        "relative_humidity_on_pressure",
        "wet_bulb_freezing"
    ),
)
def test_spatial_coord_mismatch(variable, request):
    """Tests that an error is raised if the spatial
    coordinates of all cubes don't match"""

    variable_new = request.getfixturevalue(variable)
    variable_slice = next(variable_new.slices_over("longitude"))

    fixtures = [
        "ccl_temperature",
        "ccl_pressure",
        "temperature_on_pressure_levels",
        "relative_humidity_on_pressure",
        "wet_bulb_freezing",
    ]
    fixtures.remove(variable)
    cubes = CubeList(request.getfixturevalue(fix) for fix in fixtures)
    cubes.append(variable_slice)

    ccl_temperature = cubes.extract("temperature_at_cloud_condensation_level")
    ccl_pressure = cubes.extract("pressure_at_cloud_condensation_level")
    temperature_on_pressure = cubes.extract("temperature_on_pressure_levels")
    relative_humidity_on_pressure = cubes.extract(
        "relative_humidity_on_pressure_levels"
    )
    wet_bulb_freezing = cubes.extract("wet_bulb_freezing_level_altitude")

    with pytest.raises(ValueError):
        HailSize()(
            ccl_temperature[0],
            ccl_pressure[0],
            temperature_on_pressure[0],
            relative_humidity_on_pressure[0],
            wet_bulb_freezing[0],
        )


@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(
    temperature_on_pressure_levels,
    ccl_pressure,
    relative_humidity_on_pressure,
    ccl_temperature,
    wet_bulb_freezing,
    model_id_attr,
):
    """Tests plugin if model_id_attr is set on inputs and is applied or not"""
    temperature_on_pressure_levels.attributes["mosg__model_configuration"] = "gl_ens"
    ccl_pressure.attributes["mosg__model_configuration"] = "gl_ens"
    relative_humidity_on_pressure.attributes["mosg__model_configuration"] = "gl_ens"
    ccl_temperature.attributes["mosg__model_configuration"] = "gl_ens"
    wet_bulb_freezing.attributes["mosg__model_configuration"] = "gl_ens"

    result = HailSize(model_id_attr=model_id_attr)(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        relative_humidity_on_pressure,
        wet_bulb_freezing
    )

    np.testing.assert_array_almost_equal(result.data, 0.035)
    metadata_check(result)
    cube_shape_check(result)


def test_re_ordered_cubes(
    temperature_on_pressure_levels,
    ccl_pressure,
    relative_humidity_on_pressure,
    ccl_temperature,
    wet_bulb_freezing,
):

    """Tests the plugin if the input cubes have coordinates that need to be rearranged.
    Checks that the outputted cube has coordinates in the same order as the inputs"""

    enforce_coordinate_ordering(
        temperature_on_pressure_levels,
        ["pressure", "latitude", "realization", "longitude"],
    )
    enforce_coordinate_ordering(
        relative_humidity_on_pressure,
        ["pressure", "latitude", "realization", "longitude"],
    )
    enforce_coordinate_ordering(ccl_pressure, ["latitude", "realization", "longitude"])
    enforce_coordinate_ordering(
        ccl_temperature, ["latitude", "realization", "longitude"]
    )
    enforce_coordinate_ordering(
        wet_bulb_freezing, ["latitude", "realization", "longitude"]
    )
    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        relative_humidity_on_pressure,
        wet_bulb_freezing
    )
    np.testing.assert_array_almost_equal(result.data, 0.035)
    metadata_check(result)
    coord_names = [coord.name() for coord in result.coords()]
    assert coord_names == [
        "latitude",
        "realization",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "time",
    ]
    assert result.shape == (3, 2, 2)


def test_no_realization_coordinate(
    temperature_on_pressure_levels,
    ccl_pressure,
    relative_humidity_on_pressure,
    ccl_temperature,
    wet_bulb_freezing
):
    """Test plugin if input cubes don't have a realization coordinate"""

    temp = next(temperature_on_pressure_levels.slices_over("realization"))
    temp.remove_coord("realization")
    humidity = next(relative_humidity_on_pressure.slices_over("realization"))
    humidity.remove_coord("realization")

    cloud_pressure = next(ccl_pressure.slices_over("realization"))
    cloud_pressure.remove_coord("realization")

    cloud_temp = next(ccl_temperature.slices_over("realization"))
    cloud_temp.remove_coord("realization")

    wet_bulb_zero = next(wet_bulb_freezing.slices_over("realization"))
    wet_bulb_zero.remove_coord("realization")

    result = HailSize()(cloud_temp, cloud_pressure, temp, humidity,wet_bulb_zero)
    np.testing.assert_array_almost_equal(result.data, 0.035)
    metadata_check(result)
    coord_names = [coord.name() for coord in result.coords()]
    assert coord_names == [
        "latitude",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "time",
    ]
    assert result.shape == (3, 2)

# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the HailSize plugin"""

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
pytest.importorskip("stratify")


@pytest.fixture
def ccl_temperature() -> Cube:
    """Set up a r, y, x cube of cloud condensation level temperature data"""
    data = np.full((2, 3, 2), fill_value=300, dtype=np.float32)
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
    data = np.full((2, 3, 2), fill_value=97500, dtype=np.float32)
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
    wet_bulb_freezing_cube = set_up_variable_cube(
        data,
        name="wet_bulb_freezing_level_altitude",
        units="m",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return wet_bulb_freezing_cube


@pytest.fixture
def orography() -> Cube:
    """Set up a r, y, x cube of orography data"""
    data = np.full((3, 2), fill_value=0, dtype=np.float32)
    orography_cube = set_up_variable_cube(
        data, name="surface_altitude", units="m", attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return orography_cube


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
    "ccl_p,ccl_t,wbz,orog,expected",
    (
        (75000, 290, 2200, 0, 0.035,),  # values approx from tephigram in literature
        (75000, 290, 5000, 0, 0,),  # wet bulb zero (wbz) height greater than 4400m
        (75000, 290, 5000, 2800, 0.035),  # orography reduces wbz height to 2200m
        (75000, 290, 3400, 0, 0.025,),  # wbz height above 3350m but less than 4400m
        (94000, 273, 2200, 0, 0),  # vertical value negative
        (1000, 270, 2200, 0, 0),  # horizontal value negative
        (95000, 330, 2200, 0, 0.08),  # vertical greater than length of table
        (150000, 350, 2200, 0, 0.12),  # horizontal greater than length of table
        (75000, 265, 2200, 0, 0),  # ccl temperature below 268.15
    ),
)
def test_basic_hail_size(
    ccl_pressure,
    ccl_temperature,
    temperature_on_pressure_levels,
    wet_bulb_freezing,
    orography,
    ccl_p,
    ccl_t,
    wbz,
    orog,
    expected,
):
    """Tests the hail_size plugin with values for ccl temperature, ccl pressure,
    and wet_bulb_freezing_height to check for expected result.
    Also checks the metadata of the produced hail_size cube"""
    ccl_pressure.data[..., 0, 0] = ccl_p
    ccl_temperature.data[..., 0, 0] = ccl_t
    wet_bulb_freezing.data[..., 0, 0] = wbz
    orography.data[0, 0] = orog
    expected_data = np.full_like(ccl_temperature.data, 0.1)
    expected_data[..., 0, 0] = expected

    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        wet_bulb_freezing,
        orography,
    )
    np.testing.assert_array_almost_equal(result.data, expected_data)
    metadata_check(result)
    cube_shape_check(result)


def test_temperature_too_high(
    temperature_on_pressure_levels,
    ccl_pressure,
    ccl_temperature,
    wet_bulb_freezing,
    orography,
):
    """Tests for the case where there are grid squares where the temperature
    doesn't drop below 268.15K at any pressure. At these points hail size
    should be set to zero"""
    data = temperature_on_pressure_levels.data.copy()
    data[:, :, 1] = 300
    temperature_on_pressure_levels.data = data
    expected = [
        [[0.1, 0.1], [0, 0], [0.1, 0.1]],
        [[0.1, 0.1], [0, 0], [0.1, 0.1]],
    ]

    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        wet_bulb_freezing,
        orography,
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
        "wet_bulb_freezing",
        "orography",
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
        "wet_bulb_freezing",
        "orography",
    ]
    fixtures.remove(variable)
    cubes = CubeList(request.getfixturevalue(fix) for fix in fixtures)
    cubes.append(variable_slice)

    (ccl_temperature,) = cubes.extract("temperature_at_cloud_condensation_level")
    (ccl_pressure,) = cubes.extract("pressure_at_cloud_condensation_level")
    (temperature_on_pressure,) = cubes.extract("temperature_on_pressure_levels")
    (wet_bulb_freezing,) = cubes.extract("wet_bulb_freezing_level_altitude")
    (orography,) = cubes.extract("surface_altitude")

    with pytest.raises(ValueError):
        HailSize()(
            ccl_temperature,
            ccl_pressure,
            temperature_on_pressure,
            wet_bulb_freezing,
            orography,
        )


@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(
    temperature_on_pressure_levels,
    ccl_pressure,
    ccl_temperature,
    wet_bulb_freezing,
    orography,
    model_id_attr,
):
    """Tests plugin if model_id_attr is set on inputs and is applied or not"""
    temperature_on_pressure_levels.attributes["mosg__model_configuration"] = "gl_ens"
    ccl_pressure.attributes["mosg__model_configuration"] = "gl_ens"
    ccl_temperature.attributes["mosg__model_configuration"] = "gl_ens"
    wet_bulb_freezing.attributes["mosg__model_configuration"] = "gl_ens"
    orography.attributes["mosg__model_configuration"] = "gl_ens"

    result = HailSize(model_id_attr=model_id_attr)(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        wet_bulb_freezing,
        orography,
    )

    np.testing.assert_array_almost_equal(result.data, 0.1)
    metadata_check(result)
    cube_shape_check(result)


def test_re_ordered_cubes(
    temperature_on_pressure_levels,
    ccl_pressure,
    ccl_temperature,
    wet_bulb_freezing,
    orography,
):

    """Tests the plugin if the input cubes have coordinates that need to be rearranged.
    Checks that the outputted cube has coordinates in the same order as the inputs"""

    enforce_coordinate_ordering(
        temperature_on_pressure_levels,
        ["pressure", "latitude", "realization", "longitude"],
    )
    enforce_coordinate_ordering(ccl_pressure, ["latitude", "realization", "longitude"])
    enforce_coordinate_ordering(
        ccl_temperature, ["latitude", "realization", "longitude"]
    )
    enforce_coordinate_ordering(
        wet_bulb_freezing, ["latitude", "realization", "longitude"]
    )
    enforce_coordinate_ordering(orography, ["latitude", "longitude"])

    result = HailSize()(
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure_levels,
        wet_bulb_freezing,
        orography,
    )
    np.testing.assert_array_almost_equal(result.data, 0.1)
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
    ccl_temperature,
    wet_bulb_freezing,
    orography,
):
    """Test plugin if input cubes don't have a realization coordinate"""

    temp = next(temperature_on_pressure_levels.slices_over("realization"))
    temp.remove_coord("realization")

    cloud_pressure = next(ccl_pressure.slices_over("realization"))
    cloud_pressure.remove_coord("realization")

    cloud_temp = next(ccl_temperature.slices_over("realization"))
    cloud_temp.remove_coord("realization")

    wet_bulb_zero = next(wet_bulb_freezing.slices_over("realization"))
    wet_bulb_zero.remove_coord("realization")

    result = HailSize()(cloud_temp, cloud_pressure, temp, wet_bulb_zero, orography)
    np.testing.assert_array_almost_equal(result.data, 0.1)
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

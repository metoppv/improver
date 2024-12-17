# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the CloudTopTemperature plugin"""

import copy
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations import cloud_top_temperature
from improver.psychrometric_calculations.cloud_top_temperature import (
    CloudTopTemperature,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}
POST_PROCESSED_MANDATORY_ATTRIBUTES = copy.deepcopy(LOCAL_MANDATORY_ATTRIBUTES)
POST_PROCESSED_MANDATORY_ATTRIBUTES["title"] = (
    f"Post-Processed {POST_PROCESSED_MANDATORY_ATTRIBUTES['title']}"
)


class HaltExecution(Exception):
    pass


@patch("improver.psychrometric_calculations.cloud_top_temperature.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        CloudTopTemperature(model_id_attr=sentinel.model_id_attr)(
            sentinel.t_at_ccl, sentinel.p_at_ccl, sentinel.temperature
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.t_at_ccl, sentinel.p_at_ccl, sentinel.temperature
    )


@pytest.fixture(name="t_at_ccl")
def t_at_ccl_cube_fixture() -> Cube:
    """Set up a r, y, x cube of temperature at Cloud condensation level data"""
    data = np.full((2, 2, 2), fill_value=290, dtype=np.float32)
    ccl_cube = set_up_variable_cube(
        data,
        name="air_temperature_at_condensation_level",
        units="K",
        attributes=POST_PROCESSED_MANDATORY_ATTRIBUTES,
    )
    return ccl_cube


@pytest.fixture(name="p_at_ccl")
def p_at_ccl_cube_fixture() -> Cube:
    """Set up a r, y, x cube of pressure at Cloud condensation level data"""
    data = np.full((2, 2, 2), fill_value=95000, dtype=np.float32)
    ccl_cube = set_up_variable_cube(
        data,
        name="air_pressure_at_condensation_level",
        units="Pa",
        attributes=POST_PROCESSED_MANDATORY_ATTRIBUTES,
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
        vertical_levels=np.arange(100000, 29999, -10000),
        name="air_temperature",
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
    assert cct.long_name == "air_temperature_at_convective_cloud_top"
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
def test_basic(t_at_ccl, p_at_ccl, temperature, ccl_t, ccl_p, expected):
    """
    When the profile is shifted to be colder, the same saturated ascents
    reach a higher, and therefore colder, level.
    The different CCL values show that starting at a higher point on the same
    profile has no impact on the final result.
    The last test is for a case where convection does not occur and the result is masked.
    """
    t_at_ccl.data = np.full_like(t_at_ccl.data, ccl_t)
    p_at_ccl.data = np.full_like(p_at_ccl.data, ccl_p)
    result = CloudTopTemperature()(t_at_ccl, p_at_ccl, temperature)
    metadata_ok(result, t_at_ccl)
    if expected:
        assert not result.data.mask.any()
        np.testing.assert_allclose(result.data, expected, atol=1e-2)
    else:
        assert result.data.mask.all()


@pytest.mark.parametrize("profile_shift", (0,))
@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(t_at_ccl, p_at_ccl, temperature, model_id_attr):
    """Check that tests pass if model_id_attr is set on inputs and is applied or not"""
    t_at_ccl.attributes["mosg__model_configuration"] = "gl_ens"
    p_at_ccl.attributes["mosg__model_configuration"] = "gl_ens"
    temperature.attributes["mosg__model_configuration"] = "gl_ens"
    result = CloudTopTemperature(model_id_attr=model_id_attr)(
        t_at_ccl, p_at_ccl, temperature
    )
    metadata_ok(result, t_at_ccl, model_id_attr=model_id_attr)


@pytest.mark.parametrize("profile_shift", (0,))
def test_called_methods(t_at_ccl, p_at_ccl, temperature):
    """Prove that the methods to set units and check spatial consistency have been called"""
    with patch.object(
        cloud_top_temperature, "assert_spatial_coords_match"
    ) as spatial_mock:
        with patch.object(t_at_ccl, "convert_units") as t_at_ccl_mock:
            with patch.object(p_at_ccl, "convert_units") as p_at_ccl_mock:
                with patch.object(temperature, "convert_units") as temperature_mock:
                    CloudTopTemperature()(t_at_ccl, p_at_ccl, temperature)
    spatial_mock.assert_called()
    t_at_ccl_mock.assert_called_with("K")
    p_at_ccl_mock.assert_called_with("Pa")
    temperature_mock.assert_called_with("K")

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
"""Tests for the HailFraction plugin."""

import iris
import numpy as np
import pytest

from improver.precipitation_type.hail_fraction import HailFraction
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "Post-Processed IMPROVER unit test",
}


def setup_cubes():
    """Set up cubes for testing."""
    vertical_updraught_data = np.zeros((2, 2), dtype=np.float32)
    vertical_updraught = set_up_variable_cube(
        vertical_updraught_data,
        "maximum_vertical_updraught",
        "m s-1",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )

    hail_size_data = np.zeros((2, 2), dtype=np.float32)
    hail_size = set_up_variable_cube(
        hail_size_data,
        "size_of_hail_stones",
        "m",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )

    cloud_condensation_level_data = np.zeros((2, 2), dtype=np.float32)
    cloud_condensation_level = set_up_variable_cube(
        cloud_condensation_level_data,
        "air_temperature_at_condensation_level",
        "K",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )

    convective_cloud_top_data = np.zeros((2, 2), dtype=np.float32)
    convective_cloud_top = set_up_variable_cube(
        convective_cloud_top_data,
        "air_temperature_at_convective_cloud_top",
        "K",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )

    hail_melting_level_data = np.zeros((2, 2), dtype=np.float32)
    hail_melting_level = set_up_variable_cube(
        hail_melting_level_data,
        "altitude_of_rain_from_hail_falling_level",
        "m",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )

    altitude_data = np.zeros((2, 2), dtype=np.float32)
    altitude = set_up_variable_cube(
        altitude_data,
        "altitude",
        "m",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )
    return (
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    )


@pytest.mark.parametrize("model_id_attr", (None, "mosg__model_configuration"))
@pytest.mark.parametrize(
    "vertical_updraught_value,hail_size_value,cloud_condensation_level_value,"
    + "convective_cloud_top_value,hail_melting_level_value,altitude_value,expected",
    (
        # No indications of hail
        (1, 0.0, 263.15, 261.15, 100, 50, 0),
        # Larger updraught, no other indications of hail
        (25, 0.001, 263.15, 261.15, 100, 50, 0),
        # Low vertical updraught prevents hail
        (5, 0.001, 271.15, 253.15, 20, 50, 0),
        # Sufficient vertical updraught, non-zero hail fraction
        (25, 0.001, 271.15, 253.15, 20, 50, 1 / 9),
        # Sufficient vertical updraught, non-zero hail fraction
        (50, 0.001, 271.15, 253.15, 20, 50, 0.25),
        # Large vertical updraught, non-zero hail fraction
        (75, 0.001, 271.15, 253.15, 20, 50, 0.25),
        # Hail size indicates non-zero hail fraction
        (1, 0.003, 271.15, 253.15, 20, 50, 0.05),
        # Cloud condensation level temperature prevents hail
        (75, 0.001, 263.15, 253.15, 20, 50, 0),
        # Convective cloud top temperature prevents hail
        (75, 0.001, 271.15, 263.15, 20, 50, 0),
        # Hail melting level prevents hail
        (75, 0.001, 271.15, 253.15, 100, 50, 0),
        # Hail size causes non-zero hail fraction despite inhibitive cloud condensation
        # level temperature, convective cloud top temperature and hail melting level
        (1, 0.003, 263.15, 263.15, 100, 50, 0.05),
    ),
)
def test_basic(
    vertical_updraught_value,
    hail_size_value,
    cloud_condensation_level_value,
    convective_cloud_top_value,
    hail_melting_level_value,
    altitude_value,
    expected,
    model_id_attr,
):
    """Test hail fraction plugin."""
    expected_attributes = COMMON_ATTRS.copy()
    if model_id_attr:
        expected_attributes[model_id_attr] = setup_cubes()[0].attributes[model_id_attr]

    (
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    ) = setup_cubes()

    vertical_updraught.data = np.full_like(
        vertical_updraught.data, vertical_updraught_value
    )
    hail_size.data = np.full_like(hail_size.data, hail_size_value)
    cloud_condensation_level.data = np.full_like(
        cloud_condensation_level.data, cloud_condensation_level_value,
    )
    convective_cloud_top.data = np.full_like(
        convective_cloud_top.data, convective_cloud_top_value,
    )
    hail_melting_level.data = np.full_like(
        hail_melting_level.data, hail_melting_level_value
    )
    altitude.data = np.full_like(altitude.data, altitude_value)

    result = HailFraction(model_id_attr=model_id_attr)(
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    )
    assert isinstance(result, iris.cube.Cube)
    assert str(result.units) == "1"
    assert result.name() == "hail_fraction"
    assert result.attributes == expected_attributes
    np.testing.assert_allclose(result.data, expected)

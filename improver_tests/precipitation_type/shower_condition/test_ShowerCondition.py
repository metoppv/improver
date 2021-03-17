# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for ShowerCondition plugin"""

import numpy as np
import pytest
from iris.cube import CubeList

from improver.metadata.constants import FLOAT_DTYPE
from improver.precipitation_type.shower_condition import ShowerCondition
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube

# Set up probability data to boundary-test around 0.5
PROBABILITY_DATA = np.array(
    [
        np.ones((3, 3)),
        [[0.2, 0.2, 0.3], [0.4, 0.49, 0.5], [0.5, 0.51, 0.7]],
        np.zeros((3, 3)),
    ],
    dtype=FLOAT_DTYPE,
)

EXPECTED_UK = [[0, 0, 0], [0, 0, 1], [1, 1, 1]]

EXPECTED_GLOBAL = [[0, 0, 0], [0, 0, 0], [1, 1, 1]]


@pytest.fixture(name="cloud_texture_cube")
def cloud_texture_fixture():
    """Probability of cloud texture above threshold cube"""
    thresholds = np.array([0, 0.05, 0.1], dtype=FLOAT_DTYPE)
    name = "texture_of_low_and_medium_type_cloud_area_fraction"
    return set_up_probability_cube(
        PROBABILITY_DATA,
        thresholds,
        variable_name=name,
        threshold_units="1",
        spatial_grid="equalarea",
    )


@pytest.fixture(name="below_threshold_cube")
def below_threshold_fixture():
    """Probability of cloud texture below threshold cube"""
    thresholds = np.array([0.1, 0.05, 0], dtype=FLOAT_DTYPE)
    name = "texture_of_low_and_medium_type_cloud_area_fraction"
    return set_up_probability_cube(
        PROBABILITY_DATA,
        thresholds,
        variable_name=name,
        threshold_units="1",
        spatial_grid="equalarea",
        spp__relative_to_threshold="less_than",
    )


@pytest.fixture(name="cloud_cube")
def cloud_fixture():
    """Probability of cloud above threshold cube"""
    thresholds = np.array([0.5, 0.8125, 1.0], dtype=FLOAT_DTYPE)
    name = "low_and_medium_type_cloud_area_fraction"
    return set_up_probability_cube(
        np.flip(PROBABILITY_DATA, axis=1),
        thresholds,
        variable_name=name,
        threshold_units="1",
        spatial_grid="equalarea",
    )


@pytest.fixture(name="conv_ratio_cube")
def conv_ratio_fixture():
    """Probability of convective ratio above threshold cube"""
    thresholds = np.array([0.5, 0.8, 1.0], dtype=FLOAT_DTYPE)
    name = "convective_ratio"
    return set_up_probability_cube(
        PROBABILITY_DATA,
        thresholds,
        variable_name=name,
        threshold_units="1",
        spatial_grid="equalarea",
    )


def test_basic(cloud_texture_cube):
    """Test output type and metadata"""
    expected_dims = ["projection_y_coordinate", "projection_x_coordinate"]
    expected_aux = {
        coord.name() for coord in cloud_texture_cube.coords(dim_coords=False)
    }
    expected_attributes = {
        "institution": "unknown",
        "source": "IMPROVER",
        "title": "unknown",
    }
    cubes = [cloud_texture_cube]
    result = ShowerCondition()(CubeList(cubes))
    dim_coords = [coord.name() for coord in result.coords(dim_coords=True)]
    aux_coords = {coord.name() for coord in result.coords(dim_coords=False)}

    assert result.name() == "precipitation_is_showery"
    assert result.units == "1"
    assert result.shape == (3, 3)
    assert result.data.dtype == FLOAT_DTYPE
    assert aux_coords == expected_aux
    assert dim_coords == expected_dims
    assert result.attributes == expected_attributes


def test_uk_tree(cloud_texture_cube):
    """Test correct shower diagnosis using UK decision tree"""
    cubes = [cloud_texture_cube]
    result = ShowerCondition()(CubeList(cubes))
    np.testing.assert_allclose(result.data, EXPECTED_UK)


def test_global_tree(cloud_cube, conv_ratio_cube):
    """Test correct shower diagnosis using global decision tree"""
    cubes = [cloud_cube, conv_ratio_cube]
    result = ShowerCondition()(CubeList(cubes))
    np.testing.assert_allclose(result.data, EXPECTED_GLOBAL)


def test_too_many_inputs(cloud_texture_cube, cloud_cube, conv_ratio_cube):
    """Test default behaviour using UK tree if all fields are provided"""
    cubes = [cloud_texture_cube, cloud_cube, conv_ratio_cube]
    result = ShowerCondition()(CubeList(cubes))
    np.testing.assert_allclose(result.data, EXPECTED_UK)


def test_too_few_inputs(cloud_cube):
    """Test error if too few inputs are provided"""
    cubes = [cloud_cube]
    with pytest.raises(ValueError, match="Incomplete inputs"):
        ShowerCondition()(CubeList(cubes))


def test_missing_threshold(cloud_texture_cube):
    """Test error if the required threshold is missing"""
    cubes = [cloud_texture_cube[0]]
    with pytest.raises(ValueError, match="contain required threshold"):
        ShowerCondition()(CubeList(cubes))

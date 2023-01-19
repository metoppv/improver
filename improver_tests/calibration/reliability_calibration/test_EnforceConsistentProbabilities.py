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
"""Unit tests for the EnforceConsistentProbabilities plugin"""

import numpy as np
import pytest
from iris.cube import Cube

from improver.calibration.reliability_calibration import EnforceConsistentProbabilities
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture
def hybrid_cloud() -> Cube:
    """Set up a probability cube of hybrid cloud data"""
    data = np.full((2, 3, 2), fill_value=0.3, dtype=np.float32)
    data[:, 1, 1] = 0.6
    hybrid_cloud = set_up_probability_cube(
        data,
        variable_name="low_and_medium_type_cloud_area_fraction",
        thresholds=[0.3125, 0.5],
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return hybrid_cloud


@pytest.fixture
def total_cloud() -> Cube:
    """Set up a probability cube of total cloud data"""
    data = np.full((2, 3, 2), fill_value=0.5, dtype=np.float32)
    total_cloud = set_up_probability_cube(
        data,
        variable_name="cloud_area_fraction",
        thresholds=[0.3125, 0.5],
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return total_cloud


@pytest.mark.parametrize(
    "hybrid_value,total_value",
    (
        (0.3, 0.4),  # hybrid probability < total probability
        (0.4, 0.3),  # total_probability < hybrid probability
        (None, None),  # one data point in hybrid > total _probability
    ),
)
def test_basic_probability_enforcement(
    hybrid_cloud, total_cloud, hybrid_value, total_value
):
    """Tests the enforce_consistent_probabilities plugin when making a hybrid cloud
    forecast consistent with a reference forecast of total cloud."""

    if hybrid_value is None:
        expected_data = np.full_like(hybrid_cloud.data, 0.3)
        expected_data[:, 1, 1] = 0.5
    else:
        hybrid_cloud.data = np.full_like(hybrid_cloud.data, hybrid_value)
        total_cloud.data = np.full_like(total_cloud.data, total_value)
        expected_data = np.full_like(
            hybrid_cloud.data, np.amin([hybrid_value, total_value])
        )

    result = EnforceConsistentProbabilities()(hybrid_cloud, total_cloud)
    np.testing.assert_array_almost_equal(result.data, expected_data)
    assert result.name() == hybrid_cloud.name()
    assert np.shape(result) == np.shape(hybrid_cloud.data)


def test_difference_too_large(hybrid_cloud, total_cloud):
    """Tests that a warning is raised if the plugin changes
    the forecast probabilities by more than 0.3"""

    hybrid_cloud.data = np.full_like(hybrid_cloud.data, 0.9)

    with pytest.warns():
        EnforceConsistentProbabilities()(hybrid_cloud, total_cloud)

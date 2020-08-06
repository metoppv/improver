# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
""" Tests of FieldTexture plugin."""

import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.field_texture import FieldTexture

NB_RADIUS = 10000
RATIO_THRESH = 0.4


@pytest.fixture(name='multi_cloud_cube')
def multi_cloud_fixture():
    """Multi-realization cloud data cube"""

    cloud_data = np.zeros((3, 10, 10), dtype=np.float32)
    cloud_data[:, 1:4, 1:4] = 1.0
    cube = set_up_variable_cube(
        cloud_data, name="cloud_data", spatial_grid="equalarea"
    )
    return cube


def test__calculate_ratio(multi_cloud_cube):
    """Test the _calculate_ratio function with one realization of the input cube"""

    expected_data = np.where(multi_cloud_cube.data[0] == 0.0, 1.0, 0.3333)

    plugin = FieldTexture(nbhood_radius=NB_RADIUS, ratio_threshold=RATIO_THRESH)
    ratio_result = plugin._calculate_ratio(multi_cloud_cube[0], NB_RADIUS)
    np.testing.assert_almost_equal(ratio_result.data, expected_data, decimal=4)


def test_process(multi_cloud_cube):
    """Test the _calculate_clumpiness function with multi realization input cube"""

    expected_data = np.where(multi_cloud_cube.data[0] == 0.0, 1.0, 0.0)

    plugin = FieldTexture(nbhood_radius=NB_RADIUS, ratio_threshold=RATIO_THRESH)
    clumpiness_result = plugin.process(multi_cloud_cube)
    np.testing.assert_almost_equal(clumpiness_result.data, expected_data, decimal=4)


def test__calculate_transitions(multi_cloud_cube):
    """Test the _calculate_transitions function with single realization input cube"""

    expected_data = np.zeros((10, 10), dtype=np.float32)
    expected_data[1:4, 1:4] = np.array(
        [[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]
    )
    expected_cube = set_up_variable_cube(
        expected_data, name="expected_ratio_data", spatial_grid="equalarea"
    )

    plugin = FieldTexture(nbhood_radius=NB_RADIUS, ratio_threshold=RATIO_THRESH)
    transition_result = plugin._calculate_transitions(multi_cloud_cube[0].data)
    np.testing.assert_almost_equal(
        transition_result.data, expected_cube.data, decimal=4
    )

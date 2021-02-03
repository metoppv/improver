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
"""
Unit tests for synthetic data utilities.
"""

import numpy as np
import pytest

from improver.synthetic_data import utilities


@pytest.mark.parametrize(
    "coord_data,expected_cube_type",
    [
        ({"realizations": [0], "thresholds": [0]}, None),
        ({"realizations": [0, 1, 2]}, "variable"),
        ({"percentiles": [10, 90]}, "percentile"),
        ({"thresholds": [2, 3, 4]}, "probability"),
    ],
)
def test_get_leading_dimension(coord_data, expected_cube_type):
    """ Tests leading dimension data extracted from dictionary and the correct cube
    type is assigned, or if more than one leading dimension present raises an error """
    if expected_cube_type is None:
        msg = "Only one"
        with pytest.raises(ValueError, match=msg):
            utilities.get_leading_dimension(coord_data=coord_data)
    else:
        leading_dimension, cube_type = utilities.get_leading_dimension(
            coord_data=coord_data
        )

        dimension_key = list(coord_data)[0]
        np.testing.assert_array_equal(coord_data[dimension_key], leading_dimension)
        assert expected_cube_type == cube_type


@pytest.mark.parametrize(
    "coord_data,expected_pressure",
    [
        ({"realizations": [0]}, False),
        ({"heights": [0, 1, 2]}, False),
        ({"pressures": [10, 20, 30]}, True),
    ],
)
def test_get_height_levels(coord_data, expected_pressure):
    """ Tests height level data extracted successfully and pressure flag set correctly """
    dimension_key = list(coord_data)[0]

    if dimension_key == "realizations":
        expected_height_levels = None
    else:
        expected_height_levels = coord_data[dimension_key]

    height_levels, pressure = utilities.get_height_levels(coord_data=coord_data)

    np.testing.assert_array_equal(expected_height_levels, height_levels)
    assert expected_pressure == pressure

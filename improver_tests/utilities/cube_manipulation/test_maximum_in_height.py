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
"""
Unit tests for the function "cube_manipulation.maximum_in_height".
"""

import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import maximum_in_height


@pytest.mark.parametrize(
    "lower_bound,upper_bound,expected",
    (
        (None, None, [300, 400, 300]),
        (None, 200, [300, 400, 100]),
        (250, None, [200, 300, 300]),
        (50, 1000, [300, 400, 300]),
    ),
)
def test_maximum_in_height(lower_bound, upper_bound, expected):
    """Test that the maximum over the height coordinate is correctly calculated for
    different combinations of upper and lower bounds."""

    data = np.array(
        [
            [[100, 200, 100], [100, 200, 100]],
            [[300, 400, 100], [300, 400, 100]],
            [[200, 300, 300], [200, 300, 300]],
        ]
    )
    cube = set_up_variable_cube(
        data=data, name="wet_bulb_temperature", height_levels=[100, 200, 300]
    )
    result = maximum_in_height(
        cube, lower_height_bound=lower_bound, upper_height_bound=upper_bound
    )

    assert np.allclose(result.data, [expected] * 2)
    assert "wet_bulb_temperature" == result.name()

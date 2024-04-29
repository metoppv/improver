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
Unit tests for the function manipulate_realization_dimension.
"""

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import manipulate_realization_dimension


@pytest.fixture
def temperature_cube():
    data = 281 * np.random.random_sample((3, 3, 3)).astype(np.float32)
    return set_up_variable_cube(data, realizations=[0, 1, 2])


@pytest.mark.parametrize("n_realizations", (2, 3, 4))
def test_basic(temperature_cube, n_realizations):
    """Test that a cube is returned with expected data and realization coordinate."""
    input_len = len(temperature_cube.coord("realization").points)
    expected_realizations = np.array([r for r in range(n_realizations)])
    result = manipulate_realization_dimension(temperature_cube, n_realizations)

    assert len(result.coord("realization").points) == n_realizations
    assert np.all(result.coord("realization").points == expected_realizations)
    for realization in result.coord("realization").points:
        input_constr = iris.Constraint(realization=realization % input_len)
        result_constr = iris.Constraint(realization=realization)

        input_slice = temperature_cube.extract(input_constr)
        result_slice = result.extract(result_constr)

        np.testing.assert_allclose(result_slice.data, input_slice.data)


def test_non_realization_cube(temperature_cube):
    """Test that the correct exception is raised when input cube does not contain
    a realization dimension.
    """
    temperature_cube.coord("realization").rename("percentile")
    msg = "Input cube does not contain realizations."

    with pytest.raises(ValueError, match=msg):
        manipulate_realization_dimension(temperature_cube, n_realizations=3)

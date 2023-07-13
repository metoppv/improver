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
"""Unit tests for the "forecast_reference_enforcement.split_cubes_by_name" function."""
import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_percentile_cube
from improver.utilities.forecast_reference_enforcement import split_cubes_by_name


def create_test_cubes(n_cubes, name):
    shape = (2, 3, 3)
    percentiles = [40.0, 60.0]
    data = np.ones(shape, dtype=np.float32)
    output = iris.cube.CubeList()

    if n_cubes != 0:
        cube = set_up_percentile_cube(data, percentiles, name=name, units="m s-1")
        for index in range(n_cubes):
            output.append(cube.copy())

    return output


@pytest.mark.parametrize("n_match, n_other", ((0, 1), (1, 0), (1, 1), (2, 1), (1, 2)))
def test_split_cubes_by_name(n_match, n_other):
    match_name = "rainfall_rate"
    other_name = "sleetfall_rate"

    match_cubes = create_test_cubes(n_match, match_name)
    other_cubes = create_test_cubes(n_other, other_name)
    all_cubes = match_cubes.copy()
    all_cubes.extend(other_cubes)

    match_result, other_result = split_cubes_by_name(all_cubes, match_name)

    assert match_result == match_cubes
    assert other_result == other_cubes

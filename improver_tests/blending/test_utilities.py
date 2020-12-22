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
"""Test utilities to support weighted blending"""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.blending.utilities import find_blend_dim_coord
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


@pytest.fixture(name="input_cube")
def input_cube_fixture():
    """Set up a cube for cycle blending"""
    thresholds = [10, 20]
    data = np.ones((2, 2, 2), dtype=np.float32)
    frt_list = [
        datetime(2017, 11, 10, 0),
        datetime(2017, 11, 10, 1),
        datetime(2017, 11, 10, 2),
    ]
    cycle_cubes = iris.cube.CubeList([])
    for frt in frt_list:
        cycle_cubes.append(
            set_up_probability_cube(
                data,
                thresholds,
                spatial_grid="equalarea",
                time=datetime(2017, 11, 10, 4, 0),
                frt=frt,
            )
        )
    return cycle_cubes.merge_cube()


def test_find_blend_dim_coord_noop(input_cube):
    """Test no impact if called on dimension"""
    result = find_blend_dim_coord(input_cube, "forecast_reference_time")
    assert result == "forecast_reference_time"


def test_find_blend_dim_coord_from_aux(input_cube):
    """Test returns correctly if given aux coord"""
    result = find_blend_dim_coord(input_cube, "forecast_period")
    assert result == "forecast_reference_time"


def test_find_blend_dim_coord_error_no_dim(input_cube):
    """Test error if blend coordinate has no dimension"""
    cube = next(input_cube.slices_over("forecast_reference_time"))
    with pytest.raises(ValueError, match="no associated dimension"):
        find_blend_dim_coord(cube, "forecast_reference_time")

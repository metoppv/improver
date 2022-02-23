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
"""Tests for the cube_combiner.Combine plugin."""
from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.cube_combiner import Combine, CubeCombiner, CubeMultiplier
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture(name="realization_cubes")
def realization_cubes_fixture() -> CubeList:
    """Set up a single realization cube in parameter space"""
    realizations = [0, 1, 2, 3]
    data = np.ones((len(realizations), 2, 2), dtype=np.float32)
    times = [datetime(2017, 11, 10, hour) for hour in [4, 5, 6]]
    cubes = CubeList()
    for time in times:
        cubes.append(
            set_up_variable_cube(
                data,
                realizations=realizations,
                spatial_grid="equalarea",
                time=time,
                frt=datetime(2017, 11, 10, 1),
            )
        )
    cube = cubes.merge_cube()
    return CubeList(cube.slices_over("realization"))


@pytest.mark.parametrize("broadcast_to_threshold", (False, True))
@pytest.mark.parametrize("minimum_realizations", (None, 1))
@pytest.mark.parametrize(
    "operation, expected_instance", (("+", CubeCombiner), ("*", CubeMultiplier)),
)
def test_init(
    operation, expected_instance, minimum_realizations, broadcast_to_threshold
):
    """Ensure the class initialises as expected"""
    result = Combine(
        operation,
        broadcast_to_threshold=broadcast_to_threshold,
        minimum_realizations=minimum_realizations,
        new_name="name",
    )
    assert isinstance(result.plugin, expected_instance)
    assert result.new_name == "name"
    assert result.minimum_realizations == minimum_realizations
    assert result.broadcast_to_threshold == broadcast_to_threshold
    if broadcast_to_threshold and isinstance(result.plugin, CubeMultiplier):
        assert result.plugin.broadcast_to_threshold == broadcast_to_threshold


@pytest.mark.parametrize("short_realizations", [0, 1, 2, 3])
def test_filtering_realizations(realization_cubes, short_realizations):
    """Run Combine with minimum_realizations and a realization time series where 0 or more are
    short of the final time step"""
    if short_realizations == 0:
        cubes = realization_cubes
        expected_realization_points = [0, 1, 2, 3]
    else:
        cubes = CubeList(realization_cubes[:-short_realizations])
        cubes.append(realization_cubes[-short_realizations][:-1])
        expected_realization_points = [0, 1, 2, 3][:-short_realizations]
    result = Combine(
        "+", broadcast_to_threshold=False, minimum_realizations=1, new_name="name"
    )(cubes)
    assert isinstance(result, Cube)
    assert np.allclose(result.coord("realization").points, expected_realization_points)
    assert np.allclose(result.data, 3)


@pytest.mark.parametrize(
    "minimum_realizations, error_class, msg",
    (
        (0, ValueError, "Minimum realizations must be at least 1, not 0"),
        ("0", ValueError, "Minimum realizations must be at least 1, not 0"),
        (-1, ValueError, "Minimum realizations must be at least 1, not -1"),
        (
            5,
            ValueError,
            "After filtering, number of realizations 4 is less than the minimum number of "
            r"realizations allowed \(5\)",
        ),
        ("kittens", ValueError, r"invalid literal for int\(\) with base 10: 'kittens'"),
        (
            ValueError,
            TypeError,
            r"int\(\) argument must be a string, a bytes-like object or a number, not 'type'",
        ),
    ),
)
def test_minimum_realizations_exceptions(
    minimum_realizations, error_class, msg, realization_cubes
):
    """Ensure specifying too few realizations will raise an error"""
    with pytest.raises(error_class, match=msg):
        Combine("+", minimum_realizations=minimum_realizations)(realization_cubes)


def test_empty_cubelist():
    """Ensure supplying an empty CubeList raises an error"""
    with pytest.raises(TypeError, match="A cube is needed to be combined."):
        Combine("+")(CubeList())

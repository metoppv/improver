# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the "utilities.filter_realizations" function.
"""
from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import filter_realizations


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
    sliced_cubes = CubeList(cube.slices_over("realization"))
    [
        s.attributes.update({"history": f"20171110T{i:02d}00Z"})
        for i, s in enumerate(sliced_cubes)
    ]
    return sliced_cubes


@pytest.mark.parametrize("short_realizations", [0, 1, 2, 3])
def test_filter_realizations(realization_cubes, short_realizations):
    """Run filter_realizations with realization time series where 0 or more are short of the
    final time step"""
    if short_realizations == 0:
        cubes = realization_cubes
        expected_realization_points = [0, 1, 2, 3]
    else:
        cubes = CubeList(realization_cubes[:-short_realizations])
        cubes.append(realization_cubes[-short_realizations][:-1])
        expected_realization_points = [0, 1, 2, 3][:-short_realizations]
    result = filter_realizations(cubes)
    assert isinstance(result, Cube)
    assert np.allclose(cubes[0].coord("time").points, result.coord("time").points)
    assert np.allclose(result.coord("realization").points, expected_realization_points)
    if short_realizations == 3:
        # History attribute is retained if there are no differing values
        assert result.attributes["history"] == cubes[0].attributes["history"]
    else:
        # History attribute is removed if differing values are supplied
        assert "history" not in result.attributes.keys()


def test_different_time_lengths(realization_cubes):
    """Run filter_realizations with realization time series where two are short of differing
    time steps"""
    cubes = CubeList(realization_cubes[:2])
    cubes.append(realization_cubes[2][1:])
    cubes.append(realization_cubes[3][:-1])
    result = filter_realizations(cubes)
    assert isinstance(result, Cube)
    assert np.allclose(cubes[0].coord("time").points, result.coord("time").points)
    assert np.allclose(result.coord("realization").points, [0, 1])

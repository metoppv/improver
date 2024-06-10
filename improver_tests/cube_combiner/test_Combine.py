# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the cube_combiner.Combine plugin."""
from datetime import datetime
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.cube_combiner import Combine, CubeCombiner
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class HaltExecution(Exception):
    pass


@patch("improver.cube_combiner.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        Combine("+")(sentinel.cube1, sentinel.cube2)
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(sentinel.cube1, sentinel.cube2)


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


@pytest.mark.parametrize("broadcast", (None, "threshold"))
@pytest.mark.parametrize("minimum_realizations", (None, 1))
def test_init(minimum_realizations, broadcast):
    """Ensure the class initialises as expected"""
    operation = "+"
    result = Combine(
        operation,
        broadcast=broadcast,
        minimum_realizations=minimum_realizations,
        new_name="name",
    )
    assert isinstance(result.plugin, CubeCombiner)
    assert result.new_name == "name"
    assert result.minimum_realizations == minimum_realizations
    assert result.broadcast == broadcast
    if broadcast is not None and isinstance(result.plugin, CubeCombiner):
        assert result.plugin.broadcast == broadcast


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
    result = Combine("+", broadcast=None, minimum_realizations=1, new_name="name")(
        cubes
    )
    assert isinstance(result, Cube)
    assert np.allclose(result.coord("realization").points, expected_realization_points)
    assert np.allclose(result.data, 3)


@pytest.mark.parametrize("coordinate_name", ("realization", "percentile"))
def test_cubes_different_size(
    realization_cubes, coordinate_name,
):
    """Checks Combine works with different broadcast coordinates."""
    cubes = realization_cubes.merge_cube()
    cubes.data = np.full_like(cubes.data, 1)
    cubes.coord("realization").rename(coordinate_name)
    small_cube = next(realization_cubes[0].slices_over("realization"))
    small_cube.remove_coord("realization")
    enforce_coordinate_ordering(
        small_cube, ["projection_x_coordinate", "time", "projection_y_coordinate"]
    )
    result = Combine("-", broadcast=coordinate_name)([cubes, small_cube])
    assert np.allclose(result.data, 0)


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

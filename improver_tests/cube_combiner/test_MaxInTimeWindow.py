# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the cube_combiner.MaxInTimeWindow plugin."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from iris.coords import CellMethod
from iris.cube import CubeList

from improver.cube_combiner import MaxInTimeWindow
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)


def setup_realization_cubes(period=3) -> CubeList:
    """Set up a period diagnostic cubes."""
    realizations = [0, 1, 2, 3]
    data = np.full((len(realizations), 2, 2), 0.1, dtype=np.float32)
    times = [datetime(2017, 11, 10, hour) for hour in [3, 6, 9, 12]]
    cubes = CubeList()
    for increment, time in enumerate(times):
        cubes.append(
            set_up_variable_cube(
                data + increment * 0.1,
                name="lwe_thickness_of_precipitation_amount",
                realizations=realizations,
                spatial_grid="equalarea",
                time=time,
                time_bounds=(time - timedelta(hours=period), time),
                frt=datetime(2017, 11, 9, 0),
            )
        )
    return cubes


def setup_probability_cubes(period=3) -> CubeList:
    """Set up a period diagnostic cubes."""
    thresholds = [0, 1, 2, 3]
    data = np.full((len(thresholds), 2, 2), 0.1, dtype=np.float32)
    times = [datetime(2017, 11, 10, hour) for hour in [3, 6, 9, 12]]
    cubes = CubeList()
    for increment, time in enumerate(times):
        cubes.append(
            set_up_probability_cube(
                data + increment * 0.1,
                variable_name="lwe_thickness_of_precipitation_amount",
                thresholds=thresholds,
                spatial_grid="equalarea",
                time=time,
                time_bounds=(time - timedelta(hours=period), time),
                frt=datetime(2017, 11, 9, 0),
            )
        )
    return cubes


@pytest.mark.parametrize(
    "period", [(3), (1)],
)
@pytest.mark.parametrize(
    "forecast_cubes", [(setup_realization_cubes), (setup_probability_cubes)],
)
def test_basic(forecast_cubes, period):
    """Test for max in a time window."""
    expected = np.full((4, 2, 2), 0.4)
    hour_text = "1 hour" if int(period) == 1 else "3 hours"
    sum_comment = (
        f"of lwe_thickness_of_precipitation_amount over {hour_text} within time window"
    )
    max_comment = "of lwe_thickness_of_precipitation_amount"
    cell_methods = (
        CellMethod("sum", coords=["time"], comments=sum_comment),
        CellMethod("maximum", coords=["time"], comments=max_comment),
    )
    result = MaxInTimeWindow()(forecast_cubes(period=period))
    if result.coords("lwe_thickness_of_precipitation_amount"):
        assert (
            result.name()
            == "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
        )
    else:
        assert result.name() == "lwe_thickness_of_precipitation_amount"
    assert result.cell_methods == cell_methods
    np.testing.assert_allclose(result.data, expected)


def test_existing_cell_methods():
    """Test the handling of existing cell methods."""
    cubes = setup_realization_cubes()
    comment = "of lwe_thickness_of_precipitation_amount"
    for cube in cubes:
        cube.add_cell_method(CellMethod("sum", coords=["time"], comments=comment))
        cube.add_cell_method(CellMethod("maximum", coords=["height"], comments=comment))
    sum_comment = (
        "of lwe_thickness_of_precipitation_amount over 3 hours within time window"
    )
    cell_methods = (
        CellMethod("maximum", coords=["height"], comments=comment),
        CellMethod("sum", coords=["time"], comments=sum_comment),
        CellMethod("maximum", coords=["time"], comments=comment),
    )
    result = MaxInTimeWindow()(cubes)
    assert result.cell_methods == cell_methods


def test_absent_bounds():
    """Test an exception is raised when bounds are absent from all input cubes.
    Input cubes are expected to be period diagnostics with bounds."""
    cubes = setup_realization_cubes()
    for cube in cubes:
        cube.coord("time").bounds = None
    msg = "The cubes provided do not have bounds"
    with pytest.raises(ValueError, match=msg):
        MaxInTimeWindow()(cubes)


def test_incomplete_bounds():
    """Test an exception is raised when bounds are only present on some of the input
    cubes. Input cubes are all expected to be period diagnostics with bounds."""
    cubes = setup_realization_cubes()
    cubes[0].coord("time").bounds = None
    msg = "The cubes provided do not all have bounds"
    with pytest.raises(ValueError, match=msg):
        MaxInTimeWindow()(cubes)


def test_bound_mismatch():
    """Test an exception is raised when bounds are present but the period implied by
    the bounds does not match between the input cubes. A consistent period is
    expected."""
    cubes = setup_realization_cubes()
    cubes[0].coord("time").bounds = [
        cubes[0].coord("time").bounds[0][0] - 3600 * 3,
        cubes[0].coord("time").bounds[0][1],
    ]
    msg = "The bounds on the cubes imply mismatching periods"
    with pytest.raises(ValueError, match=msg):
        MaxInTimeWindow()(cubes)


@pytest.mark.parametrize(
    "minimum_realizations", [(4), (5)],
)
def test_minimum_realizations(minimum_realizations):
    """Test the behaviour if the number of realizations is less than the minimum
    allowed."""
    plugin = MaxInTimeWindow(minimum_realizations=minimum_realizations)
    if minimum_realizations == 5:
        msg = (
            "After filtering, number of realizations 4 is less than the minimum number "
            rf"of realizations allowed \({minimum_realizations}\)"
        )
        with pytest.raises(ValueError, match=msg):
            plugin(setup_realization_cubes())
    elif minimum_realizations == 4:
        expected = np.full((4, 2, 2), 0.4)
        result = plugin(setup_realization_cubes())
        np.testing.assert_allclose(result.data, expected)

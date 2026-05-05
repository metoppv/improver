# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for SubperiodSelector."""

from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.categorical.subperiod_selector import SubperiodSelector
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)


@pytest.fixture
def main_period_cube():
    """Set up a cube containing the main period diagnostic."""
    data = np.ones((3, 2, 2), dtype=np.float32)
    cube = set_up_probability_cube(
        data,
        variable_name="lwe_precipitation_rate",
        threshold_units="m s-1",
        thresholds=[0.0, 5.56e-07, 1.11e-06],
    )
    cube = add_coordinate(
        cube, [0.0003, 0.003], "lwe_thickness_of_precipitation_amount", coord_units="m"
    )
    cube = add_coordinate(cube, [10, 50, 90], "percentile", coord_units="%")
    return cube


@pytest.fixture
def subperiod_cube():
    """Set up a cube containing the subperiod diagnostic where four time slices have differing probabilities."""
    data = np.zeros((3, 2, 2), dtype=np.float32)
    cubes = CubeList()
    for hour, probability in enumerate([0, 0.5, 0.25, 0.75]):
        cubes.append(
            set_up_probability_cube(
                np.full_like(data, probability),
                time=datetime(2017, 11, 10, hour, 0),
                variable_name="lwe_precipitation_rate",
                threshold_units="m s-1",
                thresholds=[0.0, 5.56e-07, 1.11e-06],
            )
        )
    return cubes.merge_cube()


@pytest.mark.parametrize("new_name", [None, "period_is_wet"])
@pytest.mark.parametrize(
    "wet_fraction, wet_periods",
    [[0.0, []], [0.25, [3]], [0.5, [1, 3]], [0.75, [1, 2, 3]], [1.0, [0, 1, 2, 3]]],
)
def test_basic(main_period_cube, subperiod_cube, wet_fraction, wet_periods, new_name):
    """Test that the plugin selects the correct subperiods for varying wet_fraction values."""
    main_period_cube.data *= wet_fraction

    plugin = SubperiodSelector(
        percentile=50,
        lwe_thickness_of_precipitation_amount=0.003,
        lwe_precipitation_rate=5.56e-07,
        **{"new_name": new_name} if new_name is not None else {},
    )
    result = plugin(main_period_cube, subperiod_cube)

    assert isinstance(result, Cube)
    expected_data = np.full_like(subperiod_cube.data[:, 0, ...], False, dtype=bool)
    for wp in wet_periods:
        expected_data[wp, :] = True
    assert np.array_equal(result.data, expected_data)
    assert result.name() == "selected_subperiods" if new_name is None else new_name
    assert result.units == "1"
    assert result.dtype is np.dtype(bool)


@pytest.mark.parametrize(
    "kwargs, expected_error, expected_message",
    [
        [{"percentile": 50}, ValueError, "No matching threshold coordinate found"],
        [
            {"percentile": 50, "lwe_thickness_of_precipitation_amount": 0.003},
            ValueError,
            "No matching threshold coordinate found",
        ],
        [
            {"percentile": 50, "lwe_thickness_of_precipitation_amount": 0.99},
            ValueError,
            "No data found in main period cube matching",
        ],
        [
            {"percentile": 50, "lwe_precipitation_rate": 5.56e-07},
            ValueError,
            "Expected subperiod cube to have exactly one more dimension",
        ],
        [
            {
                "percentile": 50,
                "lwe_precipitation_rate": 5.56e-07,
                "lwe_thickness_of_precipitation_amount": 0.003,
            },
            ValueError,
            "No data found in subperiod cube matching threshold constraints",
        ],
        [{}, TypeError, "missing 1 required positional argument: 'percentile'"],
    ],
)
def test_exceptions(
    main_period_cube, subperiod_cube, kwargs, expected_error, expected_message
):
    """Test that the plugin raises useful errors."""
    if (
        expected_message
        == "No data found in subperiod cube matching threshold constraints"
    ):
        threshold_coord = subperiod_cube.coord("lwe_precipitation_rate")
        threshold_coord.points = [0.0, 1.0, 2.0]
        subperiod_cube.replace_coord(threshold_coord)
    with pytest.raises(expected_error, match=expected_message):
        plugin = SubperiodSelector(**kwargs)
        plugin(main_period_cube, subperiod_cube)

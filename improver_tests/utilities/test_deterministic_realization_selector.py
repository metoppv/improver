# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.deterministic_realization_selector import (
    DeterministicRealizationSelector,
)


def basic_input_cube() -> Cube:
    """Set up an input cube with realizations"""
    data = np.array(
        [
            [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
            [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    input_cube = set_up_variable_cube(data)
    return input_cube


@pytest.fixture
def forecast_cube() -> Cube:
    """Return a forecast cube with realizations."""
    forecast_cube = basic_input_cube()
    return forecast_cube


@pytest.fixture
def cluster_cube() -> Cube:
    """Return a cluster cube with realizations and the attribute:
    primary_input_realizations_to_clusters"""
    cluster_cube = basic_input_cube()
    cluster_cube.attributes = {
        "primary_input_realizations_to_clusters": '{"0": [19], "1": [0, 18]}'
    }
    return cluster_cube


def create_input_cubelist(cube_1, cube_2) -> CubeList:
    """Return a cluster cube with realizations and the attribute:
    primary_input_realizations_to_clusters"""
    input_cubelist = CubeList([cube_1, cube_2])
    return input_cubelist


def create_output_cube(cube: Cube) -> Cube:
    """Return an output cube that has been extracted from the forecast cube,
    with only a single cluster 1 which contains the target realization 0"""
    output_cube = cube.copy()
    realization_constraint = iris.Constraint(realization=1)
    output_cube = output_cube.extract(realization_constraint)
    return output_cube


# TESTS


def test_deterministic_realization(forecast_cube, cluster_cube):
    """Test the deterministic realization selector function produces
    an output cube with the correct cluster, as expected."""
    input_cubelist = create_input_cubelist(forecast_cube, cluster_cube)
    output_cube = create_output_cube(forecast_cube)
    result = DeterministicRealizationSelector().process(input_cubelist)
    assert isinstance(result, iris.cube.Cube)
    assert result == output_cube


def test_missing_control_member(forecast_cube, cluster_cube):
    """Test the deterministic realization selector function raises an attribute error,
    when provided a realization that doesn't exist."""
    input_cubelist = create_input_cubelist(forecast_cube, cluster_cube)
    with pytest.raises(AttributeError):
        DeterministicRealizationSelector(target_realization_id=1).process(
            input_cubelist
        )


def test_missing_attribute(forecast_cube, cluster_cube):
    """Test the deterministic realization selector function raises an attribute error,
    when provided a cluster cube without the attribute:
    primary_input_realizations_to_clusters"""
    cluster_cube_no_attribute = cluster_cube.copy()
    cluster_cube_no_attribute.attributes.pop("primary_input_realizations_to_clusters")
    input_cubelist = create_input_cubelist(forecast_cube, cluster_cube_no_attribute)
    with pytest.raises(AttributeError):
        DeterministicRealizationSelector().process(input_cubelist)


def test_incorrect_cubelist(forecast_cube, cluster_cube):
    """Test the deterministic realization selector function raises an attribute error,
    when provided an input_cubelist with more than 2 cubes."""
    input_cubelist = CubeList([forecast_cube, forecast_cube, cluster_cube])
    with pytest.raises(AttributeError):
        DeterministicRealizationSelector().process(input_cubelist)

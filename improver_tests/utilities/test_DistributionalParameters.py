# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the DistributionalParameters class within statistical.py"""

import re

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)
from improver.utilities.statistical import DistributionalParameters

MANDATORY_ATTRIBUTES = {
    "title": "MOGREPS-UK Forecast",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}


@pytest.mark.parametrize("distribution", ["norm", "truncnorm", "gamma", "unsupported"])
def test__init__(distribution):
    """Test that the class initialises variables correctly."""

    if distribution == "unsupported":
        with pytest.raises(
            ValueError, match="Distribution 'unsupported' is not supported."
        ):
            DistributionalParameters(distribution=distribution)
    else:
        plugin = DistributionalParameters(distribution=distribution)
        assert plugin.distribution == distribution


@pytest.mark.parametrize("distribution", ["norm", "truncnorm", "gamma"])
@pytest.mark.parametrize("forecast_type", ["gridded", "spot"])
def test_process(distribution, forecast_type):
    """Test that the process method returns the expected output for supported
    distributions.
    """
    if forecast_type == "gridded":
        data_shape = [2, 2]  # Latitude, Longitude.
        func = set_up_variable_cube
    else:
        data_shape = [2]  # Number of sites.
        func = set_up_spot_variable_cube

    mean_cube = func(
        data=np.full(data_shape, 3.0, np.float32), attributes=MANDATORY_ATTRIBUTES
    )
    sd_cube = func(
        data=np.full(data_shape, 2.0, np.float32), attributes=MANDATORY_ATTRIBUTES
    )

    kwargs = {}

    if distribution == "norm":
        expected_shape = None
        expected_location = np.full(data_shape, 3.0, dtype=np.float32)
        expected_scale = np.full(data_shape, 2.0, dtype=np.float32)
    elif distribution == "truncnorm":
        kwargs.update({"truncation_points": [0.0, np.inf]})
        expected_shape = [
            np.full(data_shape, 0.0, dtype=np.float32),
            np.full(data_shape, np.inf, dtype=np.float32),
        ]
        expected_location = np.full(data_shape, 3.0, dtype=np.float32)
        expected_scale = np.full(data_shape, 2.0, dtype=np.float32)
    elif distribution == "gamma":
        expected_shape = np.full(data_shape, (3.0 / 2.0) ** 2, dtype=np.float32)
        expected_location = np.full(data_shape, 0.0, dtype=np.float32)
        expected_scale = np.full(data_shape, (2.0**2) / 3.0, dtype=np.float32)

    expected_data_arrays = [expected_shape, expected_location, expected_scale]
    expected_names = ["shape_parameter", "location_parameter", "scale_parameter"]

    result_shape, result_loc, result_scale = DistributionalParameters(
        distribution=distribution
    ).process(mean_cube, sd_cube, **kwargs)

    for expected_data, expected_name, result in zip(
        expected_data_arrays, expected_names, [result_shape, result_loc, result_scale]
    ):
        if expected_data is None:
            assert result is None
        elif isinstance(expected_data, list):
            assert isinstance(result, iris.cube.CubeList)
            for res, exp in zip(result, expected_data):
                np.testing.assert_array_almost_equal(res.data, exp)
                assert res.name() == expected_name
        else:
            np.testing.assert_array_almost_equal(result.data, expected_data)
            assert result.name() == expected_name


@pytest.mark.parametrize("truncation_points", [None, [0.0], [0.0, 1.0, 2.0]])
def test_truncnorm_exception(truncation_points):
    """Test that an exception is raised if truncation points are not provided."""
    mean_cube = set_up_variable_cube(
        data=np.full([2, 2], 3.0, np.float32), attributes=MANDATORY_ATTRIBUTES
    )
    sd_cube = set_up_variable_cube(
        data=np.full([2, 2], 2.0, np.float32), attributes=MANDATORY_ATTRIBUTES
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Upper and lower truncation points must be provided for truncated "
            "normal distribution. The following truncation points were provided: "
            f"{truncation_points}."
        ),
    ):
        DistributionalParameters(distribution="truncnorm").process(
            mean_cube, sd_cube, truncation_points=truncation_points
        )

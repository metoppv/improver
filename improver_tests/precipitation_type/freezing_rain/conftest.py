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
"""Fixtures for freezing rain tests"""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)

COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "Post-Processed IMPROVER unit test",
}
PERIOD_TIMEBOUNDS = (datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 4, 0))


def setup_cube(
    data, thresholds, threshold_units, cube_name, time_bounds, inequality="greater_than"
):
    """Construct a probability cube"""
    return set_up_probability_cube(
        data,
        thresholds,
        variable_name=cube_name,
        threshold_units=threshold_units,
        time_bounds=time_bounds,
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="uk_ens",
        spp__relative_to_threshold=inequality,
    )


def rain_cube(cube_name, thresholds, threshold_units, time_bounds):
    """Construct a rainfall rate or accumulation cube."""
    threshold_0 = np.linspace(0.1, 0.4, 4).reshape(2, 2).astype(np.float32)
    data = np.stack([threshold_0, threshold_0 - 0.1])
    return setup_cube(
        data, thresholds, threshold_units, cube_name.format("rain"), time_bounds
    )


def sleet_cube(cube_name, thresholds, threshold_units, time_bounds):
    """Construct a sleetfall rate or accumulation cube."""
    threshold_0 = np.linspace(0.2, 0.5, 4).reshape(2, 2).astype(np.float32)
    data = np.stack([threshold_0, threshold_0 - 0.1])
    return setup_cube(
        data, thresholds, threshold_units, cube_name.format("lwe_sleet"), time_bounds
    )


def precipitation_cubes(period):
    """Construct rain and sleet rate or accumulation cubes."""
    thresholds = [0.1, 0.2]
    if period == "instantaneous":
        cube_name = "{}rate"
        threshold_units = "mm hr-1"
        time_bounds = None
    if period == "period":
        cube_name = "thickness_of_{}fall_amount"
        threshold_units = "mm"
        time_bounds = PERIOD_TIMEBOUNDS

    return (
        rain_cube(cube_name, thresholds, threshold_units, time_bounds),
        sleet_cube(cube_name, thresholds, threshold_units, time_bounds),
    )


def temperature_cube(period, inequality="greater_than"):
    """Construct an instantaneous or minimum in period air temperature cube."""
    cube_name = "air_temperature"
    thresholds = [273.15, 274.15]
    threshold_units = "K"

    if period == "instantaneous":
        time_bounds = None
    if period == "period":
        time_bounds = PERIOD_TIMEBOUNDS

    data = np.linspace(1, 0.3, 8).reshape(2, 2, 2).astype(np.float32)
    return setup_cube(data, thresholds, threshold_units, cube_name, time_bounds)


@pytest.fixture(params=["instantaneous", "period"])
def input_cubes(request):
    """Return rain, sleet, and air temperature cubes as an iris CubeList. This
    fixture is parameterised such that any test using it will be run with both
    instantaneous and period diagnostics."""
    return iris.cube.CubeList(
        [*precipitation_cubes(request.param), temperature_cube(request.param)]
    )


@pytest.fixture
def precipitation_only(period):
    """Return rain and sleet cubes as a tuple. This fixture is parameterised
    such that any test using it will be run with both instantaneous and period
    diagnostics."""
    return precipitation_cubes(period)


@pytest.fixture
def precipitation_multi_realization(period):
    """Return multi-realization rain and sleet cubes as a tuple. This fixture
    is parameterised such that any test using it will be run with both
    instantaneous and period diagnostics."""
    rain, sleet = precipitation_cubes(period)
    rain = add_coordinate(rain, [0, 1], "realization", coord_units=1, dtype=np.int32)
    sleet = add_coordinate(sleet, [0, 1], "realization", coord_units=1, dtype=np.int32)
    return rain, sleet


@pytest.fixture
def temperature_only(period):
    """Return an air temperature cube. This fixture is parameterised such that
    any test using it will be run with both instantaneous and period air
    temperature as an input."""
    return temperature_cube(period)


@pytest.fixture
def temperature_multi_realization(period):
    """Return a multi-realization air temperature cube. This fixture is
    parameterised such that any test using it will be run with both instantaneous
    and period air temperature as an input."""
    temperature = temperature_cube(period)
    temperature = add_coordinate(
        temperature, [1, 2, 3], "realization", coord_units=1, dtype=np.int32
    )
    return temperature


@pytest.fixture
def temperature_below(period):
    """Return an air temperature cube that has thresholds created using a
    "less than" inequality. This fixture is parameterised such that any test
    using it will be run with both instantaneous and period air temperature as
    an input."""
    return temperature_cube(period, inequality="less_than")


@pytest.fixture
def expected_probabilities():
    """Return the expected freezing rain probabilities."""
    return np.array(
        [[[0.0, 0.05], [0.14, 0.27]], [[0.0, 0.03], [0.1, 0.21]]], dtype=np.float32
    )


@pytest.fixture
def expected_attributes():
    """Return the expected freezing rain cube attributes."""
    return {
        "source": "Unit test",
        "institution": "Met Office",
        "title": "Post-Processed IMPROVER unit test",
    }

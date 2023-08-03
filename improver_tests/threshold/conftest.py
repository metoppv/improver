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
"""Fixtures for freezing rain tests"""

from typing import List, Optional, Tuple, Union
from datetime import datetime

from iris.cube import Cube
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube
)
from improver.threshold import Threshold as Threshold
from improver.utilities.probability_manipulation import comparison_operator_dict

COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "IMPROVER unit test",
}


def diagnostic_cube(n_realizations=1, data=None):
    """Return a cube with a single configurable data point."""
    if data is None:
        data = np.zeros((n_realizations, 5, 5), dtype=np.float32)
        data[..., 2, 2]  = 0.5
        data = np.squeeze(data)

    return set_up_variable_cube(
        data,
        name="precipitation_rate",
        units="mm/hr",
        spatial_grid="equalarea",
        realizations=range(n_realizations),
        attributes=COMMON_ATTRS,
        standard_grid_metadata="uk_ens",
    )


def deterministic_cube():
    return diagnostic_cube()


def single_realization_cube():
    cube = diagnostic_cube()
    cube = add_coordinate(cube, [0], "realization", dtype=np.int)
    return cube


def multi_realization_cube():
    data = np.zeros((2, 5, 5), dtype=np.float32)
    data[..., 2, 2]  = [0.45, 0.55]
    return diagnostic_cube(n_realizations=2, data=data)


def expected_name_func(comparison_operator):
    """Generate the text description of the comparison operator
    for incorporation in the cube name or coordinate attributes."""

    return comparison_operator_dict()[comparison_operator].spp_string


@pytest.fixture
def custom_cube(n_realizations, data):
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    return diagnostic_cube(n_realizations, data)


@pytest.fixture
def expected_cube_name(comparison_operator, vicinity):
    """Return a template for the name of a thresholded cube taking into
    account the comparison operator."""

    vicinity_name = "_in_vicinity" if vicinity is not None else ""

    if "g" in comparison_operator.lower() or ">" in comparison_operator:
        return f"probability_of_{{cube_name}}{vicinity_name}_above_threshold"
    else:
        return f"probability_of_{{cube_name}}{vicinity_name}_below_threshold"


@pytest.fixture
def expected_result(
    expected_single_value: float,
    expected_multi_value: List[float],
    collapse: bool,
    comparator: str,
    default_cube: Cube):
    """Return the expected values following thresholding.
    The default cube has a 0.5 value at location (2, 2), with all other
    values set to 0. The expected result, for a ">" or ">=" threshold
    comparator, is achieved by placing the expected value, or values for
    multi-realization data without coordinate collapsing, at the (..., 2, 2)
    location in an array full of zeros. This array matches the size of
    the input cube.

    If the comparator is changed to be be "<" or "<=" then the array is
    filled with ones, and the expected value(s) is subtracted from 1 prior
    to being places at the (..., 2, 2) location.

    This function does no more than this to avoid making the tests harder
    to follow. Cases for which the threshold value is equal to any of
    the data values (without fuzziness) meaning that the ">" and ">=", or
    "<" and "<=" comparators would give different results are handled
    separately.

    Args:
        expected_single_value:
            The value expected if a single realization or deterministic
            input cube is being tested.
        expected_multi_value:
            The values expected for each realization is a multi-realization
            input cube is being tested. If the collapse argument is true the
            expected result is the mean of these values.
        collapse:
            Whether the test includes the collapsing of the realization
            coordinate to calculate the final result.
        comparator:
            The comapartor being applied in the thresholding, either ">",
            ">=", "<", or "<=".
        default_cube:
            The input cube to the test.
    """

    try:
        n_realizations = default_cube.coord("realization").shape[0]
    except CoordinateNotFoundError:
        n_realizations = 1

    if "l" in comparator:
        expected_single_value = 1. - expected_single_value
        expected_multi_value = [1. - value for value in expected_multi_value]
        expected_result_array = np.ones_like(default_cube.data)
    else:
        expected_result_array = np.zeros_like(default_cube.data)

    if n_realizations == 1:
        expected_result_array[..., 2, 2] = expected_single_value
    elif collapse:
        expected_result_array[..., 2, 2] = np.mean(expected_multi_value)
    else:
        expected_result_array[..., 2, 2] = expected_multi_value

    if collapse and n_realizations > 1:
        expected_result_array = expected_result_array[0]

    return expected_result_array


@pytest.fixture(params=[deterministic_cube, single_realization_cube, multi_realization_cube])
def default_cube(request):
    return request.param()


@pytest.fixture
def threshold_coord(threshold_values, threshold_units, comparison_operator):
    """
    Generate an expected threshold coordinate based on the threshold
    values and comparison operator.
    """
    attributes = {"spp__relative_to_threshold": expected_name_func(comparison_operator)}

    threshold = DimCoord(
        np.array(threshold_values, dtype=np.float32),
        long_name="precipitation_rate",
        var_name="threshold",
        units=threshold_units,
        attributes=attributes,
    )
    threshold.convert_units("mm/hr")
    return threshold

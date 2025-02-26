# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Fixtures for threshold tests"""

from typing import List, Optional, Union

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.probability_manipulation import comparison_operator_dict

COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "IMPROVER unit test",
}


def diagnostic_cube(
    n_realizations: int = 1,
    n_times: Optional[int] = 1,
    data: Optional[np.ndarray] = None,
) -> Cube:
    """Return a cube of precipitation rate in mm/hr with n_realizations
    containing the data provided by the data argument.

    Args:
        n_realizations:
            The number of realizations to be present in the returned cube.
        n_times:
            Number of times in the time coordinate. If equal to 1 the
            returned cube contains only a scalar time coordinate.
        data:
            The data to be contained in the cube. If specified this must
            contain a leading dimension of length n_realizations.
    Returns:
        A diagnostic cube for use in testing.
    """
    if data is None:
        data = np.zeros((n_realizations, 5, 5), dtype=np.float32)
        data[..., 2, 2] = 0.5
        data = np.squeeze(data)

    # print("n_times", n_times)
    # print("dims", (3 + (n_realizations > 1)))
    # print("data dims", data.ndim)

    if n_times > 1 and data.ndim != (3 + (n_realizations > 1)):
        data = np.ma.stack([data] * n_times)
    elif n_times == 1:
        data = [data]

    cubes = CubeList()
    for index, dslice in enumerate(data):
        cube = set_up_variable_cube(
            dslice,
            name="precipitation_rate",
            units="mm/hr",
            spatial_grid="equalarea",
            realizations=range(n_realizations),
            attributes=COMMON_ATTRS,
            standard_grid_metadata="uk_ens",
        )
        cube.coord("time").points = cube.coord("time").points + ((index - 1) * 3600)
        cubes.append(cube)

    return cubes.merge_cube()


def deterministic_cube() -> Cube:
    """Return the diagnostic cube without a realization coordinate."""
    return diagnostic_cube()


def single_realization_cube() -> Cube:
    """Return the diagnostic cube with a scalar realization coordinate."""
    cube = diagnostic_cube()
    cube = add_coordinate(cube, [0], "realization", dtype=np.int)
    return cube


def multi_realization_cube() -> Cube:
    """Return the diagnostic cube with a multi-valued realization coordinate."""
    data = np.zeros((2, 5, 5), dtype=np.float32)
    data[..., 2, 2] = [0.45, 0.55]
    return diagnostic_cube(n_realizations=2, data=data)


def multi_realization_multi_time_cube() -> Cube:
    """Return the diagnostic cube with multi-valued realization and time
    coordinates. The data is duplicated for each time, meaning tests for
    expected values should return the same data as the standard multi-
    realization tests."""
    data = np.zeros((2, 5, 5), dtype=np.float32)
    data[..., 2, 2] = [0.45, 0.55]
    return diagnostic_cube(n_realizations=2, n_times=2, data=data)


def expected_name_func(comparison_operator: str) -> str:
    """Return the text description of the comparison operator
    for incorporation in the cube name or coordinate attributes.

    Args:
        comparison_operator:
            The comparator used for thresholding, e.g. >.
    Returns:
        e.g. "greater_than"
    """
    return comparison_operator_dict()[comparison_operator].spp_string


@pytest.fixture
def custom_cube(n_realizations: int, n_times: int, data: np.ndarray) -> Cube:
    """Provide a diagnostic cube with a configurable number of realizations
    and custom data.

    Args:
        n_realizations:
            The number of realizations to be present in the returned cube.
        n_times:
            The length of the time dimension.
        data:
            The data to be contained in the cube. If specified this must
            contain a leading dimension of length n_realizations.
    Returns:
        A diagnostic cube for use in testing.
    """

    if data.dtype == np.float64:
        data = data.astype(np.float32)
    return diagnostic_cube(n_realizations=n_realizations, n_times=n_times, data=data)


@pytest.fixture
def expected_cube_name(
    comparison_operator: str, vicinity: Optional[Union[float, List[float]]]
) -> str:
    """Return a template for the name of a thresholded cube taking into
    account the comparison operator and the application of vicinity
    processing.

    Args:
        comparison_operator:
            The comparator used for thresholding, e.g. >.
        vicinity:
            A vicinity radius if set, or None. Used to determine whether
            the diagnostic name should include the vicinity term.
    Returns:
        The expected diagnostic name after thresholding.
    """

    vicinity_name = "_in_vicinity" if vicinity else ""

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
    default_cube: Cube,
) -> np.ndarray:
    """Return the expected values following thresholding.
    The default cube has a 0.5 value at location (2, 2), with all other
    values set to 0. The expected result, for a "gt" or "ge" threshold
    comparator, is achieved by placing the expected value, or values for
    multi-realization data without coordinate collapsing, at the (..., 2, 2)
    location in an array full of zeros. This array matches the size of
    the input cube.

    If the comparator is changed to be "lt" or "le" then the array is
    filled with ones, and the expected value(s) is subtracted from 1 prior
    to being placed at the (..., 2, 2) location.

    This function does no more than this to avoid making the tests harder
    to follow. Cases for which the threshold value is equal to any of
    the data values (without fuzziness) meaning that the "gt" and "ge", or
    "lt" and "le" comparators would give different results are handled
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
            A list of coordinate being collapsed, or None if not collapsing.
        comparator:
            The comparator being applied in the thresholding, either "gt",
            "ge", "lt", or "le".
        default_cube:
            The input cube to the test.
    Returns:
        A data array that contains the expected values after thresholding.
    """

    try:
        n_realizations = default_cube.coord("realization").shape[0]
    except CoordinateNotFoundError:
        n_realizations = 1

    n_times = default_cube.coord("time").shape[0]

    if "l" in comparator:
        expected_single_value = 1.0 - expected_single_value
        expected_multi_value = [1.0 - value for value in expected_multi_value]
        expected_result_array = np.ones_like(default_cube.data)
    else:
        expected_result_array = np.zeros_like(default_cube.data)

    if n_realizations == 1:
        expected_result_array[..., 2, 2] = expected_single_value
    elif collapse:
        expected_result_array[..., 2, 2] = np.mean(expected_multi_value)
    else:
        expected_result_array[..., 2, 2] = expected_multi_value

    if collapse and "time" in collapse and n_times > 1:
        expected_result_array = expected_result_array[0]

    if collapse and "realization" in collapse and n_realizations > 1:
        expected_result_array = expected_result_array[0]

    return expected_result_array


@pytest.fixture(
    params=[
        deterministic_cube,
        single_realization_cube,
        multi_realization_cube,
        multi_realization_multi_time_cube,
    ]
)
def default_cube(request) -> Cube:
    """Parameterised to provide a deterministic cube, scalar realization cube,
    and multi-realization cube for testing."""
    return request.param()


@pytest.fixture
def threshold_coord(
    threshold_values: Union[float, List[float]],
    threshold_units: str,
    comparison_operator: str,
) -> DimCoord:
    """
    Generate an expected threshold coordinate based on the threshold
    values and comparison operator.

    Args:
        threshold_values:
            A list of threshold values, or a single value that will make
            up the points in the returned coordinate.
        threshold_units:
            The units in which the threshold values were specified. Used
            to set the threshold coordinate prior to conversion to the units
            of the diagnostic cube (mm/hr).
        comparison_operator:
            The comparator used for thresholding, e.g. >.
    Returns:
        A constructed threshold coordinate that is expected to be found on
        a cube thresholded with options that match the provided arguments.
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


@pytest.fixture
def landmask(mask: np.ndarray) -> Cube:
    """
    Return a landmask cube containing the data provided by the mask
    argument. Points set to 1 are land, points set to 0 are sea.

    Args:
        The mask to be included within the cube.
    Returns:
        A land mask cube.
    """
    return set_up_variable_cube(
        mask.astype(np.int8), name="land_binary_mask", units=1, spatial_grid="equalarea"
    )

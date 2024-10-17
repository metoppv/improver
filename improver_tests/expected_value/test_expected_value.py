# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ExpectedValue plugin."""
import numpy as np
import pytest
from iris.coords import CellMethod
from iris.exceptions import CoordinateNotFoundError
from numpy.testing import assert_allclose

from improver.expected_value import ExpectedValue
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.probability_manipulation import to_threshold_inequality


@pytest.fixture
def realizations_cube():
    data = np.array(
        [range(0, 9), range(10, 19), range(30, 39)], dtype=np.float32
    ).reshape([3, 3, 3])
    return set_up_variable_cube(data, realizations=[0, 1, 2])


@pytest.fixture
def percentile_cube():
    data = np.array(
        [range(10, 19), range(20, 29), range(40, 49)], dtype=np.float32
    ).reshape([3, 3, 3])
    return set_up_percentile_cube(data, percentiles=[25, 50, 75])


@pytest.fixture(params=[5, 9, 13])
def threshold_cube(request):
    thresholds = np.linspace(280, 284, 5)
    thresholds_interp = np.linspace(280, 284, request.param)
    probs = np.array([1.0, 0.7, 0.5, 0.45, 0.0])
    probs_interp = np.interp(thresholds_interp, thresholds, probs)
    data = np.broadcast_to(
        probs_interp[:, np.newaxis, np.newaxis], [len(thresholds_interp), 3, 2]
    ).astype(np.float32)
    return set_up_probability_cube(data, thresholds=thresholds_interp)


@pytest.fixture
def unequal_threshold_cube():
    thresholds = np.array([272, 280.75, 281, 282, 282.5, 284, 291])
    probs = np.array([1.0, 0.99, 0.7, 0.5, 0.45, 0.0, 0.0])
    data = np.broadcast_to(probs[:, np.newaxis, np.newaxis], [7, 3, 2]).astype(
        np.float32
    )
    return set_up_probability_cube(data, thresholds=thresholds)


@pytest.fixture
def single_bounded_threshold_cube():
    thresholds = np.array([0.0, 0.2, 0.5, 1.0, 5.0, 100.0], dtype=np.float32)
    probs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.7, 0.2, 0.1, 0.05, 0.02],
        ],
    ).transpose(1, 0)
    data = np.broadcast_to(probs[:, np.newaxis, :], [6, 2, 3])
    return set_up_probability_cube(
        data.astype(np.float32),
        thresholds=thresholds,
        variable_name="lwe_thickness_of_precipitation_amount",
        threshold_units="mm",
        spp__relative_to_threshold="greater_than",
    )


@pytest.fixture
def double_bounded_threshold_cube():
    thresholds = np.linspace(0.0, 0.9, 10, dtype=np.float32)
    probs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    ).transpose(1, 0)
    data = np.broadcast_to(probs[:, np.newaxis, :], [10, 2, 3])
    return set_up_probability_cube(
        data.astype(np.float32),
        thresholds=thresholds,
        variable_name="low_type_cloud_area_fraction",
        threshold_units="1",
        spp__relative_to_threshold="greater_than_or_equal_to",
    )


def test_process_realizations_basic(realizations_cube):
    """Check that the expected value of realisations calculates the mean and
    appropriately updates metadata."""
    expval = ExpectedValue().process(realizations_cube)
    # coords should be the same, but with the first (realization) dimcoord removed
    assert expval.coords() == realizations_cube.coords()[1:]
    # a cell method indicating mean over realizations should be added
    assert expval.cell_methods == (CellMethod("mean", "realization"),)
    # mean of the realisation coord (was the first dim of data)
    expected_data = np.mean(realizations_cube.data, axis=0)
    assert_allclose(expval.data, expected_data)


def test_process_percentile_basic(percentile_cube):
    """Check that percentiles are converted to realisations and the mean is
    calculated."""
    expval = ExpectedValue().process(percentile_cube)
    # coords should be the same, but with the first (percentile) dimcoord removed
    assert expval.coords() == percentile_cube.coords()[1:]
    # a cell method indicating mean over realizations should be added
    assert expval.cell_methods == (CellMethod("mean", "realization"),)
    # this works out to be a mean over percentiles
    # since the percentiles are equally spaced
    expected_data = np.linspace(23 + 1.0 / 3.0, 31 + 1.0 / 3.0, 9).reshape([3, 3])
    assert_allclose(expval.data, expected_data)


def test_process_threshold_basic(threshold_cube):
    """Check basic calculation of expected value using threshold data."""
    expval = ExpectedValue().process(threshold_cube)
    # threshold probablities are asymmetric, so the mean is slightly above the
    # 282 kelvin threshold
    assert_allclose(expval.data, 282.15, atol=1e-6, rtol=0.0)


def test_process_threshold_unequal(unequal_threshold_cube):
    """Check calculation of expected value using unevenly spaced threshold data."""
    expval = ExpectedValue().process(unequal_threshold_cube)
    assert_allclose(expval.data, 282.0925, atol=1e-6, rtol=0.0)


def test_process_threshold_doublebounded(double_bounded_threshold_cube):
    """Check expected value of a double bounded distribution eg. cloud cover."""
    thresholds = double_bounded_threshold_cube.coord(
        "low_type_cloud_area_fraction"
    ).points
    expval = ExpectedValue().process(double_bounded_threshold_cube)
    np.testing.assert_allclose(
        expval.data[0],
        [(thresholds[0] + thresholds[1]) / 2.0, 0.113, thresholds[-1]],
        rtol=0,
        atol=1e-7,
    )


def test_process_threshold_bounded(single_bounded_threshold_cube):
    """Check expected value of a below bounded distribution eg. precipitation."""
    expval = ExpectedValue().process(single_bounded_threshold_cube)
    np.testing.assert_allclose(
        expval.data[0], [0.0, 0.01, 3.985], rtol=0, atol=1e-6,
    )


def test_process_threshold_abovebelow(threshold_cube):
    """Check that probabilities above and below threshold are handled correctly."""
    # set up cubes that are the same other than threshold greater than/less than
    threshold_below_cube = to_threshold_inequality(threshold_cube, above=False)
    threshold_cube_airtemp = threshold_cube.coord("air_temperature")
    threshold_below_cube_airtemp = threshold_below_cube.coord("air_temperature")
    np.testing.assert_array_equal(
        threshold_cube_airtemp.points, threshold_below_cube_airtemp.points,
    )
    assert (
        threshold_cube_airtemp.attributes["spp__relative_to_threshold"]
        == "greater_than"
    )
    assert (
        threshold_below_cube_airtemp.attributes["spp__relative_to_threshold"]
        == "less_than_or_equal_to"
    )
    # calculate expected value for both, they should be the same
    expval_above = ExpectedValue().process(threshold_cube)
    expval_below = ExpectedValue().process(threshold_below_cube)
    assert expval_above == expval_below


def test_process_threshold_non_monotonic(threshold_cube):
    """Check that non-monotonic threshold data raises an exception."""
    thresholds = np.linspace(280, 284, 5)
    probs = np.array([1.0, 0.4, 0.5, 0.6, 0.0])
    probs_interp = np.interp(
        threshold_cube.coord("air_temperature").points, thresholds, probs
    )
    threshold_cube.data = np.broadcast_to(
        probs_interp[:, np.newaxis, np.newaxis],
        [threshold_cube.coord("air_temperature").shape[0], 3, 2],
    ).astype(np.float32)
    with pytest.raises(Exception, match="monotonic"):
        ExpectedValue().process(threshold_cube)


def test_process_non_probabilistic(realizations_cube, percentile_cube):
    """Check that attempting to process non-probabilistic data raises an exception."""
    realizations_cube.remove_coord("realization")
    with pytest.raises(CoordinateNotFoundError, match="realization"):
        ExpectedValue().process(realizations_cube)

    percentile_cube.remove_coord("percentile")
    with pytest.raises(CoordinateNotFoundError, match="realization"):
        ExpectedValue().process(percentile_cube)

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the TrainGAMsForSAMOS class within samos_calibration.py"""

from copy import deepcopy

import numpy as np
import pytest
from iris.cube import CubeList

from improver.calibration.samos_calibration import TrainGAMsForSAMOS
from improver_tests.calibration.samos_calibration.helper_functions import (
    create_cubes_for_gam_fitting,
    create_simple_cube,
)


@pytest.fixture
def model_specification():
    """Fixture for creating a model specification as used in SAMOS plugins."""
    return [["linear", [0], {}], ["linear", [1], {}]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model_specification": [["linear", [0], {}]]
        },  # Define a model specification but leave all other inputs as default.
        {
            "model_specification": [["linear", [0], {}]],
            "max_iter": 200,
            "tol": 0.1,
        },  # Check that inputs related to model fitting are initialised correctly.
        {
            "model_specification": [["linear", [0], {}]],
            "distribution": "gamma",
            "link": "inverse",
            "fit_intercept": False,
        },  # Check that inputs related to the model design are initialised correctly.
        {
            "model_specification": [["linear", [0], {}]],
            "window_length": 7,
            "valid_rolling_window_fraction": 0.7,
            "rolling_window_type": "trailing",
        },  # Check that inputs related to rolling window calculations are initialised
        # correctly.
    ],
)
def test__init__(kwargs):
    """Test that the class initializes variables correctly."""
    # Define the default, then update with any differently specified inputs.
    expected = {
        "model_specification": None,
        "max_iter": 100,
        "tol": 0.0001,
        "distribution": "normal",
        "link": "identity",
        "fit_intercept": True,
        "window_length": 11,
        "valid_rolling_window_fraction": 0.5,
        "rolling_window_type": "centered",
    }
    expected.update(kwargs)
    result = TrainGAMsForSAMOS(**kwargs)

    for key in kwargs.keys():
        assert getattr(result, key) == kwargs[key]


@pytest.mark.parametrize("window_length", [4, -3, 3.5])
def test_init_window_length_exception(model_specification, window_length):
    """Test that an exception is raised if the window_length is not a positive,
    odd integer."""
    msg = (
        "The window_length input must be an odd integer greater than 1. "
        f"Received: {window_length}."
    )
    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(
            model_specification=model_specification, window_length=window_length
        )


@pytest.mark.parametrize("valid_rolling_window_fraction", [-0.05, 1.05, 23])
def test_init_valid_rolling_window_fraction_exception(
    model_specification, valid_rolling_window_fraction
):
    """Test that an exception is raised if the valid_rolling_window_fraction is not a
    float between 0 and 1, inclusive."""
    msg = (
        "The valid_rolling_window_fraction input must be between 0 and 1. "
        f"Received: {valid_rolling_window_fraction}."
    )
    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(
            model_specification=model_specification,
            valid_rolling_window_fraction=valid_rolling_window_fraction,
        )


@pytest.mark.parametrize("rolling_window_type", ["invalid_type", "", 5])
def test_init_rolling_window_type_exception(model_specification, rolling_window_type):
    """Test that an exception is raised if the rolling_window_type is not one of the
    accepted strings.
    """
    msg = (
        "The rolling_window_type input must be either 'centered' or "
        f"'trailing'. Received: {rolling_window_type}."
    )
    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(
            model_specification=model_specification,
            rolling_window_type=rolling_window_type,
        )


@pytest.mark.parametrize("forecast_type", ["gridded", "spot"])
@pytest.mark.parametrize("n_realizations,n_times", [[5, 1], [5, 5], [1, 5]])
@pytest.mark.parametrize("include_blend_time", [False, True])
def test_calculate_cube_statistics(
    forecast_type,
    n_realizations,
    n_times,
    include_blend_time,
    model_specification,
):
    """Test that this method correctly calculates the mean and standard deviation of
    the input cube."""
    create_cube_kwargs = {
        "forecast_type": forecast_type,
        "n_spatial_points": 2,
        "n_realizations": n_realizations,
        "n_times": n_times,
        "fill_value": 305.0,
    }
    expected_cube_kwargs = {
        "forecast_type": forecast_type,
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": n_times,
    }

    input_cube = create_simple_cube(**create_cube_kwargs)
    if forecast_type == "spot":
        add_values = np.array([-1.0, 0.0, 0.0, 0.0, 1.0]).reshape([5, 1])
    else:
        shape = [5, 1, 1] if n_realizations != n_times else [1, 5, 1, 1]
        add_values = np.array([-1.0, 0.0, 0.0, 0.0, 1.0]).reshape(shape)

    input_cube.data = input_cube.data + np.broadcast_to(
        add_values, input_cube.data.shape
    )

    # Create cubelist containing expected mean and standard deviations cubes.
    # If n_realizations is greater than 1 then the realization dimension gets collapsed
    # so all points in the resulting mean and standard deviation cubes should be equal.
    # Otherwise, the rolling window calculations over the time dimension results in
    # means and standard deviations which vary.
    if n_realizations == 5:
        expected_mean = create_simple_cube(fill_value=305.0, **expected_cube_kwargs)
        expected_sd = create_simple_cube(fill_value=0.707106, **expected_cube_kwargs)
    else:
        shape = [5, 1] if forecast_type == "spot" else [5, 1, 1]
        expected_mean = create_simple_cube(fill_value=305.0, **expected_cube_kwargs)
        add_values_mean = np.array([-0.333333, -0.25, 0.0, 0.25, 0.333333]).reshape(
            shape
        )
        expected_mean.data = expected_mean.data + add_values_mean

        expected_sd = create_simple_cube(fill_value=0.0, **expected_cube_kwargs)
        add_values_sd = np.array([0.577350, 0.5, 0.707106, 0.5, 0.577350]).reshape(
            shape
        )
        expected_sd.data = expected_sd.data + add_values_sd

    if include_blend_time:
        # Add a blend time coordinate to the input cubes and additional cubes which is a
        # renamed copy of the pre-existing forecast_reference_time coordinate.
        blend_time_coord = input_cube.coord("forecast_reference_time").copy()
        blend_time_coord.rename("blend_time")
        input_cube.add_aux_coord(blend_time_coord)
        expected_mean.add_aux_coord(blend_time_coord)
        expected_sd.add_aux_coord(blend_time_coord)

    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS(
        model_specification=model_specification, window_length=5
    ).calculate_cube_statistics(input_cube=input_cube)

    assert expected == result


def test_calculate_cube_statistics_missing_data(model_specification):
    """Test that this method still calculates the mean and standard deviations
    correctly when there is missing data in the time period covered by the
    input_cube.
    """
    create_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
        "fill_value": 305.0,
    }

    expected_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
    }
    shape = [5, 1]

    # Set up input cube. Time coordinate is modified so that the time points are not
    # evenly spaced, but a single artificial time point can be added during processing
    # to allow rolling window calculations.
    input_cube = create_simple_cube(**create_cube_kwargs)
    add_values = np.array([-1.0, 0.0, 0.0, 0.0, 1.0]).reshape(shape)
    input_cube.data = input_cube.data + np.broadcast_to(
        add_values, input_cube.data.shape
    )

    input_cube.coord("time").points = input_cube.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    # Set up expected output cubes.
    expected_mean = create_simple_cube(fill_value=305.0, **expected_cube_kwargs)
    add_values_mean = np.array([-0.5, -0.25, 0.25, 0.25, 0.333333]).reshape(shape)
    expected_mean.data = expected_mean.data + add_values_mean
    # These values have insufficient valid data points contributing to them. Therefore,
    # these points are expected to be nans in the output.
    expected_mean.data[0, :] = np.array([np.nan, np.nan])
    expected_mean.coord("time").points = expected_mean.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    expected_sd = create_simple_cube(fill_value=0.0, **expected_cube_kwargs)
    add_values_sd = np.array([0.707106, 0.5, 0.5, 0.5, 0.577350]).reshape(shape)
    expected_sd.data = expected_sd.data + add_values_sd
    # These values have insufficient valid data points contributing to them. Therefore,
    # these points are expected to be nans in the output.
    expected_sd.data[0, :] = np.array([np.nan, np.nan])
    expected_sd.coord("time").points = expected_sd.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS(
        model_specification=model_specification, window_length=5
    ).calculate_cube_statistics(input_cube=input_cube)

    assert expected == result


def test_calculate_cube_statistics_lookback_window(model_specification):
    """Test that this method still calculates the mean and standard deviations
    correctly when there a trailing rolling window is used.

    The time points in the input_cube are modified so that they are not evenly spaced,
    but a single artificial time point can be added during processing to allow rolling
    window calculations.
    """
    create_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
        "fill_value": 305.0,
    }

    expected_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
    }
    shape = [5, 1]

    # Set up input cube. Time coordinate is modified so that the time points are not
    # evenly spaced, but a single artificial time point can be added during processing
    # to allow rolling window calculations.
    input_cube = create_simple_cube(**create_cube_kwargs)
    add_values = np.array([-1.0, 0.0, 0.0, 0.0, 1.0]).reshape(shape)
    input_cube.data = input_cube.data + np.broadcast_to(
        add_values, input_cube.data.shape
    )

    input_cube.coord("time").points = input_cube.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    # Set up expected output cubes.
    expected_mean = create_simple_cube(fill_value=305.0, **expected_cube_kwargs)
    add_values_mean = np.array([0.0, 0.0, -0.33333333, -0.25, 0.25]).reshape(shape)
    expected_mean.data = expected_mean.data + add_values_mean
    # These values have insufficient valid data points contributing to them. Therefore,
    # these points are expected to be nans in the output.
    expected_mean.data[:2, :] = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    expected_mean.coord("time").points = expected_mean.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    expected_sd = create_simple_cube(fill_value=0.0, **expected_cube_kwargs)
    add_values_sd = np.array([0.0, 0.0, 0.57735027, 0.5, 0.5]).reshape(shape)
    expected_sd.data = expected_sd.data + add_values_sd
    # These values have insufficient valid data points contributing to them. Therefore,
    # these points are expected to be nans in the output.
    expected_sd.data[:2, :] = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    expected_sd.coord("time").points = expected_sd.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS(
        model_specification=model_specification,
        window_length=5,
        rolling_window_type="trailing",
    ).calculate_cube_statistics(input_cube=input_cube)

    assert expected == result


def test_calculate_cube_statistics_insufficient_data(model_specification):
    """Test that this method returns nan for all means and standard deviation where
    there is insufficient data in the time period covered by the input_cube.
    """
    create_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
        "fill_value": 305.0,
    }

    expected_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
    }
    shape = [5, 1]

    # Set up input cube. Time coordinate is modified so that the time points are not
    # evenly spaced, but a single artificial time point can be added during processing
    # to allow rolling window calculations.
    input_cube = create_simple_cube(**create_cube_kwargs)
    add_values = np.array([-1.0, 0.0, 0.0, 0.0, 1.0]).reshape(shape)
    input_cube.data = input_cube.data + np.broadcast_to(
        add_values, input_cube.data.shape
    )

    input_cube.coord("time").points = input_cube.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    # Set up expected output cubes.
    expected_mean = create_simple_cube(fill_value=np.nan, **expected_cube_kwargs)
    expected_mean.coord("time").points = expected_mean.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    expected_sd = create_simple_cube(fill_value=np.nan, **expected_cube_kwargs)
    expected_sd.coord("time").points = expected_sd.coord("time").points + np.array(
        [0, 86400, 86400, 86400, 86400], dtype=np.int64
    )

    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS(
        model_specification=model_specification, window_length=11
    ).calculate_cube_statistics(input_cube=input_cube)

    assert expected == result


def test_calculate_cube_statistics_period_diagnostic(model_specification):
    """Test that this method correctly calculates the mean and standard deviation when
    the input cube contains a period diagnostic.
    """
    create_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
        "fill_value": 305.0,
    }

    expected_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 5,
    }

    time_bounds = np.array(
        [
            [1510200000, 1510286400],
            [1510286400, 1510372800],
            [1510372800, 1510459200],
            [1510459200, 1510545600],
            [1510545600, 1510632000],
        ],
    )

    input_cube = create_simple_cube(**create_cube_kwargs)
    expected_mean = create_simple_cube(fill_value=305.0, **expected_cube_kwargs)
    expected_sd = create_simple_cube(fill_value=0.0, **expected_cube_kwargs)

    input_cube.coord("time").bounds = time_bounds

    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS(
        model_specification=model_specification, window_length=5
    ).calculate_cube_statistics(input_cube=input_cube)

    assert expected == result


def test_calculate_cube_statistics_exception(model_specification):
    """Test that this method raises the correct exception when a rolling window
    calculation over the time coordinate is required to calculate the cube statistics,
    but the time coordinate has unevenly spaced points.
    """
    create_cube_kwargs = {
        "forecast_type": "spot",
        "n_spatial_points": 2,
        "n_realizations": 1,
        "n_times": 3,
        "fill_value": 305.0,
    }

    # Returns cube with 3 time points at one day intervals.
    test_cube = create_simple_cube(**create_cube_kwargs)

    # Modify the time points so that they are not equally spaced
    test_cube.coord("time").points = test_cube.coord("time").points + np.array(
        [0, 0, 1], dtype=np.int64
    )

    msg = (
        "The increments between points in the time coordinate of the input "
        "cube must be divisible by the smallest increment between points to "
        "allow for rolling window calculations to be performed over the time "
        "coordinate. The increments between points in the time coordinate "
        "were: \\[86400 86401\\]. The smallest increment was: 86400."
    )
    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(
            model_specification=model_specification
        ).calculate_cube_statistics(input_cube=test_cube)


@pytest.mark.parametrize(
    "include_altitude,spatial_model_specification,n_realizations,expected",
    [
        [
            False,
            [["linear", [0], {}], ["linear", [1], {}]],
            11,
            np.array([288.1492261877016, 0.5494898838013565], dtype=np.float64),
        ],
        [
            True,
            [["linear", [0], {}], ["linear", [1], {}]],
            11,
            np.array([288.16844872649284, 0.5496814185908583], dtype=np.float64),
        ],
        [
            False,
            [["tensor", [0, 1], {}]],
            11,
            np.array([288.1285965937152, 0.5290237015452911], dtype=np.float64),
        ],
        [
            True,
            [["tensor", [0, 1], {}]],
            11,
            np.array([288.13439901991666, 0.5280646018499808], dtype=np.float64),
        ],
        [
            False,
            [["linear", [0], {}], ["linear", [1], {}]],
            1,
            np.array([288.1659800896903, 0.561020575955558], dtype=np.float64),
        ],
        [
            True,
            [["linear", [0], {}], ["linear", [1], {}]],
            1,
            np.array([288.15797722833526, 0.5427105296998994], dtype=np.float64),
        ],
        [
            False,
            [["tensor", [0, 1], {}]],
            1,
            np.array([287.97628879844785, 0.5074109794042518], dtype=np.float64),
        ],
        [
            True,
            [["tensor", [0, 1], {}]],
            1,
            np.array([287.9622424769424, 0.48765295694648797], dtype=np.float64),
        ],
    ],
)
def test_process(
    include_altitude, spatial_model_specification, n_realizations, expected
):
    """Test that this method takes an input cube, a list of features, and possibly
    additional predictor cubes and correctly returns a fitted GAM.
    """
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    full_model_specification = deepcopy(spatial_model_specification)
    features = ["latitude", "longitude"]
    gam_kwargs = {
        "max_iter": 250,
        "tol": 0.01,
        "distribution": "normal",
        "link": "identity",
        "fit_intercept": True,
    }
    n_spatial_points = 5
    n_times = 20

    if include_altitude:
        features.append("surface_altitude")
        full_model_specification.append(["spline", [2], {}])

    input_cube, additional_cubes = create_cubes_for_gam_fitting(
        n_spatial_points,
        n_realizations,
        n_times,
        include_altitude,
    )

    result_gams = TrainGAMsForSAMOS(full_model_specification, **gam_kwargs).process(
        input_cube, features, additional_cubes
    )

    # Make predictions from the fitted GAMs to compare to expected results.
    new_predictors = np.full([1, len(features)], 0.0)
    mean_prediction = result_gams[0].predict(new_predictors)
    sd_prediction = result_gams[1].predict(new_predictors)

    # Check that our arguments have been used correctly in the GAM models. Also check
    # that the GAMs were fitted using the correct data.
    for gam in result_gams:
        for key in ["max_iter", "tol", "fit_intercept"]:
            assert gam.get_params()[key] == gam_kwargs[key]
        assert f"{gam.distribution}" == gam_kwargs["distribution"]
        assert f"{gam.link}" == gam_kwargs["link"]
        assert gam.statistics_["n_samples"] == n_times * n_spatial_points**2
        assert gam.statistics_["m_features"] == len(features)

    np.testing.assert_almost_equal(mean_prediction[0], expected[0])
    np.testing.assert_almost_equal(sd_prediction[0], expected[1])


def test_process_insufficient_data():
    """Test that this method returns None when there is insufficient data to fit the
    GAMs.

    In this test we provide 5 days of training data but specify a window_length of 11
    days meaning that our training data does not fulfil the 50%
    valid_rolling_window_fraction. The values are set to nan as an indicator of
    incompleteness, which results in GAMFit raising a warning and returning None.
    """
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    full_model_specification = [["linear", [0], {}], ["linear", [1], {}]]
    n_realizations = 1

    features = ["latitude", "longitude"]
    gam_kwargs = {
        "max_iter": 250,
        "tol": 0.01,
        "distribution": "normal",
        "link": "identity",
        "fit_intercept": True,
        "window_length": 11,
        "valid_rolling_window_fraction": 0.5,
    }
    n_spatial_points = 5
    n_times = 5

    input_cube, additional_cubes = create_cubes_for_gam_fitting(
        n_spatial_points,
        n_realizations,
        n_times,
        False,
    )

    msg = (
        "After removing NaN values from the input data, there are no "
        "remaining data points to fit the GAM model. No model has been fitted."
    )
    with pytest.warns(UserWarning, match=msg):
        result_gams = TrainGAMsForSAMOS(full_model_specification, **gam_kwargs).process(
            input_cube, features, additional_cubes
        )
        assert result_gams is None


@pytest.mark.parametrize("exception", ["no_time_coord", "single_point_time_coord"])
def test_missing_required_coordinates_exception(exception, model_specification):
    """Test that the correct exceptions are raised when the input cube does not contain
    suitable realization or time coordinates."""
    # Create a cube with no realization coordinate and a single point time coordinate.
    input_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=2,
        n_realizations=1,
        n_times=1,
        fill_value=305.0,
    )
    features = ["latitude", "longitude"]

    if exception == "no_time_coord":
        input_cube.remove_coord("time")
        msg = (
            "The input cube must contain at least one of a realization or time "
            "coordinate in order to allow the calculation of means and standard "
            "deviations."
        )
        with pytest.raises(ValueError, match=msg):
            TrainGAMsForSAMOS(model_specification).process(input_cube, features)
    elif exception == "single_point_time_coord":
        msg = (
            "The input cube does not contain a realization coordinate. In order to "
            "calculate means and standard deviations the time coordinate must "
            "contain more than one point."
        )
        with pytest.warns(UserWarning, match=msg):
            result = TrainGAMsForSAMOS(model_specification).process(
                input_cube, features
            )
            assert result is None

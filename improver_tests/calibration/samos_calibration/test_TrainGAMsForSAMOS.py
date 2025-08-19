# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the TrainGAMsForSAMOS class within samos_calibration.py"""

from copy import deepcopy

import numpy as np
import pytest
from iris.coords import CellMethod
from iris.cube import CubeList

from improver.calibration.samos_calibration import TrainGAMsForSAMOS
from improver_tests.calibration.samos_calibration.helper_functions import (
    create_cubes_for_gam_fitting,
    create_simple_cube,
)

np.random.seed(1)


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
    ],
)
def test__init__(kwargs):
    """Test that the class initializes variables correctly."""
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    # Define the default, then update with any differently specified inputs.
    expected = {
        "model_specification": None,
        "max_iter": 100,
        "tol": 0.0001,
        "distribution": "normal",
        "link": "identity",
        "fit_intercept": True,
    }
    expected.update(kwargs)
    result = TrainGAMsForSAMOS(**kwargs)

    for key in kwargs.keys():
        assert getattr(result, key) == kwargs[key]


@pytest.mark.parametrize("forecast_type", ["gridded", "spot"])
@pytest.mark.parametrize("n_realizations,n_times", [[5, 1], [5, 5], [1, 5]])
@pytest.mark.parametrize("include_blend_time", [False, True])
def test_calculate_cube_statistics(
    forecast_type,
    n_realizations,
    n_times,
    include_blend_time,
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

    if n_realizations > 1:
        # Expect statistics to be calculated over the realization dimension.
        expected_mean.add_cell_method(CellMethod("mean", coords="realization"))
        expected_sd.add_cell_method(
            CellMethod("standard_deviation", coords="realization")
        )
    else:
        # Expect statistics to be calculated over the time dimension.
        expected_mean.add_cell_method(CellMethod("mean", coords="time"))
        expected_sd.add_cell_method(CellMethod("standard_deviation", coords="time"))
    if include_blend_time:
        # Add a blend time coordinate to the input cubes and additional cubes which is a
        # renamed copy of the pre-existing forecast_reference_time coordinate.
        blend_time_coord = input_cube.coord("forecast_reference_time").copy()
        blend_time_coord.rename("blend_time")
        input_cube.add_aux_coord(blend_time_coord)
        expected_mean.add_aux_coord(blend_time_coord)
        expected_sd.add_aux_coord(blend_time_coord)

    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS.calculate_cube_statistics(input_cube)

    assert expected == result


def test_calculate_cube_statistics_exception():
    """Test that this method raises the correct exception when a rolling window
    calculation over the time coordinate is required to calculate the cube statistics,
    but the time coordinate as unevenly spaced points.
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
        [0, 0, 1]
    )

    msg = (
        "In order to extend the time coordinate to permit calculation of means and "
        "standard deviations, the existing points on the time coordinate must be "
        "evenly spaced."
    )

    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS.calculate_cube_statistics(test_cube)


@pytest.mark.parametrize(
    "include_altitude,spatial_model_specification,n_realizations,expected",
    [
        [
            False,
            [["linear", [0], {}], ["linear", [1], {}]],
            11,
            [288.15298254, 0.48331375],
        ],
        [
            True,
            [["linear", [0], {}], ["linear", [1], {}]],
            11,
            [288.15007863, 0.49052109],
        ],
        [False, [["tensor", [0, 1], {}]], 11, [288.22168378, 0.5010813]],
        [True, [["tensor", [0, 1], {}]], 11, [288.1290978, 0.44678148]],
        [
            False,
            [["linear", [0], {}], ["linear", [1], {}]],
            1,
            [288.17666906, 0.48082173],
        ],
        [
            True,
            [["linear", [0], {}], ["linear", [1], {}]],
            1,
            [288.14031069, 0.42215065],
        ],
        [False, [["tensor", [0, 1], {}]], 1, [288.14178362, 0.44964758]],
        [True, [["tensor", [0, 1], {}]], 1, [288.14402253, 0.51517678]],
    ],
)
def test_process(
    include_altitude, spatial_model_specification, n_realizations, expected
):
    """Test that this method takes an input cube, a list of features, and possibly
    additional predictor cubes and correctly returns a fitted GAM.
    """
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
    elif exception == "single_point_time_coord":
        msg = (
            "The input cube does not contain a realization coordinate. In order to "
            "calculate means and standard deviations the time coordinate must "
            "contain more than one point."
        )

    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(model_specification).process(input_cube, features)

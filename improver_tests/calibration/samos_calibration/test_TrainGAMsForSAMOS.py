# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the TrainGAMsForSAMOS class within samos_calibration.py"""
from copy import deepcopy
import numpy as np
import pytest
from iris.cube import CubeList
from iris.coords import CellMethod
from improver.calibration.samos_calibration import TrainGAMsForSAMOS
from improver_tests.calibration.samos_calibration.helper_functions import create_cube


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
@pytest.mark.parametrize("realizations,times", [[2, 1], [2, 2], [1, 2]])
@pytest.mark.parametrize("include_blend_time", [False, True])
def test_calculate_cube_statistics(
    forecast_type,
    realizations,
    times,
    include_blend_time,
):
    """Test that this method correctly calculates the mean and standard deviation of
    the input cube."""
    create_cube_kwargs = {
        "forecast_type": forecast_type,
        "n_spatial_points": 2,
        "realizations": realizations,
        "times": times,
        "fill_value": 305.0
    }
    expected_cube_kwargs = {
        "forecast_type": forecast_type,
        "n_spatial_points": 2,
        "realizations": 1,
        "times": times,
    }

    input_cube = create_cube(**create_cube_kwargs)

    # Create cubelist containing expected mean and standard deviations cubes.
    expected_mean = create_cube(fill_value=305, **expected_cube_kwargs)
    expected_sd = create_cube(fill_value=0, **expected_cube_kwargs)
    if realizations > 1:
        # Expect statistics to be calculated over the realization dimension.
        expected_mean.add_cell_method(CellMethod("mean", coords="realization"))
        expected_sd.add_cell_method(
            CellMethod("standard_deviation", coords="realization")
        )
    else:
        # Expect statistics to be calculated over the time dimension.
        expected_mean.add_cell_method(CellMethod("mean", coords="time"))
        expected_sd.add_cell_method(
            CellMethod("standard_deviation", coords="time")
        )
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


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
@pytest.mark.parametrize(
    "spatial_model_specification",  # Define the terms which lat and lon contribute to.
    [
        [["linear", [0], {}], ["linear", [1], {}]],
        [["tensor", [0, 1], {}]],
    ]
)
@pytest.mark.parametrize("n_realizations", [11, 1])
def test_process(
    include_altitude,
    include_land_fraction,
    spatial_model_specification,
    n_realizations,
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
        "fit_intercept": True
    }
    n_spatial_points = 5
    n_times = 20
    input_cube = create_cube(
        forecast_type="gridded",
        n_spatial_points=n_spatial_points,
        realizations=n_realizations,
        times=n_times,
        fill_value=273.15
    )
    # Create array of data to add to cube which increases with lat and lon, so that
    # these features are useful in the GAMs.
    lat_addition = np.linspace(
        start=0, stop=15, num=n_spatial_points
    ).reshape([n_spatial_points, 1])
    lon_addition = np.linspace(
        start=0, stop=15, num=n_spatial_points
    ).reshape([1, n_spatial_points])
    addition = lat_addition + lon_addition  # 10x10 array
    addition = np.broadcast_to(
        addition,
        shape=input_cube.data.shape
    )
    # Create array of random noise which increases with lat and lon, so that there is
    # some variance in the data to model in the standard deviation GAM.
    noise = np.random.normal(loc=0.0, scale=addition/30)
    input_cube.data = input_cube.data + addition + noise

    additional_cubes = []
    if include_altitude:
        # Create an altitude cube with small values in the centre of the domain and
        # large values around the outside of the domain, with a smooth gradient in
        # between.
        altitude_cube = create_cube(
            forecast_type="gridded",
            n_spatial_points=n_spatial_points,
            realizations=1,
            times=1,
            fill_value=1000.0
        )
        altitude_cube.rename("surface_altitude")

        lat_multiplier = np.abs(np.linspace(
            start=-1, stop=1, num=n_spatial_points
        ).reshape([n_spatial_points, 1]))  # 1 at ends, close to 0 in the middle.
        lon_multiplier = np.abs(np.linspace(
            start=-1, stop=1, num=n_spatial_points
        ).reshape([1, n_spatial_points]) - 1)  # 1 at ends, close to 0 in the middle.
        altitude_multiplier = lat_multiplier * lon_multiplier

        altitude_cube.data = altitude_cube.data * altitude_multiplier
        additional_cubes.append(altitude_cube)
        features.append("surface_altitude")
        full_model_specification.append(
            ["spline", [features.index("surface_altitude")], {}]
        )

        # Subtract values from input_cube data which increase with altitude.
        altitude_multiplier = np.broadcast_to(
            altitude_multiplier,
            shape=input_cube.data.shape
        )
        input_cube.data = input_cube.data - (5.0 * altitude_multiplier)

    if include_land_fraction:
        # Create a land fraction cube with full land in the top left corner of the
        # domain, full sea in the bottom right, and a smooth gradient of fractions in
        # between.
        lf_cube = create_cube(
            forecast_type="gridded",
            n_spatial_points=n_spatial_points,
            realizations=1,
            times=1,
            fill_value=1
        )
        lf_cube.rename("land_fraction")

        lat_multiplier = np.linspace(
            start=1.0, stop=0.0, num=n_spatial_points
        ).reshape([n_spatial_points, 1])
        lon_multiplier = np.linspace(
            start=1.0, stop=0.0, num=n_spatial_points
        ).reshape([1, n_spatial_points])
        lf_multiplier = lat_multiplier * lon_multiplier

        lf_cube.data = lf_cube.data * lf_multiplier
        additional_cubes.append(lf_cube)
        features.append("land_fraction")
        full_model_specification.append(
            ["spline", [features.index("land_fraction")], {}]
        )

        # Add values to input_cube data which increase with land fraction.
        lf_multiplier = np.broadcast_to(
            lf_multiplier,
            shape=input_cube.data.shape
        )
        input_cube.data = input_cube.data + (2.0 * lf_multiplier)

    result_gams = TrainGAMsForSAMOS(full_model_specification, **gam_kwargs).process(
        input_cube, features, additional_cubes
    )

    # Check that our arguments have been used correctly in the GAM models. Also check
    # that the GAMs were fitted using the correct data.
    for gam in result_gams:
        for key in ["max_iter", "tol", "fit_intercept"]:
            assert gam.get_params()[key] == gam_kwargs[key]
        assert f"{gam.distribution}" == gam_kwargs["distribution"]
        assert f"{gam.link}" == gam_kwargs["link"]
        assert gam.statistics_["n_samples"] == n_times * n_spatial_points ** 2
        assert gam.statistics_["m_features"] == len(features)

    # Check that we've ended up with 2 different GAM models.
    assert result_gams[0] != result_gams[1]


@pytest.mark.parametrize("exception", ["no_time_coord", "single_point_time_coord"])
def test_missing_required_coordinates_exception(
    exception,
    model_specification
):
    """Test that the correct exceptions are raised when the input cube does not contain
    suitable realization or time coordinates."""
    # Create a cube with no realization coordinate and a single point time coordinate.
    input_cube = create_cube(
        forecast_type="gridded",
        n_spatial_points=2,
        realizations=1,
        times=1,
        fill_value=305.0
    )
    features = ["latitude", "longitude"]

    if exception is "no_time_coord":
        input_cube.remove_coord("time")
        msg = ("The input cube must contain at least one of a realization or time "
               "coordinate in order to allow the calculation of means and standard "
               "deviations.")
    elif exception is "single_point_time_coord":
        msg = ("The input cube does not contain a realization coordinate. In order to "
               "calculate means and standard deviations the time coordinate must "
               "contain more than one point.")

    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(model_specification).process(input_cube, features)

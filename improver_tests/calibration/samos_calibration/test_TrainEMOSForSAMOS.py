# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the TrainEMOSForSAMOS class within samos_calibration.py"""

import numpy as np
import pytest

from improver.calibration.samos_calibration import TrainEMOSForSAMOS, TrainGAMsForSAMOS
from improver_tests.calibration.samos_calibration.helper_functions import (
    create_cubes_for_gam_fitting,
    create_simple_cube,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distribution": "norm"},
        {
            "distribution": "norm",
            "emos_kwargs": {"point_by_point": True, "use_default_initial_guess": True},
        },
        {"distribution": "truncnorm", "emos_kwargs": {}},
    ],
)
def test__init__(kwargs):
    """Test that the class initializes variables correctly."""
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    # Define the default, then update with any differently specified inputs.
    expected = {
        "distribution": None,
        "emos_kwargs": None,
    }
    expected.update(kwargs)
    result = TrainEMOSForSAMOS(**kwargs)

    for key in kwargs.keys():
        assert getattr(result, key) == kwargs[key]


def test_climate_anomaly_emos():
    """Test that the climate_anomaly_emos method returns the expected results."""
    np.random.seed(1)  # Set random seed to enable test to be reproducible.

    create_cube_kwargs = {
        "forecast_type": "gridded",
        "n_spatial_points": 2,
        "n_realizations": 5,
        "n_times": 10,
        "fixed_forecast_period": True,
    }

    forecast_raw = create_simple_cube(fill_value=300.0, **create_cube_kwargs)
    forecast_mean = create_simple_cube(fill_value=300.0, **create_cube_kwargs)
    forecast_sd = create_simple_cube(fill_value=1.0, **create_cube_kwargs)
    forecast_cubes = [forecast_raw, forecast_mean, forecast_sd]

    create_cube_kwargs["n_realizations"] = 1
    truth_raw = create_simple_cube(fill_value=300.0, **create_cube_kwargs)
    truth_mean = create_simple_cube(fill_value=300.0, **create_cube_kwargs)
    truth_sd = create_simple_cube(fill_value=1.0, **create_cube_kwargs)
    truth_cubes = [truth_raw, truth_mean, truth_sd]
    for cube in truth_cubes:
        cube.remove_coord("forecast_reference_time")
        cube.remove_coord("forecast_period")

    result = TrainEMOSForSAMOS(distribution="norm").climate_anomaly_emos(
        forecast_cubes=forecast_cubes, truth_cubes=truth_cubes
    )

    expected_names = [
        "emos_coefficient_alpha",
        "emos_coefficient_beta",
        "emos_coefficient_gamma",
        "emos_coefficient_delta",
    ]
    expected_data = [8.196854e-06, 9.267808e-05, -2.6859516e-05, 0.99098396]

    for i, cube in enumerate(result):
        assert expected_names[i] == cube.name()
        assert expected_data[i] == cube.data


@pytest.mark.parametrize("include_altitude", [False, True])
def test_process(include_altitude):
    """Test that the process method returns the expected results."""
    np.random.seed(1)  # Set random seed to enable test to be reproducible.

    # Set up model terms for spatial predictors.
    model_specification = [["linear", [0], {}], ["linear", [1], {}]]
    features = ["latitude", "longitude"]
    n_spatial_points = 5
    n_realizations = 5
    n_times = 20

    if include_altitude:
        features.append("surface_altitude")
        model_specification.append(["spline", [features.index("surface_altitude")], {}])

    forecast_cube, additional_cubes = create_cubes_for_gam_fitting(
        n_spatial_points=n_spatial_points,
        n_realizations=n_realizations,
        n_times=n_times,
        include_altitude=include_altitude,
        fixed_forecast_period=True,
    )

    truth_cube, _ = create_cubes_for_gam_fitting(
        n_spatial_points=n_spatial_points,
        n_realizations=1,
        n_times=n_times,
        include_altitude=include_altitude,
        fixed_forecast_period=True,
    )
    truth_cube.remove_coord("forecast_reference_time")
    truth_cube.remove_coord("forecast_period")

    forecast_gams = TrainGAMsForSAMOS(model_specification).process(
        forecast_cube, features, additional_cubes
    )
    truth_gams = TrainGAMsForSAMOS(model_specification).process(
        truth_cube, features, additional_cubes
    )

    result = TrainEMOSForSAMOS(distribution="norm").process(
        historic_forecasts=forecast_cube,
        truths=truth_cube,
        forecast_gams=forecast_gams,
        truth_gams=truth_gams,
        gam_features=features,
        gam_additional_fields=additional_cubes,
    )

    expected_names = [
        "emos_coefficient_alpha",
        "emos_coefficient_beta",
        "emos_coefficient_gamma",
        "emos_coefficient_delta",
    ]
    if include_altitude:
        expected_data = [-0.04414984, -0.030644448, 0.00040429688, 1.0839844]
    else:
        expected_data = [-0.10599241, 0.035578612, 0.0038651018, 1.066569]

    for i, cube in enumerate(result):
        assert expected_names[i] == cube.name()
        assert expected_data[i] == cube.data

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplySAMOS class within samos_calibration.py"""

import numpy as np
import pytest
from iris.cube import CubeList

from improver.calibration.samos_calibration import (
    ApplySAMOS,
    TrainGAMsForSAMOS,
)
from improver.threshold import Threshold
from improver_tests.calibration.emos_calibration.test_ApplyEMOS import (
    build_coefficients_cubelist,
)
from improver_tests.calibration.samos_calibration.helper_functions import (
    FORECAST_ATTRIBUTES,
    create_cubes_for_gam_fitting,
    create_simple_cube,
)

EXPECTED_DATA_DICT = {
    "real_real_emosalt_notgamalt": np.array(
        [
            [
                [313.50665, 305.0316, 287.89963],
                [280.07907, 287.8345, 296.38815],
                [333.15, 333.15, 304.0257],
            ],
            [
                [313.4974, 304.8424, 288.25708],
                [279.9784, 288.0995, 295.84637],
                [333.15, 333.15, 302.26242],
            ],
            [
                [313.50204, 304.65323, 288.61453],
                [280.17975, 287.967, 296.11728],
                [333.15, 333.15, 303.14404],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_notemosalt_notgamalt": np.array(
        [
            [
                [273.1705, 280.7735, 287.9103],
                [280.7542, 288.1842, 296.4416],
                [288.2182, 295.7106, 304.0292],
            ],
            [
                [273.1613, 280.58432, 288.2678],
                [280.65353, 288.44922, 295.8998],
                [288.01764, 295.00946, 302.26596],
            ],
            [
                [273.1659, 280.39514, 288.62524],
                [280.8549, 288.3167, 296.17072],
                [288.41873, 296.41177, 303.1476],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_emosalt_gamalt": np.array(
        [
            [
                [313.50665, 305.0316, 287.89963],
                [280.07907, 287.8345, 296.38815],
                [333.15, 333.15, 304.0257],
            ],
            [
                [313.4974, 304.8424, 288.25708],
                [279.9784, 288.0995, 295.84637],
                [333.15, 333.15, 302.26242],
            ],
            [
                [313.50204, 304.65323, 288.61453],
                [280.17975, 287.967, 296.11728],
                [333.15, 333.15, 303.14404],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_notemosalt_gamalt": np.array(
        [
            [
                [273.1705, 280.77353, 287.91034],
                [280.7542, 288.18423, 296.4416],
                [288.2182, 295.7106, 304.02924],
            ],
            [
                [273.1613, 280.58432, 288.2678],
                [280.65353, 288.44922, 295.8998],
                [288.01764, 295.00946, 302.26596],
            ],
            [
                [273.1659, 280.39514, 288.62524],
                [280.8549, 288.3167, 296.17072],
                [288.41873, 296.41177, 303.1476],
            ],
        ],
        dtype=np.float32,
    ),
    "real_prob_emosalt_notgamalt": np.array(
        [
            [
                [1.0000, 1.0000, 0.6862],
                [0.0000, 0.4333, 1.0000],
                [1.0000, 1.0000, 1.0000],
            ],
            [
                [1.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [1.0000, 1.0000, 0.5439],
            ],
        ],
        dtype=np.float32,
    ),
    "prob_prob_notemosalt_notgamalt": np.array(
        [
            [
                [0.9772, 1.0000, 0.9974],
                [1.0000, 0.9974, 0.9974],
                [0.9974, 0.9974, 0.9974],
            ],
            [
                [0.0228, 0.9772, 0.9918],
                [0.9772, 0.9918, 0.9918],
                [0.9918, 0.9918, 0.9918],
            ],
            [
                [0.0000, 0.0228, 0.9772],
                [0.0228, 0.9772, 0.9772],
                [0.9772, 0.9772, 0.9772],
            ],
        ],
        dtype=np.float32,
    ),
    "perc_perc_notemosalt_notgamalt": np.array(
        [
            [
                [273.1613, 280.39514, 287.91034],
                [280.65353, 288.18423, 295.8998],
                [288.01764, 295.00946, 302.266],
            ],
            [
                [273.1659, 280.58432, 288.2678],
                [280.7542, 288.3167, 296.17072],
                [288.2182, 295.7106, 303.1476],
            ],
            [
                [273.1705, 280.7735, 288.62524],
                [280.8549, 288.44922, 296.44162],
                [288.41873, 296.41177, 304.02924],
            ],
        ],
        dtype=np.float32,
    ),
    "perc_prob_notemosalt_notgamalt": np.array(
        [
            [
                [0.0000, 0.0000, 0.6933],
                [0.0000, 0.9466, 1.0000],
                [0.7685, 1.0000, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5450],
            ],
        ],
        dtype=np.float32,
    ),
    "real_prob_notemosalt_notgamalt": np.array(
        [
            [
                [0.0000, 0.0000, 0.6933],
                [0.0000, 0.9466, 1.0000],
                [0.7685, 1.0000, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5450],
            ],
        ],
        dtype=np.float32,
    ),
}


def get_expected_result(
    input_format, output_format, emos_include_altitude, gam_include_altitude
):
    """Function to get the expected result of the process() method."""
    expected_name = "air_temperature"
    if output_format == "probability":
        expected_name = "probability_of_air_temperature_above_threshold"

    expected_dims = (
        ["air_temperature", "latitude", "longitude"]
        if output_format == "probability"
        else [output_format, "latitude", "longitude"]
    )

    emos_field = "emosalt" if emos_include_altitude else "notemosalt"
    gam_field = "gamalt" if gam_include_altitude else "notgamalt"
    expected_data_key = (
        f"{input_format[:4]}_{output_format[:4]}_{emos_field}_{gam_field}"
    )
    try:
        expected_data = EXPECTED_DATA_DICT[expected_data_key]
    except KeyError:
        expected_data = None

    return expected_data, expected_name, expected_dims


@pytest.mark.parametrize("percentiles", [None, 5, [5], [5, 10, 15]])
def test__init__(percentiles):
    """Test that the class initializes variables correctly."""
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    if percentiles == 5:
        msg = "'int' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            ApplySAMOS(percentiles)
    else:
        result = ApplySAMOS(percentiles)
        assert getattr(result, "percentiles") == percentiles


@pytest.mark.parametrize(
    "input_format,output_format,emos_include_altitude,gam_include_altitude",
    [
        ["realization", "realization", False, False],
        ["realization", "realization", True, False],
        ["realization", "realization", False, True],
        ["realization", "realization", True, True],
        ["realization", "probability", False, False],
        ["realization", "probability", True, False],
        ["percentile", "percentile", False, False],
        ["percentile", "probability", False, False],
        ["probability", "probability", False, False],
    ],
)
def test_process(
    input_format, output_format, emos_include_altitude, gam_include_altitude
):
    """Test that the process method returns the expected results."""
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    # Set up model terms for spatial predictors.
    model_specification = [["linear", [0], {}], ["linear", [1], {}]]
    features = ["latitude", "longitude"]
    n_spatial_points = 3
    n_realizations = 3
    n_times = 20

    forecast_cube, additional_cubes = create_cubes_for_gam_fitting(
        n_spatial_points=n_spatial_points,
        n_realizations=n_realizations,
        n_times=n_times,
        include_altitude=emos_include_altitude,
        fixed_forecast_period=True,
    )

    for key, value in FORECAST_ATTRIBUTES.items():
        forecast_cube.attributes[key] = value

    forecast_gams = TrainGAMsForSAMOS(model_specification).process(
        forecast_cube, features, additional_cubes
    )

    forecast_slice = next(forecast_cube.slices_over(["forecast_reference_time"]))
    if emos_include_altitude:
        emos_coefficients = build_coefficients_cubelist(
            forecast_slice,
            [0, [0.9, 0.1], 0, 1],
            CubeList([forecast_slice, additional_cubes[0]]),
        )
    else:
        emos_coefficients = build_coefficients_cubelist(
            forecast_slice,
            [0, 1, 0, 1],
            CubeList([forecast_slice]),
        )

    if input_format == "realization":
        process_kwargs = {}
    elif input_format == "percentile":
        forecast_slice.coord("realization").rename("percentile")
        forecast_slice.coord("percentile").points = np.array([25, 50, 75])
        process_kwargs = {
            "realizations_count": n_realizations,
        }
    elif input_format == "probability":
        thresholds = [273, 278, 283]
        forecast_slice = Threshold(thresholds, collapse_coord="realization").process(
            forecast_slice
        )
        process_kwargs = {
            "realizations_count": n_realizations,
        }

    if output_format == "probability" and not input_format == "probability":
        thresholds = [288, 303]
        prob_template = Threshold(thresholds, collapse_coord=input_format).process(
            forecast_slice
        )
        process_kwargs.update({"prob_template": prob_template})

    expected_data, expected_name, expected_dims = get_expected_result(
        input_format, output_format, emos_include_altitude, gam_include_altitude
    )

    result = ApplySAMOS(percentiles=None).process(
        forecast=forecast_slice,
        forecast_gams=forecast_gams,
        truth_gams=forecast_gams,
        gam_features=features,
        emos_coefficients=emos_coefficients,
        gam_additional_fields=None,
        emos_additional_fields=additional_cubes,
        **process_kwargs,
    )

    np.testing.assert_array_almost_equal(result.data, expected_data, decimal=4)
    assert result.name() == expected_name
    assert [c.name() for c in result.coords(dim_coords=True)] == expected_dims


@pytest.mark.parametrize("output_format", ["realization", "probability"])
def test_process_gamma(output_format):
    """Test that the process method returns the expected results when using a gamma
    distribution."""
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    # Set up model terms for spatial predictors.
    model_specification = [["linear", [0], {}], ["linear", [1], {}]]
    features = ["latitude", "longitude"]
    n_spatial_points = 3
    n_realizations = 3
    n_times = 20
    process_kwargs = {}

    forecast_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=n_spatial_points,
        n_realizations=n_realizations,
        n_times=n_times,
        fill_value=273.15,
        fixed_forecast_period=True,
    )

    for key, value in FORECAST_ATTRIBUTES.items():
        forecast_cube.attributes[key] = value

    addition = np.abs(
        np.linspace(start=2, stop=5, num=n_spatial_points).reshape(
            [n_spatial_points, 1]
        )
    )  # Create increasing trend in data with latitude.
    addition = np.broadcast_to(addition, shape=forecast_cube.data.shape)
    rng = np.random.RandomState(210825)  # Set seed for reproducible results.
    noise = rng.gamma(
        shape=addition, scale=2.0
    )  # Create gamma distributed noise which increases with latitude.
    forecast_cube.data = forecast_cube.data + addition + noise

    forecast_gams = TrainGAMsForSAMOS(model_specification).process(
        forecast_cube, features, CubeList([])
    )

    forecast_slice = next(forecast_cube.slices_over(["forecast_reference_time"]))
    emos_coefficients = build_coefficients_cubelist(
        forecast_slice,
        [0, 1, 0, 1],
        CubeList([forecast_slice]),
    )

    if output_format == "probability":
        thresholds = [278, 283]
        prob_template = Threshold(thresholds, collapse_coord="realization").process(
            forecast_slice
        )
        process_kwargs.update({"prob_template": prob_template})
        expected_data = np.array(
            [
                [
                    [0.4935, 0.8348, 0.6408],
                    [0.9462, 0.9999, 1.0000],
                    [0.9995, 0.9911, 0.9946],
                ],
                [
                    [0.0018, 0.4832, 0.1224],
                    [0.6000, 0.9747, 0.9034],
                    [0.9780, 0.9165, 0.8773],
                ],
            ],
            dtype=np.float32,
        )
        expected_name = "probability_of_air_temperature_above_threshold"
        expected_dims = ["air_temperature", "latitude", "longitude"]
    else:  # output_format == "realization"
        expected_data = np.array(
            [
                [
                    [279.1376, 282.7929, 279.1829],
                    [286.421, 288.3678, 284.4564],
                    [288.2437, 290.0013, 284.7539],
                ],
                [
                    [276.8062, 279.4716, 276.9695],
                    [283.9340, 286.9906, 285.2114],
                    [293.5306, 293.4192, 287.1828],
                ],
                [
                    [277.9719, 286.1142, 281.3962],
                    [281.4469, 285.6135, 283.70135],
                    [290.8871, 286.5834, 289.61172],
                ],
            ],
            dtype=np.float32,
        )
        expected_name = "air_temperature"
        expected_dims = ["realization", "latitude", "longitude"]

    result = ApplySAMOS(percentiles=None).process(
        forecast=forecast_slice,
        forecast_gams=forecast_gams,
        truth_gams=forecast_gams,
        gam_features=features,
        emos_coefficients=emos_coefficients,
        gam_additional_fields=None,
        emos_additional_fields=CubeList([]),
        **process_kwargs,
    )

    np.testing.assert_array_almost_equal(result.data, expected_data, decimal=4)
    assert result.name() == expected_name
    assert [c.name() for c in result.coords(dim_coords=True)] == expected_dims

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
                [313.17203, 301.30695, 287.89954],
                [280.74792, 288.16888, 296.38806],
                [333.15, 333.15, 304.0256],
            ],
            [
                [313.1628, 301.11777, 288.257],
                [280.64725, 288.43387, 295.84628],
                [333.15, 333.15, 302.2623],
            ],
            [
                [313.16742, 300.9286, 288.61444],
                [280.8486, 288.3014, 296.1172],
                [333.15, 333.15, 303.14395],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_notemosalt_gamalt": np.array(
        [
            [
                [263.1705, 275.77353, 287.91034],
                [280.7542, 288.18423, 296.4416],
                [278.2182, 290.7106, 304.02924],
            ],
            [
                [263.1613, 275.58432, 288.2678],
                [280.65353, 288.44922, 295.8998],
                [278.01764, 290.00946, 302.26596],
            ],
            [
                [263.1659, 275.39514, 288.62524],
                [280.8549, 288.3167, 296.17072],
                [278.41873, 291.41177, 303.1476],
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
    "input_format,output_format,emos_include_altitude,gam_include_altitude,constant_extrapolation",
    [
        ["realization", "realization", False, False, False],
        ["realization", "realization", True, False, False],
        ["realization", "realization", False, True, False],
        ["realization", "realization", False, True, True],
        ["realization", "realization", True, True, False],
        ["realization", "probability", False, False, False],
        ["realization", "probability", True, False, False],
        ["percentile", "percentile", False, False, False],
        ["percentile", "probability", False, False, False],
        ["probability", "probability", False, False, False],
    ],
)
def test_process(
    input_format,
    output_format,
    emos_include_altitude,
    gam_include_altitude,
    constant_extrapolation,
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

    if gam_include_altitude:
        features.append("surface_altitude")
        model_specification.append(["spline", [features.index("surface_altitude")], {}])

    forecast_cube, additional_cubes = create_cubes_for_gam_fitting(
        n_spatial_points=n_spatial_points,
        n_realizations=n_realizations,
        n_times=n_times,
        include_altitude=emos_include_altitude or gam_include_altitude,
        fixed_forecast_period=True,
    )
    gam_additional_cubes = additional_cubes.copy() if gam_include_altitude else None
    emos_additional_cubes = additional_cubes.copy() if emos_include_altitude else None

    for key, value in FORECAST_ATTRIBUTES.items():
        forecast_cube.attributes[key] = value

    forecast_gams = TrainGAMsForSAMOS(model_specification).process(
        forecast_cube,
        features,
        additional_fields=gam_additional_cubes,
    )

    forecast_slice = next(forecast_cube.slices_over(["forecast_reference_time"]))

    if constant_extrapolation:
        # Modify latitude and longitude coordinates so that the first and last point
        # are outside the bounds of those variables in the training data. Constant
        # extrapolation should ensure that the expected result is unchanged.
        for coord_name in ["latitude", "longitude"]:
            forecast_slice.coord(coord_name).points = np.array(
                [-20.0, 0.0, 20.0], dtype=np.float32
            )
            forecast_slice.coord(coord_name).bounds = np.array(
                [[-30.0, -10.0], [-10.0, 10.0], [10.0, 30.0]], dtype=np.float32
            )
            gam_additional_cubes[0].coord(coord_name).points = np.array(
                [-20.0, 0.0, 20.0], dtype=np.float32
            )
            gam_additional_cubes[0].coord(coord_name).bounds = np.array(
                [[-30.0, -10.0], [-10.0, 10.0], [10.0, 30.0]],
                dtype=np.float32,
            )

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
        gam_additional_fields=gam_additional_cubes,
        emos_additional_fields=emos_additional_cubes,
        **process_kwargs,
    )

    np.testing.assert_array_almost_equal(result.data, expected_data, decimal=4)
    assert result.name() == expected_name
    assert [c.name() for c in result.coords(dim_coords=True)] == expected_dims

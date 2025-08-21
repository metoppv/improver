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
                [266.6658, 299.4605, 287.6780],
                [279.8986, 288.0487, 294.7888],
                [387.0688, 366.3009, 303.3625],
            ],
            [
                [266.6648, 299.5598, 287.4164],
                [279.962, 287.9135, 296.2049],
                [386.4832, 366.5633, 302.5225],
            ],
            [
                [266.6653, 299.5101, 287.5472],
                [280.0255, 288.1839, 295.4969],
                [386.7760, 366.4321, 302.9425],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_notemosalt_notgamalt": np.array(
        [
            [
                [273.1741, 280.7437, 287.6045],
                [280.5615, 288.4035, 294.7645],
                [288.7262, 295.0875, 303.3314],
            ],
            [
                [273.1731, 280.8429, 287.3429],
                [280.6249, 288.2683, 296.1807],
                [288.1406, 295.3499, 302.4914],
            ],
            [
                [273.1736, 280.7933, 287.4737],
                [280.6884, 288.5387, 295.4726],
                [288.4334, 295.2187, 302.9114],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_emosalt_gamalt": np.array(
        [
            [
                [266.6658, 299.4605, 287.6780],
                [279.8986, 288.0487, 294.7888],
                [387.0688, 366.3009, 303.3625],
            ],
            [
                [266.6648, 299.5598, 287.4164],
                [279.962, 287.9135, 296.2049],
                [386.4832, 366.5633, 302.5225],
            ],
            [
                [266.6653, 299.5101, 287.5472],
                [280.0255, 288.1839, 295.4969],
                [386.7760, 366.4321, 302.9425],
            ],
        ],
        dtype=np.float32,
    ),
    "real_real_notemosalt_gamalt": np.array(
        [
            [
                [273.1742, 280.7437, 287.6045],
                [280.5615, 288.4035, 294.7645],
                [288.7262, 295.0875, 303.3314],
            ],
            [
                [273.1730, 280.8429, 287.3429],
                [280.6249, 288.2683, 296.1807],
                [288.1406, 295.3499, 302.4914],
            ],
            [
                [273.1736, 280.7933, 287.4737],
                [280.6884, 288.5387, 295.4726],
                [288.4334, 295.2187, 302.9114],
            ],
        ],
        dtype=np.float32,
    ),
    "real_prob_emosalt_notgamalt": np.array(
        [
            [
                [0.0000, 1.0000, 0.0098],
                [0.0000, 0.5961, 1.0000],
                [1.0000, 1.0000, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [1.0000, 1.0000, 0.4632],
            ],
        ],
        dtype=np.float32,
    ),
    "prob_prob_notemosalt_notgamalt": np.array(
        [
            [
                [1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000],
                [1.0000, 0.9999, 0.9979],
            ],
            [
                [0.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.9996],
                [1.0000, 0.9992, 0.9928],
            ],
            [
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.9999, 0.9973],
                [0.9999, 0.9958, 0.9794],
            ],
        ],
        dtype=np.float32,
    ),
    "perc_perc_notemosalt_notgamalt": np.array(
        [
            [
                [273.1731, 280.7437, 287.3429],
                [280.5615, 288.2683, 294.7645],
                [288.1406, 295.0875, 302.4914],
            ],
            [
                [273.1736, 280.7933, 287.4737],
                [280.6249, 288.4035, 295.4726],
                [288.4334, 295.2187, 302.9114],
            ],
            [
                [273.1741, 280.8429, 287.6045],
                [280.6884, 288.5387, 296.1807],
                [288.7262, 295.3499, 303.3314],
            ],
        ],
        dtype=np.float32,
    ),
    "perc_prob_notemosalt_notgamalt": np.array(
        [
            [
                [0.0000, 0.0000, 0.0033],
                [0.0000, 0.9780, 1.0000],
                [0.8410, 1.0000, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.4434],
            ],
        ],
        dtype=np.float32,
    ),
    "real_prob_notemosalt_notgamalt": np.array(
        [
            [
                [0.0000, 0.0000, 0.0033],
                [0.0000, 0.9780, 1.0000],
                [0.8410, 1.0000, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.4434],
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
        gam_features=features,
        emos_coefficients=emos_coefficients,
        gam_additional_fields=None,
        emos_additional_fields=additional_cubes,
        **process_kwargs,
    )

    print(result)
    print(result.data)

    np.testing.assert_array_almost_equal(result.data, expected_data, decimal=4)
    assert result.name() == expected_name
    assert [c.name() for c in result.coords(dim_coords=True)] == expected_dims

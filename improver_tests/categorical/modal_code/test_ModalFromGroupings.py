# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for ModalFromGroupings class."""

from calendar import timegm
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pytest

from improver.categorical.modal_code import ModalFromGroupings
from improver_tests.categorical.decision_tree import wxcode_decision_tree
from improver_tests.categorical.modal_code.test_ModalCategory import (
    wxcode_series_fixture,
)

MODEL_ID_ATTR = "mosg__model_configuration"
RECORD_RUN_ATTR = "mosg__model_run"
TARGET_TIME = dt(2020, 6, 15, 18)
BROAD_CATEGORIES = {
    "wet": np.arange(9, 31),
    "dry": np.arange(0, 9),
}
# Priority ordered categories (keys) in case of ties
WET_CATEGORIES = {
    "extreme_convection": [30, 29, 28, 21, 20, 19],
    "frozen": [27, 26, 25, 24, 23, 22, 18, 17, 16],
    "liquid": [15, 14, 13, 12, 11, 10, 9],
}
INTENSITY_CATEGORIES = {
    "rain_shower": [10, 14],
    "rain": [12, 15],
    "snow_shower": [23, 26],
    "snow": [24, 27],
    "thunder": [29, 30],
}


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, expected",
    (
        # Sunny day (1), one rain code (15) that is in the minority, expect sun
        # code (1).
        ([1, 1, 1, 15], 1),
        # # Short period with an equal split. The most significant weather
        # # (hail, 21) should be returned.
        ([1, 21], 21),
        # # A single time is provided in which a sleet shower is forecast (16).
        # # We expect the cube to be returned with the night code changed to a
        # # day code (17).
        ([16], 17),
        # # Equal split in day codes, but a night code corresponding to one
        # # of the day types means a valid mode can be calculated. We expect the
        # # day code (10) to be returned.
        ([1, 1, 10, 10, 9], 10),
        # # No clear representative code. Groups by wet and dry codes and selects
        # # the most significant dry code (8).
        ([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15], 8),
        # No clear representative code. More dry symbols are present, so the most
        # significant dry code is selected (8).
        ([1, 3, 4, 5, 6, 7, 8, 16, 11, 12, 14, 15], 8),
        # No clear representative code. More dry symbols are present, so the most
        # significant dry code is selected (8).
        ([5, 5, 5, 5, 6, 6, 6, 6, 8, 8, 8, 8, 7, 7, 7, 7], 8),
        # An extreme edge case in which all the codes across time for a site
        # are different. More dry symbols are present, so the most
        # significant dry code is selected (8).
        ([1, 3, 4, 5, 7, 8, 10, 17, 20, 23], 8),
        # Equal numbers of dry and wet symbols leads to a wet symbol being chosen.
        # Code 23 and 17 are both frozen precipitation, so are grouped together,
        # and the most significant of these is chosen based on the order of the codes
        # within the frozen precipitation categorisation.
        ([1, 3, 4, 5, 10, 17, 20, 23], 23),
        # All dry symbols. The most significant dry symbol is chosen, asssuming that
        # higher
        ([1, 3, 4, 5], 5),
        # All wet codes. Two frozen precipitation and two "extreme". Codes from the
        # extreme category chosen in preference.
        ([29, 29, 26, 26], 29),
        # All wet codes. Two frozen preciptiation and two liquid precipitation.
        # Frozen precipitation chosen in preference.
        ([10, 10, 26, 26], 26),
    ),
)
def test_expected_values(wxcode_series, expected):
    """Test that the expected period representative symbol is returned."""
    _, _, _, _, wxcode_cubes = wxcode_series
    result = ModalFromGroupings(
        wxcode_decision_tree(), BROAD_CATEGORIES, WET_CATEGORIES, INTENSITY_CATEGORIES
    )(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, wet_bias, expected",
    (
        # More dry codes (6) than wet codes (4),the most significant dry symbol
        # is selected.
        ([1, 3, 4, 5, 7, 8, 10, 10, 10, 10], 1, 8),
        # With a wet bias of 2, there are more wet codes than dry codes, so the modal
        # wet code is selected.
        ([1, 3, 4, 5, 7, 8, 10, 10, 10, 10], 2, 10),
        # More dry codes (7) than wet codes (3),the most significant dry symbol
        # is selected.
        ([1, 3, 4, 5, 7, 8, 8, 10, 10, 10], 1, 8),
        # A wet bias of 2 doubles the number of wet codes to 6, however, this is
        # still fewer than the number of dry codes.
        ([1, 3, 4, 5, 7, 8, 8, 10, 10, 10], 2, 8),
        # A wet bias of 3 triples the number of wet codes to 9, so the modal wet symbol
        # is selected.
        ([1, 3, 4, 5, 7, 8, 8, 10, 10, 10], 3, 10),
        #
    ),
)
def test_expected_values_wet_bias(wxcode_series, wet_bias, expected):
    """Test that the expected period representative symbol is returned."""
    _, _, _, _, wxcode_cubes = wxcode_series
    result = ModalFromGroupings(
        wxcode_decision_tree(),
        BROAD_CATEGORIES,
        WET_CATEGORIES,
        INTENSITY_CATEGORIES,
        wet_bias=wet_bias,
    )(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, day_weighting, day_start, day_end, expected",
    (
        # First time is valid at 18Z. Subsequent codes are backwards in time from 18Z.
        # There are more light shower codes, so this is the modal code.
        ([10, 10, 10, 10, 1, 1, 1], 1, 11, 15, 10),
        # A day weighting of 2 results in the number of clear day codes doubling,
        # and one more shower symbol giving 6 dry codes, and 5 wet codes.
        ([10, 10, 10, 10, 1, 1, 1], 2, 11, 15, 1),
        # Altering the day_end to 16Z results in 6 dry codes in total and 6 wet codes,
        # so the resulting code is wet.
        ([10, 10, 10, 10, 1, 1, 1], 2, 11, 16, 10),
        # Increasing the day weighting to 3 results in 9 dry codes and 8 wet codes, so
        # the resulting code is dry.
        ([10, 10, 10, 10, 1, 1, 1], 3, 11, 16, 1),
    ),
)
def test_expected_values_day_weighting(
    wxcode_series, day_weighting, day_start, day_end, expected
):
    """Test that the expected period representative symbol is returned."""
    _, _, _, _, wxcode_cubes = wxcode_series
    result = ModalFromGroupings(
        wxcode_decision_tree(),
        BROAD_CATEGORIES,
        WET_CATEGORIES,
        INTENSITY_CATEGORIES,
        day_weighting=day_weighting,
        day_start=day_start,
        day_end=day_end,
    )(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, ignore_intensity, expected",
    (
        # All precipitation is frozen. Sleet shower is the modal code.
        ([23, 23, 23, 26, 17, 17, 17, 17], False, 17),
        # When snow shower codes are grouped, snow shower is chosen.
        ([23, 23, 23, 26, 17, 17, 17, 17], True, 23),
    ),
)
def test_expected_values_ignore_intensity(wxcode_series, ignore_intensity, expected):
    """Test that the expected period representative symbol is returned."""
    _, _, _, _, wxcode_cubes = wxcode_series
    result = ModalFromGroupings(
        wxcode_decision_tree(),
        BROAD_CATEGORIES,
        WET_CATEGORIES,
        INTENSITY_CATEGORIES,
        ignore_intensity=ignore_intensity,
    )(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, wet_bias, day_weighting, day_start, day_end, ignore_intensity, expected",
    (
        # The sleet code is the most common, so this is the modal code.
        ([23, 23, 23, 26, 17, 17, 17, 17], 1, 1, 11, 15, False, 17),
        # A day weighting of 10 at 15Z emphasises the heavy snow shower, and this code
        # is therefore the modal code.
        ([23, 23, 23, 26, 17, 17, 17, 17], 1, 10, 15, 15, False, 26),
        # Without any weighting, there would be a dry symbol. A day weighting of 2
        # results in 6 dry codes and 5 wet codes. A wet bias results in 6 dry codes
        # and 10 wet codes.
        ([1, 1, 1, 1, 1, 17, 17, 17], 2, 2, 12, 14, False, 17),
        # All precipitation is frozen. Ignoring the intensities means that a
        # day weighting of 2 results in 8 sleet codes and 8 light snow shower codes.
        # A wet bias results in 16 sleet codes and 16 light snow shower codes.
        # The snow code is chosen as it is the most significant frozen precipitation,
        # and ignoring intensity option ensures that the modal code is set to the most
        # common snow shower code.
        ([23, 23, 23, 26, 17, 17, 17, 17], 2, 2, 11, 18, True, 23),
    ),
)
def test_expected_values_interactions(
    wxcode_series,
    wet_bias,
    day_weighting,
    day_start,
    day_end,
    ignore_intensity,
    expected,
):
    """Test that the expected period representative symbol is returned."""
    _, _, _, _, wxcode_cubes = wxcode_series
    result = ModalFromGroupings(
        wxcode_decision_tree(),
        BROAD_CATEGORIES,
        WET_CATEGORIES,
        INTENSITY_CATEGORIES,
        wet_bias=wet_bias,
        day_weighting=day_weighting,
        day_start=day_start,
        day_end=day_end,
        ignore_intensity=ignore_intensity,
    )(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("record_run_attr", [False, True])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1, 3])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize("data", [np.ones(12), np.ones(1)])
def test_metadata(wxcode_series):
    """Check that the returned metadata is correct. In this case we expect a
    time coordinate with bounds that describe the full period over which the
    representative symbol has been calculated while the forecast_reference_time
    will be the latest of those input and the forecast_period will be the
    difference between the forecast_reference_time and time.

    A single data point is tested which means a single cube is passed in. This
    ensures the metadata is consistent whether or not the input data has passed
    through the modal aggregator."""

    def as_utc_timestamp(time):
        return timegm(time.utctimetuple())

    (
        interval,
        model_id_attr,
        record_run_attr,
        offset_reference_times,
        wxcode_cubes,
    ) = wxcode_series

    kwargs = {}
    if model_id_attr:
        kwargs.update({"model_id_attr": MODEL_ID_ATTR})
    if record_run_attr:
        kwargs.update({"record_run_attr": RECORD_RUN_ATTR})

    result = ModalFromGroupings(
        wxcode_decision_tree(),
        BROAD_CATEGORIES,
        WET_CATEGORIES,
        INTENSITY_CATEGORIES,
        **kwargs,
    )(wxcode_cubes)

    n_times = len(wxcode_cubes)
    expected_time = TARGET_TIME
    expected_bounds = [TARGET_TIME - timedelta(hours=n_times * interval), TARGET_TIME]
    expected_reference_time = TARGET_TIME - timedelta(hours=42)
    expected_forecast_period = (expected_time - expected_reference_time).total_seconds()
    expected_forecast_period_bounds = [
        expected_forecast_period - n_times * interval * 3600,
        expected_forecast_period,
    ]
    expected_model_id_attr = "uk_det uk_ens"
    expected_record_det = "uk_det:20200613T2300Z:\n"
    expected_record_ens = "uk_ens:20200613T{}00Z:"

    # Expected record_run attribute contains all contributing cycle times.
    if offset_reference_times and len(wxcode_cubes) > 1:
        expected_record_run_attr = expected_record_det + "\n".join(
            [expected_record_ens.format(value) for value in range(10, 22)]
        )
    else:
        expected_record_run_attr = expected_record_det + expected_record_ens.format(21)

    assert result.coord("time").points[0] == as_utc_timestamp(expected_time)
    assert result.coord("time").bounds[0][0] == as_utc_timestamp(expected_bounds[0])
    assert result.coord("time").bounds[0][1] == as_utc_timestamp(expected_bounds[1])
    assert result.coord("forecast_reference_time").points[0] == as_utc_timestamp(
        expected_reference_time
    )
    assert not result.coord("forecast_reference_time").has_bounds()
    assert result.coord("forecast_period").points[0] == expected_forecast_period
    assert np.allclose(
        result.coord("forecast_period").bounds[0], expected_forecast_period_bounds
    )
    assert result.cell_methods[0].method == "mode"
    assert result.cell_methods[0].coord_names[0] == "time"
    assert result.cell_methods[0].intervals[0] == f"{interval} hour"
    if model_id_attr:
        assert result.attributes[MODEL_ID_ATTR] == expected_model_id_attr
    else:
        assert MODEL_ID_ATTR not in result.attributes.keys()
    if record_run_attr and model_id_attr:
        assert RECORD_RUN_ATTR in result.attributes.keys()
        assert result.attributes[RECORD_RUN_ATTR] == expected_record_run_attr
    else:
        assert RECORD_RUN_ATTR not in result.attributes.keys()


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False])
@pytest.mark.parametrize("interval", [1, 3])
@pytest.mark.parametrize("offset_reference_times", [False])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize("data", [[1, 1, 1, 15]])
def test_unmatching_bounds_exception(wxcode_series):
    """Test that an exception is raised if inputs do not represent the same
    intervals."""
    _, _, _, _, wxcode_cubes = wxcode_series
    bounds = wxcode_cubes[0].coord("time").bounds.copy()
    bounds[0][0] += 1800
    wxcode_cubes[0].coord("time").bounds = bounds
    with pytest.raises(
        ValueError, match="Input diagnostics do not have consistent periods."
    ):
        ModalFromGroupings(
            wxcode_decision_tree(),
            BROAD_CATEGORIES,
            WET_CATEGORIES,
            INTENSITY_CATEGORIES,
        )(wxcode_cubes)

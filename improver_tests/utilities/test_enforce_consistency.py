# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Unit tests for enforce_consistency utilities."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.enforce_consistency import EnforceConsistentForecasts

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "Unit Test Data",
    "source": "Unit Test",
    "institution": "Met Office",
}


def get_percentiles(value, shape):
    """
    Correctly shape the value provided into the desired shape. The value is used as the
    middle of three percentiles. The value is reduced or increased by 10 to create the
    bounding percentiles.
    """
    return np.broadcast_to(
        np.expand_dims([value - 10, value, value + 10], axis=(1, 2),), shape,
    )


def get_percentile_forecast(value, shape, name):
    """
    Create a percentile forecast cube.
    """
    data = get_percentiles(value, shape)
    forecast_cube = set_up_percentile_cube(
        data, percentiles=[10, 50, 90], name=name, units="m s-1",
    )
    return forecast_cube


def get_probability_forecast(value, shape, name) -> Cube:
    """
    Create a probability forecast cube.
    """
    data = np.full(shape, fill_value=value, dtype=np.float32)
    forecast_cube = set_up_probability_cube(
        data,
        variable_name=name,
        thresholds=[0.3125, 0.5, 0.6875],
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return forecast_cube


def get_realization_forecast(value, shape, name) -> Cube:
    """
    Create a realization forecast cube.
    """
    data = np.full(shape, fill_value=value, dtype=np.float32)
    forecast_cube = set_up_variable_cube(
        data,
        name=name,
        units="Celsius",
        realizations=[0, 1, 2],
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return forecast_cube


def get_expected(forecast_data, bound_data, comparison_operator):
    """
    Calculate the expected result via a different method to that used within the plugin.
    """
    diff = bound_data - forecast_data
    if comparison_operator == ">=":
        diff = np.clip(diff, 0, None)
    else:
        diff = np.clip(diff, None, 0)
    expected = forecast_data + diff
    return expected


@pytest.mark.parametrize(
    "forecast_type, additive_amount, multiplicative_amount, reference_value, "
    "forecast_value, comparison_operator",
    (
        ("probability", None, None, 0.5, 0.4, ">="),
        ("probability", None, None, 0.4, 0.5, "<="),
        ("probability", None, None, 0.4, 0.5, ">="),  # no change required
        ("percentile", 0, 1.1, 50, 40, ">="),
        ("percentile", 0, 0.9, 40, 50, "<="),
        ("percentile", 10, 1, 50, 40, ">="),
        ("percentile", -10, 0.8, 50, 40, ">="),  # no change required
        ("realization", 0, 1.1, 20, 25, "<="),
        ("realization", 5, 1.2, 20, 15, ">="),
        ("realization", -5, 0.75, 20, 5, "<="),  # no change required
    ),
)
def test_basic(
    forecast_type,
    additive_amount,
    multiplicative_amount,
    reference_value,
    forecast_value,
    comparison_operator,
):
    """
    Test that consistency between forecasts is enforced correctly for a variety of
    percentile or probability forecasts.
    """
    shape = (3, 2, 2)

    if forecast_type == "probability":
        reference_cube_name = "cloud_area_fraction"
        forecast_cube_name = "low_and_medium_type_cloud_area_fraction"
        get_forecast = get_probability_forecast
        reference_value = reference_value / 100
        forecast_value = forecast_value / 100
    elif forecast_type == "reference":
        reference_cube_name = "surface_temperature"
        forecast_cube_name = "feels_like_temperature"
        get_forecast = get_realization_forecast
    else:
        reference_cube_name = "wind_speed_at_10m"
        forecast_cube_name = "wind_gust_at_10m_max-PT01H"
        get_forecast = get_percentile_forecast

    reference_cube = get_forecast(reference_value, shape, reference_cube_name)
    forecast_cube = get_forecast(forecast_value, shape, forecast_cube_name)

    if forecast_type == "probability":
        forecast_cube.data[:, 1, 1] = 0.6

    result = EnforceConsistentForecasts(
        additive_amount=additive_amount,
        multiplicative_amount=multiplicative_amount,
        comparison_operator=comparison_operator,
    )(forecast_cube, reference_cube)

    if additive_amount is None:
        additive_amount = 0
    if multiplicative_amount is None:
        multiplicative_amount = 1
    bound_data = additive_amount + (multiplicative_amount * reference_cube.copy().data)
    expected = get_expected(forecast_cube.data, bound_data, comparison_operator)

    assert isinstance(result, Cube)
    assert result.name() == forecast_cube.name()
    assert np.shape(result) == shape
    np.testing.assert_array_almost_equal(result.data, expected)


@pytest.mark.parametrize(
    "forecast_type, forecast_value, reference_value, comparison_operator, "
    "diff_for_warning",
    (
        ("percentile", 10, 50, ">=", 30),  # change too big
        ("percentile", 20, 30, "=", 30),  # bad comparison operator
        (
            "probability",
            0.4,
            0.6,
            ">=",
            0.5,
        ),  # cannot specify additive and multiplicative amounts with probabilities
        ("realization", 15, 293.15, ">=", 30),  # mismatching units
    ),
)
def test_exceptions(
    forecast_type,
    forecast_value,
    reference_value,
    comparison_operator,
    diff_for_warning,
):
    """
    Test that a warning is raised if the plugin changes the forecast by more than
    diff_for_warning.
    """
    shape = (3, 2, 2)

    if forecast_type == "probability":
        reference_cube_name = "cloud_area_fraction"
        forecast_cube_name = "low_and_medium_type_cloud_area_fraction"
        get_forecast = get_probability_forecast
        additive_amount = 0.1
        multiplicative_amount = 1.1
    elif forecast_type == "realization":
        reference_cube_name = "surface_temperature"
        forecast_cube_name = "feels_like_temperature"
        get_forecast = get_realization_forecast
    else:
        reference_cube_name = "wind_speed_at_10m"
        forecast_cube_name = "wind_gust_at_10m_max-PT01H"
        get_forecast = get_percentile_forecast

    reference_cube = get_forecast(reference_value, shape, reference_cube_name)
    forecast_cube = get_forecast(forecast_value, shape, forecast_cube_name)

    if comparison_operator == "=":
        with pytest.raises(ValueError, match="Comparison_operator must be either"):
            EnforceConsistentForecasts(comparison_operator=comparison_operator)(
                forecast_cube, reference_cube
            )
    elif forecast_type == "probability":
        with pytest.raises(ValueError, match="For probability data"):
            EnforceConsistentForecasts(
                additive_amount=additive_amount,
                multiplicative_amount=multiplicative_amount,
                comparison_operator=comparison_operator,
            )(forecast_cube, reference_cube)
    elif forecast_type == "realization":
        reference_cube.units = "K"
        with pytest.raises(ValueError, match="The units in the forecast"):
            EnforceConsistentForecasts(comparison_operator=comparison_operator)(
                forecast_cube, reference_cube
            )
    else:
        with pytest.warns(UserWarning, match="Inconsistency between forecast"):
            EnforceConsistentForecasts(diff_for_warning=diff_for_warning)(
                forecast_cube, reference_cube
            )

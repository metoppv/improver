# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for enforce_consistency utilities."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.forecast_times import _create_frt_type_coord
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.forecast_reference_enforcement import EnforceConsistentForecasts

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
        np.expand_dims([value - 10, value, value + 10], axis=(1, 2)), shape
    )


def get_percentile_forecast(value, shape, name):
    """
    Create a percentile forecast cube.
    """
    data = get_percentiles(value, shape)
    forecast_cube = set_up_percentile_cube(
        data, percentiles=[10, 50, 90], name=name, units="m s-1"
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
    diff_list = []
    for index, bound in enumerate(bound_data):
        diff = bound - forecast_data
        if comparison_operator[index] == ">=":
            diff = np.clip(diff, 0, None)
        else:
            diff = np.clip(diff, None, 0)
        diff_list.append(diff)

    diff = sum(diff_list)
    expected = forecast_data + diff
    return expected


@pytest.mark.parametrize(
    "forecast_type, additive_amount, multiplicative_amount, reference_value, "
    "forecast_value, comparison_operator",
    (
        ("probability", 0, 1, 0.5, 0.4, ">="),
        ("probability", 0, 1, 0.4, 0.5, "<="),
        ("probability", 0, 1, 0.4, 0.5, ">="),  # no change required
        ("percentile", 0, 1.1, 50, 40, ">="),
        ("percentile", 0, 0.9, 40, 50, "<="),
        ("percentile", 10, 1, 50, 40, ">="),
        ("percentile", -10, 0.8, 50, 40, ">="),  # no change required
        ("realization", 0, 1.1, 20, 25, "<="),
        ("realization", 5, 1.2, 20, 15, ">="),
        ("realization", -5, 0.75, 20, 5, "<="),  # no change required
    ),
)
def test_single_bound(
    forecast_type,
    additive_amount,
    multiplicative_amount,
    reference_value,
    forecast_value,
    comparison_operator,
):
    """
    Test that consistency between forecasts is enforced correctly for a variety of
    percentile, probability, or realization forecasts when a single bound is provided.
    """
    shape = (3, 2, 2)

    if forecast_type == "probability":
        reference_cube_name = "cloud_area_fraction"
        forecast_cube_name = "low_and_medium_type_cloud_area_fraction"
        get_forecast = get_probability_forecast
        reference_value = reference_value / 100
        forecast_value = forecast_value / 100
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
    expected = get_expected(forecast_cube.data, [bound_data], [comparison_operator])

    assert isinstance(result, Cube)
    assert result.name() == forecast_cube.name()
    assert np.shape(result) == shape
    np.testing.assert_array_almost_equal(result.data, expected)


@pytest.mark.parametrize(
    "forecast_type, additive_amount, multiplicative_amount, reference_value, "
    "forecast_value, comparison_operator",
    (
        ("percentile", [0, 0], [1.1, 2], 50, 40, [">=", "<="]),
        ("percentile", [0, 0], [0.9, 0.1], 40, 50, ["<=", ">="]),
        ("realization", [0, 0], [1.1, 1.5], 20, 25, [">=", "<="]),
        ("realization", [5, 5], [1.2, 0.5], 20, 15, ["<=", ">="]),
    ),
)
def test_double_bound(
    forecast_type,
    additive_amount,
    multiplicative_amount,
    reference_value,
    forecast_value,
    comparison_operator,
):
    """
    Test that consistency between forecasts is enforced correctly for a variety of
    percentile or realization forecasts when two bounds are provided.
    """
    shape = (3, 2, 2)

    if forecast_type == "realization":
        reference_cube_name = "surface_temperature"
        forecast_cube_name = "feels_like_temperature"
        get_forecast = get_realization_forecast
    else:
        reference_cube_name = "wind_speed_at_10m"
        forecast_cube_name = "wind_gust_at_10m_max-PT01H"
        get_forecast = get_percentile_forecast

    reference_cube = get_forecast(reference_value, shape, reference_cube_name)
    forecast_cube = get_forecast(forecast_value, shape, forecast_cube_name)

    result = EnforceConsistentForecasts(
        additive_amount=additive_amount,
        multiplicative_amount=multiplicative_amount,
        comparison_operator=comparison_operator,
    )(forecast_cube, reference_cube)

    bound_data = []
    for i in range(2):
        bound_data.append(
            additive_amount[i] + (multiplicative_amount[i] * reference_cube.copy().data)
        )

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
def test_single_bound_exceptions(
    forecast_type,
    forecast_value,
    reference_value,
    comparison_operator,
    diff_for_warning,
):
    """
    Test that correct errors and warnings are raised when using one bound.
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
        with pytest.raises(ValueError, match="When enforcing consistency with one"):
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
        reference_cube.units = "m/s"
        with pytest.raises(ValueError, match="The units in the forecast"):
            EnforceConsistentForecasts(comparison_operator=comparison_operator)(
                forecast_cube, reference_cube
            )
    else:
        with pytest.warns(UserWarning, match="Inconsistency between forecast"):
            EnforceConsistentForecasts(diff_for_warning=diff_for_warning)(
                forecast_cube, reference_cube
            )


@pytest.mark.parametrize(
    "forecast_type, additive_amount, multiplicative_amount, comparison_operator, msg",
    (
        (
            "probability",
            [0, 0],
            [1.1, 2],
            [">=", "<="],
            "For probability data",
        ),  # cannot specify additive and multiplicative amounts with probabilities
        (
            "percentile",
            [0, 0],
            [1.1, 2],
            [">=", ">="],
            "When comparison operators are provided as a list",
        ),  # bad comparison operator
        (
            "percentile",
            0,
            [1.1, 2],
            [">=", "="],
            "If any of additive_amount",
        ),  # not all inputs are lists
        (
            "percentile",
            [0, 0],
            [1.1, 2],
            ["<=", ">="],
            "The provided reference_cube",
        ),  # contradictory bounds
    ),
)
def test_double_bounds_exceptions(
    forecast_type, additive_amount, multiplicative_amount, comparison_operator, msg
):
    """
    Test that correct errors are raised when using two bounds.
    """
    shape = (3, 2, 2)
    forecast_value = 50
    reference_value = 50

    if forecast_type == "probability":
        reference_cube_name = "cloud_area_fraction"
        forecast_cube_name = "low_and_medium_type_cloud_area_fraction"
        get_forecast = get_probability_forecast
        forecast_value = forecast_value / 100
        reference_value = reference_value / 100
    else:
        reference_cube_name = "wind_speed_at_10m"
        forecast_cube_name = "wind_gust_at_10m_max-PT01H"
        get_forecast = get_percentile_forecast

    reference_cube = get_forecast(reference_value, shape, reference_cube_name)
    forecast_cube = get_forecast(forecast_value, shape, forecast_cube_name)

    with pytest.raises(ValueError, match=msg):
        EnforceConsistentForecasts(
            additive_amount=additive_amount,
            multiplicative_amount=multiplicative_amount,
            comparison_operator=comparison_operator,
        )(forecast_cube, reference_cube)


@pytest.mark.parametrize("blend_time", [False, True])
@pytest.mark.parametrize(
    "ref_shift,forecast_shift,expected_shift",
    (
        [3600, 0, 3600],
        [-3600, 0, 0],
    ),
)
def test_updating_times(blend_time, ref_shift, forecast_shift, expected_shift):
    """
    Test that forecast_reference_time and / or blend_time are updated on
    cubes to which consitency is being applied.
    """
    shape = (3, 2, 2)
    reference_cube = get_realization_forecast(275, shape, "air_temperature")
    forecast_cube = get_realization_forecast(274, shape, "air_temperature")

    expected_frt = (
        forecast_cube.coord("forecast_reference_time").points[0] + expected_shift
    )
    reference_cube.coord("forecast_reference_time").points = [1510272000 + ref_shift]
    forecast_cube.coord("forecast_reference_time").points = [
        1510272000 + forecast_shift
    ]

    if blend_time:
        for cube in [reference_cube, forecast_cube]:
            blend_coord = _create_frt_type_coord(
                cube,
                cube.coord("forecast_reference_time").cell(0).point,
                name="blend_time",
            )
            cube.add_aux_coord(blend_coord, data_dims=None)

    result = EnforceConsistentForecasts(
        comparison_operator=">=", use_latest_update_time=True
    )(forecast_cube, reference_cube)

    assert result.coord("forecast_reference_time").points[0] == expected_frt
    if blend_time:
        assert result.coord("blend_time").points[0] == expected_frt


@pytest.mark.parametrize(
    "ref_coords",
    (
        ["forecast_reference_time"],
        ["blend_time"],
        ["forecast_reference_time", "blend_time"],
    ),
)
@pytest.mark.parametrize(
    "forecast_coords",
    (
        ["forecast_reference_time"],
        ["blend_time"],
        ["forecast_reference_time", "blend_time"],
    ),
)
def test_mismatched_coord_exception(ref_coords, forecast_coords):
    """Test an exception is raised if the forecast and reference cubes
    don't both have the same cycle related coordinates. This means that
    one might have both a blend_time and forecast_reference_time, and one
    might have only forecast_reference_time."""

    if set(ref_coords) == set(forecast_coords):
        pytest.skip()

    shape = (3, 2, 2)
    reference_cube = get_realization_forecast(275, shape, "air_temperature")
    forecast_cube = get_realization_forecast(274, shape, "air_temperature")

    for cube, crds in zip(
        [reference_cube, forecast_cube], [ref_coords, forecast_coords]
    ):
        crd = cube.coord("forecast_reference_time").copy()
        cube.remove_coord("forecast_reference_time")
        for crd_name in crds:
            new_crd = crd.copy()
            new_crd.rename(crd_name)
            cube.add_aux_coord(new_crd, data_dims=None)

    with pytest.raises(ValueError, match="Cubes do not include "):
        EnforceConsistentForecasts(
            comparison_operator=">=", use_latest_update_time=True
        )(forecast_cube, reference_cube)

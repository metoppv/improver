# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the LapseRate plugin."""

import cf_units
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from improver.constants import DALR
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.temperature.lapse_rate import LapseRate
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


def test__repr__():
    """Test that the __repr__ returns the expected string."""
    result = str(LapseRate())
    msg = (
        "<LapseRate: max_height_diff: 35, nbhood_radius: 7,"
        "max_lapse_rate: 0.0294, min_lapse_rate: -0.0098>"
    )
    assert result == msg

    """Test the _calc_lapse_rate function."""


@pytest.fixture()
def temperature():
    """Sets up arrays."""
    temperature = np.array(
        [
            [280.06, 279.97, 279.90],
            [280.15, 280.03, 279.96],
            [280.25, 280.33, 280.27],
        ]
    )
    return temperature


@pytest.fixture()
def orography():
    orography = np.array(
        [
            [174.67, 179.87, 188.46],
            [155.84, 169.58, 185.05],
            [134.90, 144.00, 157.89],
        ]
    )
    return orography


@pytest.fixture()
def land_sea_mask(temperature):
    land_sea_mask = ~np.zeros_like(temperature, dtype=bool)
    return land_sea_mask


def test_returns_expected_values(temperature, orography, land_sea_mask):
    """Test that the function returns expected lapse rate."""

    expected_out = -0.00765005774676
    result, _, _ = LapseRate(nbhood_radius=1)._generate_lapse_rate_array(
        temperature, orography, land_sea_mask
    )
    np.testing.assert_array_almost_equal(result[1, 1], expected_out)


@pytest.mark.parametrize("default, expected", ((None, DALR), (-DALR, -DALR)))
def test_handles_nan(temperature, orography, land_sea_mask, default, expected):
    """Test that the function returns default lapse-rate value when central point
    is NaN."""
    temperature[..., 1, 1] = np.nan
    kwargs = {"default": default} if default is not None else {}
    result, _, _ = LapseRate(nbhood_radius=1, **kwargs)._generate_lapse_rate_array(
        temperature, orography, land_sea_mask
    )
    np.testing.assert_array_almost_equal(result[1, 1], expected)


def test_handles_height_difference(temperature, orography, land_sea_mask):
    """Test that the function calculates the correct value when a large height
    difference is present in the orography data."""
    temperature[..., 1, 1] = 280.03
    orography[..., 0, 0] = 205.0
    expected_out = np.array(
        [
            [0.00358138, -0.00249654, -0.00615844],
            [-0.00759706, -0.00775436, -0.0098],
            [-0.00755349, -0.00655047, -0.0098],
        ]
    )

    result, _, _ = LapseRate(nbhood_radius=1)._generate_lapse_rate_array(
        temperature, orography, land_sea_mask
    )
    np.testing.assert_array_almost_equal(result, expected_out)


@pytest.mark.parametrize(
    "x_outlier, y_outlier, expected_error",
    [
        (0.1, 0.2, 0.0326),
        (np.nan, 0.2, 0.0),
        (0.1, np.nan, 0.0),
        (0.0, 0.0, 0.0),
        (-0.1, 0.1, 0.0614),
        (0.1, -0.1, 0.0652),
        (0.0, 1.0, 0.5**0.5 / (2 * 1.5**2 + 2 * 0.5**2) ** 0.5),
    ],
)
def test_standard_error(x_outlier, y_outlier, expected_error):
    data = np.arange(4)
    x_array = data.reshape((1, 2, 2)).astype(np.float32)
    y_array = data.reshape((1, 2, 2)).astype(np.float32)
    x_array[0, 0, 0] = x_outlier
    y_array[0, 0, 0] = y_outlier
    slope_array = np.full((1, 1, 1), 1.0)
    intercept_array = np.full_like(slope_array, 0.0)
    valid_elements = np.full((1, 1, 1), 4, dtype=np.int32)
    result = LapseRate._standard_error(
        x_array, y_array, slope_array, intercept_array, valid_elements, axis=(-2, -1)
    )
    assert np.allclose(result, expected_error, rtol=1e-3, equal_nan=True)


def test_t_score_():
    dof = np.array([10, 2, 5, 8, 10, 12, 15, 20, 25, 30])
    t = LapseRate._t_score(dof, confidence_level=95.0)
    expected_t_values = [
        2.228,
        4.303,
        2.571,
        2.306,
        2.228,
        2.179,
        2.131,
        2.086,
        2.060,
        2.042,
    ]
    assert np.allclose(t, expected_t_values, atol=1e-3)
    assert t.shape == (10,)


def test_margin_of_error():
    """Test that the margin of error is calculated correctly."""
    x = np.array([[[0.0, 1.0], [2.0, 3.0]]])
    y = np.array([[[1.0, 1.0], [2.0, 3.0]]])
    slope = np.array([[[1.0]]])
    intercept = np.array([[[0.0]]])
    # Manually compute expected margin of error
    expected_margin_of_error = (0.5**0.5 / (2 * 1.5**2 + 2 * 0.5**2) ** 0.5) * 4.303
    result = LapseRate()._margin_of_error(x, y, slope, intercept, axis=(-2, -1))
    assert np.allclose(result, expected_margin_of_error, atol=1e-4)


def test_margin_of_error_many_points():
    """Test that the margin of error is calculated correctly."""
    x = np.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
        ]
    )
    y = np.array(
        [
            [
                [[2.0, 4.1], [6.1, 8.0]],
                [[2.1, 4.1], [6.1, 8.0]],
                [[2.0, 4.1], [6.1, 8.1]],
            ],
            [
                [[2.0, 4.1], [6.1, 8.0]],
                [[2.1, 4.1], [6.1, 8.0]],
                [[2.0, 4.1], [6.1, 8.1]],
            ],
        ]
    )
    slope = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    intercept = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    expected_margin_of_error = [[0.1924, 0.2357, 0.2357], [0.1924, 0.2357, 0.2357]]
    result = LapseRate()._margin_of_error(x, y, slope, intercept, axis=(-2, -1))
    assert np.allclose(result, expected_margin_of_error, atol=1e-4)

    """Test the LapseRate processing works"""


@pytest.fixture()
def temperature_cube():
    """Create cubes containing a regular grid."""
    grid_size = 5
    data = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    height = AuxCoord(
        np.array([1.5], dtype=np.float32), standard_name="height", units="m"
    )
    _temperature = set_up_variable_cube(
        data,
        spatial_grid="equalarea",
        include_scalar_coords=[height],
        standard_grid_metadata="uk_det",
    )
    return _temperature


@pytest.fixture()
def orography_cube(temperature_cube):
    # Copies temperature cube to create orography cube.
    _orography = set_up_variable_cube(
        temperature_cube.data[0].copy(),
        name="surface_altitude",
        units="m",
        spatial_grid="equalarea",
    )
    for coord in ["time", "forecast_period", "forecast_reference_time"]:
        _orography.remove_coord(coord)
    return _orography


@pytest.fixture()
def land_sea_mask_cube(orography_cube):
    # Copies orography cube to create land/sea mask cube.
    _land_sea_mask = orography_cube.copy(
        data=np.ones_like(orography_cube.data, dtype=np.float32)
    )
    _land_sea_mask.rename("land_binary_mask")
    _land_sea_mask.units = cf_units.Unit("1")
    return _land_sea_mask


def test_basic(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the plugin returns expected data type."""
    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    assert isinstance(result, Cube)
    assert result.name() == "air_temperature_lapse_rate"
    assert result.units == "K m-1"


def test_dimensions(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the output cube has the same shape and dimensions as
    the input temperature cube"""
    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    assert np.array_equal(result.shape, temperature_cube.shape)
    assert result.coords(dim_coords=True) == temperature_cube.coords(dim_coords=True)


def test_dimension_order(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test dimension order is preserved if realization is not the leading
    dimension"""
    enforce_coordinate_ordering(temperature_cube, "realization", anchor_start=False)
    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    assert result.coord_dims("realization")[0] == 2


def test_scalar_realization(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test dimensions are treated correctly if the realization coordinate
    is scalar"""
    _temperature = next(temperature_cube.slices_over("realization"))
    result = LapseRate(nbhood_radius=1).process(
        _temperature, orography_cube, land_sea_mask_cube
    )
    assert np.array_equal(result.shape, _temperature.shape)
    assert result.coords(dim_coords=True) == _temperature.coords(dim_coords=True)


def test_model_id_attr(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test model ID attribute can be inherited"""
    result = LapseRate(nbhood_radius=1).process(
        temperature_cube,
        orography_cube,
        land_sea_mask_cube,
        model_id_attr="mosg__model_configuration",
    )
    assert result.attributes["mosg__model_configuration"] == "uk_det"


@pytest.mark.parametrize(
    "replace_arg, error_id", ((0, "Diagnostic"), (1, "Orography"), (2, "Land/Sea mask"))
)
def test_fails_if_input_is_not_cube(
    temperature_cube, orography_cube, land_sea_mask_cube, replace_arg, error_id
):
    """Test code raises a Type Error if input temperature cube is
    not a cube."""
    incorrect_input = 50.0
    args = [temperature_cube, orography_cube, land_sea_mask_cube]
    args[replace_arg] = incorrect_input
    msg = f"{error_id} input is not a cube, but {type(incorrect_input)}"
    with pytest.raises(TypeError, match=msg):
        LapseRate(nbhood_radius=1).process(*args)


def test_fails_if_orography_wrong_units(
    temperature_cube, orography_cube, land_sea_mask_cube
):
    """Test code raises a Value Error if the orography cube is the
    wrong unit."""
    orography_cube.units = "K"
    msg = r"Unable to convert from 'Unit\('K'\)' to 'Unit\('metres'\)'."
    with pytest.raises(ValueError, match=msg):
        LapseRate(nbhood_radius=1).process(
            temperature_cube, orography_cube, land_sea_mask_cube
        )


def test_correct_lapse_rate_units(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the plugin returns the correct unit type"""
    result = LapseRate().process(temperature_cube, orography_cube, land_sea_mask_cube)
    assert result.units == "K m-1"


def test_correct_lapse_rate_units_with_arguments(
    temperature_cube, orography_cube, land_sea_mask_cube
):
    """Test that the plugin returns the correct unit type when non-default
    arguments specified"""
    result = LapseRate(
        max_height_diff=15,
        nbhood_radius=3,
        max_lapse_rate=0.06,
        min_lapse_rate=-0.01,
    ).process(temperature_cube, orography_cube, land_sea_mask_cube)
    assert result.units == "K m-1"


def test_return_single_precision(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function returns cube of float32."""
    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    assert result.dtype == np.float32


@pytest.mark.parametrize("default, expected", ((None, DALR), (-DALR, -DALR)))
def test_constant_orog(
    temperature_cube, orography_cube, land_sea_mask_cube, default, expected
):
    """Test that the function returns expected default lapse-rate values where the
    orography fields are constant values.
    """
    expected_out = np.full((1, 5, 5), expected)
    kwargs = {"default": default} if default is not None else {}

    temperature_cube.data[:, :, :] = 0.08
    temperature_cube.data[:, 1, 1] = 0.09
    orography_cube.data[:, :] = 10

    result = LapseRate(nbhood_radius=1, **kwargs).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out, decimal=4)


def test_fails_if_max_less_min_lapse_rate(
    temperature_cube, orography_cube, land_sea_mask_cube
):
    """Test code raises a Value Error if input maximum lapse rate is
    less than input minimum lapse rate"""
    msg = "Maximum lapse rate is less than minimum lapse rate"

    with pytest.raises(ValueError, match=msg):
        LapseRate(max_lapse_rate=-1, min_lapse_rate=1).process(
            temperature_cube, orography_cube, land_sea_mask_cube
        )


@pytest.mark.parametrize(
    "min_lapse_rate, max_lapse_rate, default",
    (
        (0, 1, None),
        (-2, -1, None),
        (-1, 1, 2),
        (-1, 1, -2),
        (None, None, -1),
        (None, None, 1),
    ),
)
def test_fails_if_default_out_of_bounds(
    temperature_cube,
    orography_cube,
    land_sea_mask_cube,
    max_lapse_rate,
    min_lapse_rate,
    default,
):
    """Test code raises a Value Error if default lapse rate is
    not between the min and max lapse rates."""
    msg = "Default lapse rate is not between the minimum and maximum lapse rates"
    kwargs = {}
    if default is not None:
        kwargs["default"] = default
    if max_lapse_rate is not None:
        kwargs["max_lapse_rate"] = max_lapse_rate
    if min_lapse_rate is not None:
        kwargs["min_lapse_rate"] = min_lapse_rate

    with pytest.raises(ValueError, match=msg):
        LapseRate(**kwargs).process(
            temperature_cube, orography_cube, land_sea_mask_cube
        )


def test_fails_if_nbhood_radius_less_than_zero(
    temperature_cube, orography_cube, land_sea_mask_cube
):
    """Test code raises a Value Error if input neighbourhood radius
    is less than zero"""
    msg = "Neighbourhood radius is less than zero"

    with pytest.raises(ValueError, match=msg):
        LapseRate(nbhood_radius=-1).process(
            temperature_cube, orography_cube, land_sea_mask_cube
        )


def test_fails_if_max_height_diff_less_than_zero(
    temperature_cube, orography_cube, land_sea_mask_cube
):
    """Test code raises a Value Error if the maximum height difference
    is less than zero"""
    msg = "Maximum height difference is less than zero"

    with pytest.raises(ValueError, match=msg):
        LapseRate(max_height_diff=-1).process(
            temperature_cube, orography_cube, land_sea_mask_cube
        )


def test_lapse_rate_limits(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function limits the lapse rate to +DALR and -3*DALR.
    Where DALR = Dry Adiabatic Lapse Rate.
    """
    expected_out = np.array(
        [
            [
                [0.0294, 0.0294, 0.0294, 0.0, DALR],
                [0.0294, 0.0294, 0.0294, 0.0, DALR],
                [0.0294, 0.0294, 0.0294, 0.0, DALR],
                [0.0294, 0.0294, 0.0294, 0.0, DALR],
                [0.0294, 0.0294, 0.0294, 0.0, DALR],
            ]
        ]
    )

    # West data points should be -3*DALR and East should be DALR.
    temperature_cube.data[:, :, 0] = 2
    temperature_cube.data[:, :, 1] = 1
    temperature_cube.data[:, :, 3] = -1
    temperature_cube.data[:, :, 4] = -2
    orography_cube.data[:, :] = 10
    orography_cube.data[:, 0] = 15
    orography_cube.data[:, 3] = 0

    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_specified_max_lapse_rate(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function correctly applies a specified, non default
    maximum lapse rate."""
    expected_out = np.array(
        [
            [
                [0.0392, 0.0392, 0.0, DALR, DALR],
                [0.0392, 0.0392, 0.0, DALR, DALR],
                [0.0392, 0.0392, 0.0, DALR, DALR],
                [0.0392, 0.0392, 0.0, DALR, DALR],
                [0.0392, 0.0392, 0.0, DALR, DALR],
            ]
        ]
    )

    # West data points should be -4*DALR and East should be DALR.
    temperature_cube.data[:, :, 0] = 2
    temperature_cube.data[:, :, 1] = 1
    temperature_cube.data[:, :, 3] = -1
    temperature_cube.data[:, :, 4] = -2
    orography_cube.data[:, :] = 10
    orography_cube.data[:, 0] = 15
    orography_cube.data[:, 2] = 0

    result = LapseRate(nbhood_radius=1, max_lapse_rate=-4 * DALR).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )

    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_specified_min_lapse_rate(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function correctly applies a specified, non default
    minimum lapse rate."""
    expected_out = np.array(
        [
            [
                [0.0294, 0.0294, 0.0, -0.0196, -0.0196],
                [0.0294, 0.0294, 0.0, -0.0196, -0.0196],
                [0.0294, 0.0294, 0.0, -0.0196, -0.0196],
                [0.0294, 0.0294, 0.0, -0.0196, -0.0196],
                [0.0294, 0.0294, 0.0, -0.0196, -0.0196],
            ]
        ]
    )

    # West data points should be -3*DALR and East should be 2*DALR.
    temperature_cube.data[:, :, 0] = 2
    temperature_cube.data[:, :, 1] = 1
    temperature_cube.data[:, :, 3] = -1
    temperature_cube.data[:, :, 4] = -2
    orography_cube.data[:, :] = 10
    orography_cube.data[:, 0] = 15
    orography_cube.data[:, 2] = 0
    orography_cube.data[:, 4] = 12

    result = LapseRate(nbhood_radius=1, min_lapse_rate=2 * DALR).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )

    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_specified_max_and_min_lapse_rate(
    temperature_cube, orography_cube, land_sea_mask_cube
):
    """Test that the function correctly applies a specified, non default
    maximum and minimum lapse rate."""
    expected_out = np.array(
        [
            [
                [0.0392, 0.0392, 0.0, -0.0196, -0.0196],
                [0.0392, 0.0392, 0.0, -0.0196, -0.0196],
                [0.0392, 0.0392, 0.0, -0.0196, -0.0196],
                [0.0392, 0.0392, 0.0, -0.0196, -0.0196],
                [0.0392, 0.0392, 0.0, -0.0196, -0.0196],
            ]
        ]
    )

    # West data points should be -4*DALR and East should be 2*DALR.
    temperature_cube.data[:, :, 0] = 2
    temperature_cube.data[:, :, 1] = 1
    temperature_cube.data[:, :, 3] = -1
    temperature_cube.data[:, :, 4] = -2
    orography_cube.data[:, :] = 10
    orography_cube.data[:, 0] = 15
    orography_cube.data[:, 2] = 0
    orography_cube.data[:, 4] = 12

    result = LapseRate(
        nbhood_radius=1, max_lapse_rate=-4 * DALR, min_lapse_rate=2 * DALR
    ).process(temperature_cube, orography_cube, land_sea_mask_cube)

    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_handles_nan_value(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function handles a NaN temperature value by replacing
    it with DALR.
    """
    expected_out = np.array(
        [
            [
                [DALR, 0.015, 0.01, 0.006428571, 0.005],
                [DALR, 0.015, 0.01, 0.00625, 0.005],
                [DALR, 0.015, DALR, 0.00625, 0.005],
                [DALR, 0.015, 0.01, 0.00625, 0.005],
                [DALR, 0.015, 0.01, 0.006428571, 0.005],
            ]
        ]
    )

    # West data points should be -3*DALR and East should be DALR.
    temperature_cube.data[:, :, 0] = -0.2
    temperature_cube.data[:, :, 1] = -0.1
    temperature_cube.data[:, :, 2] = 0.0
    temperature_cube.data[:, :, 3] = 0.1
    temperature_cube.data[:, :, 4] = 0.2
    temperature_cube.data[:, 2, 2] = np.nan
    orography_cube.data[:, 0:2] = 0
    orography_cube.data[:, 2] = 10
    orography_cube.data[:, 3] = 20
    orography_cube.data[:, 4] = 40

    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_landsea_mask(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function returns DALR values wherever a land/sea
    mask is true. Mask is True for land-points and False for Sea.
    """
    expected_out = np.array(
        [
            [
                [DALR, 0.003, 0.006, 0.009, DALR],
                [DALR, 0.003, 0.006, 0.009, DALR],
                [DALR, 0.003, 0.006, 0.009, DALR],
                [DALR, DALR, DALR, DALR, DALR],
                [DALR, DALR, DALR, DALR, DALR],
            ]
        ]
    )

    # West data points should be -3*DALR and East should be DALR, South
    # should be zero.
    temperature_cube.data[:, :, 0] = 0.02
    temperature_cube.data[:, :, 1] = 0.01
    temperature_cube.data[:, :, 2] = 0.03
    temperature_cube.data[:, :, 3] = -0.01
    temperature_cube.data[:, :, 4] = -0.02
    orography_cube.data[:, :] = 10
    orography_cube.data[:, 2] = 15
    land_sea_mask_cube.data[3:5, :] = 0

    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_mask_max_height_diff(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function removes neighbours where their height
    difference from the centre point is greater than the default
    max_height_diff = 35metres."""
    expected_out = np.array(
        [
            [
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.0065517, -0.003],
                [DALR, DALR, DALR, -0.0065517, DALR],
                [DALR, DALR, DALR, -0.0065517, -0.003],
                [DALR, DALR, DALR, -0.00642857, -0.005],
            ]
        ]
    )

    temperature_cube.data[:, :, 0:2] = 0.4
    temperature_cube.data[:, :, 2] = 0.3
    temperature_cube.data[:, :, 3] = 0.2
    temperature_cube.data[:, :, 4] = 0.1

    orography_cube.data[:, 2] = 10
    orography_cube.data[:, 3] = 20
    orography_cube.data[:, 4] = 40
    orography_cube.data[2, 4] = 60

    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_mask_max_height_diff_arg(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test that the function removes or leaves neighbours where their
    height difference from the centre point is greater than a
    specified, non-default max_height_diff."""
    expected_out = np.array(
        [
            [
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00454128, -0.003],
                [DALR, DALR, DALR, -0.00454128, -0.003],
                [DALR, DALR, DALR, -0.00454128, -0.003],
                [DALR, DALR, DALR, -0.00642857, -0.005],
            ]
        ]
    )

    temperature_cube.data[:, :, 0:2] = 0.4
    temperature_cube.data[:, :, 2] = 0.3
    temperature_cube.data[:, :, 3] = 0.2
    temperature_cube.data[:, :, 4] = 0.1

    orography_cube.data[:, 2] = 10
    orography_cube.data[:, 3] = 20
    orography_cube.data[:, 4] = 40
    orography_cube.data[2, 4] = 60

    result = LapseRate(max_height_diff=50, nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_decr_temp_incr_orog(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test code where temperature is decreasing with height. This is the
    expected scenario for lapse rate.
    """
    expected_out = np.array(
        [
            [
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
            ]
        ]
    )

    temperature_cube.data[:, :, 0:2] = 0.4
    temperature_cube.data[:, :, 2] = 0.3
    temperature_cube.data[:, :, 3] = 0.2
    temperature_cube.data[:, :, 4] = 0.1

    orography_cube.data[:, 2] = 10
    orography_cube.data[:, 3] = 20
    orography_cube.data[:, 4] = 40

    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


def test_decr_temp_decr_orog(temperature_cube, orography_cube, land_sea_mask_cube):
    """Test code where the temperature increases with height."""
    expected_out = np.array(
        [
            [
                [DALR, 0.01, 0.01, 0.00642857, 0.005],
                [DALR, 0.01, 0.01, 0.00642857, 0.005],
                [DALR, 0.01, 0.01, 0.00642857, 0.005],
                [DALR, 0.01, 0.01, 0.00642857, 0.005],
                [DALR, 0.01, 0.01, 0.00642857, 0.005],
            ]
        ]
    )

    temperature_cube.data[:, :, 0:2] = 0.1
    temperature_cube.data[:, :, 2] = 0.2
    temperature_cube.data[:, :, 3] = 0.3
    temperature_cube.data[:, :, 4] = 0.4

    orography_cube.data[:, 2] = 10
    orography_cube.data[:, 3] = 20
    orography_cube.data[:, 4] = 40

    result = LapseRate(nbhood_radius=1).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)


@pytest.mark.parametrize("least_significant_digit", (None, 2, 5))
def test_min_data_value(
    temperature_cube, orography_cube, land_sea_mask_cube, least_significant_digit
):
    """Test that the function returns expected values when the minimum
    data value is set. To prove this, I have taken the same test as test_decr_temp_incr_orog,
    raised the temperature of most points to be a little above zero, where they would not be ignored,
    and used the min_data_value argument in combination with the tolerance derived from the
    least_significant_digit attribute, if present, to cause these points to be ignored so that
    the original expected output is returned.
    """
    expected_out = np.array(
        [
            [
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
                [DALR, DALR, DALR, -0.00642857, -0.005],
            ]
        ]
    )

    temperature_cube.data = np.full_like(temperature_cube.data, 0.05)
    temperature_cube.data[:, :, 0:2] = 0.4
    temperature_cube.data[:, :, 2] = 0.3
    temperature_cube.data[:, :, 3] = 0.2
    temperature_cube.data[:, :, 4] = 0.1
    if least_significant_digit is not None:
        temperature_cube.attributes["least_significant_digit"] = least_significant_digit
        min_data_value = 0.05 - 10 ** (-least_significant_digit)
    else:
        min_data_value = 0.05 - 10 ** (-7)

    orography_cube.data[:, 2] = 10
    orography_cube.data[:, 3] = 20
    orography_cube.data[:, 4] = 40

    result = LapseRate(nbhood_radius=1, min_data_value=min_data_value).process(
        temperature_cube, orography_cube, land_sea_mask_cube
    )
    np.testing.assert_array_almost_equal(result.data, expected_out)

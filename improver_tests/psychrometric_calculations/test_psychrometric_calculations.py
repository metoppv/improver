# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the dry_adiabatic_temperature and dry_adiabatic_pressure methods."""

import numpy as np
import pytest

import improver.constants as consts
from improver.psychrometric_calculations.psychrometric_calculations import (
    _calculate_latent_heat,
    adjust_for_latent_heat,
    dry_adiabatic_pressure,
    dry_adiabatic_temperature,
    saturated_humidity,
)

t_1 = 280.0
p_1 = 100000.0
t_2 = 271.7008
p_2 = 90000.0
t_3 = 263.0
p_3 = 50000.0
t_4 = 227.301
p_4 = 30000.0


@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "method, t1, p1, n2, expected",
    (
        (dry_adiabatic_temperature, t_1, p_1, p_2, t_2),
        (dry_adiabatic_temperature, t_2, p_2, p_1, t_1),
        (dry_adiabatic_pressure, t_1, p_1, t_2, p_2),
        (dry_adiabatic_pressure, t_2, p_2, t_1, p_1),
        (dry_adiabatic_temperature, t_3, p_3, p_4, t_4),
        (dry_adiabatic_temperature, t_4, p_4, p_3, t_3),
        (dry_adiabatic_pressure, t_3, p_3, t_4, p_4),
        (dry_adiabatic_pressure, t_4, p_4, t_3, p_3),
    ),
)
def test_dry_adiabatic_methods(shape, method, t1, p1, n2, expected):
    """Test that we can move between pairs of points in both directions with both methods.
    Point pairs are t_1,p_1 and t_2,p_2. t1,p1 is the starting point for a test and
    n2 is the target point, either temperature or pressure depending on the method being tested.
    """
    result = method(
        np.full(shape, t1, dtype=np.float32),
        np.full(shape, p1, dtype=np.float32),
        np.full(shape, n2, dtype=np.float32),
    )
    assert np.isclose(result, expected).all()
    assert result.shape == shape


@pytest.mark.parametrize("p1, p2", ((50000, -40000), (-50000, 40000)))
def test_dry_adiabatic_errors(p1, p2):
    """Test with some unphysical pressure values (of opposite signs) where we get NaN."""
    result = dry_adiabatic_temperature(
        np.array([100], dtype=np.float32),
        np.array([p1], dtype=np.float32),
        np.array([p2], dtype=np.float32),
    )
    assert np.isnan(result).all()


@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "t, p, expected",
    (
        (243.15, 30000, 0.78882e-3),  # Tephigram value is about 1.1e-3
        (273.15, 60000, 6.3717e-3),
        (273.15, 90000, 4.2481e-3),
        (273.15, 100000, 3.8240e-3),
        (293.15, 100000, 1.47358e-2),
        (290, 100000, 1.20745e-2),
        (185, 100000.0, 8.477e-08),
        (260.65, 99000.0, 1.31079e-03),
        (338.15, 98000.0, 1.77063e-01),
    ),
)
def test_saturated_humidity(shape, t, p, expected):
    """Test the saturated_humidity method"""
    result = saturated_humidity(
        np.full(shape, t, dtype=np.float32), np.full(shape, p, dtype=np.float32)
    )
    assert np.isclose(result, expected).all()
    assert result.shape == shape
    assert result.dtype == np.float32


@pytest.mark.parametrize("invalid_value", (np.nan, np.inf, -np.inf))
def test_saturated_humidity_invalid(invalid_value):
    """Test the saturated_humidity method with invalid inputs"""
    t = np.array([[290.0, invalid_value], [273.15, 270.0]], dtype=np.float32)
    p = np.array([[100000.0, 90000.0], [90000.0, invalid_value]], dtype=np.float32)
    expected = np.array([[1.20745e-2, np.nan], [4.2481e-3, np.nan]], dtype=np.float32)
    result = saturated_humidity(t, p)
    assert np.isclose(result, expected, equal_nan=True).all()
    assert result.shape == t.shape
    assert result.dtype == np.float32


def test_saturated_humidity_masked():
    """Test the saturated_humidity method with masked inputs"""
    t = np.ma.MaskedArray(
        [[290.0, -99], [273.15, 270.0]],
        [[False, True], [False, False]],
        dtype=np.float32,
    )
    p = np.ma.MaskedArray(
        [[100000.0, 90000.0], [90000.0, -99]],
        [[False, False], [False, True]],
        dtype=np.float32,
    )
    expected = np.ma.MaskedArray(
        [[1.20745e-2, -99], [4.2481e-3, -99]],
        [[False, True], [False, True]],
        dtype=np.float32,
    )
    result = saturated_humidity(t, p)
    assert np.isclose(result, expected, equal_nan=True).all()
    np.testing.assert_equal(result.mask, expected.mask)
    assert result.shape == t.shape
    assert result.dtype == np.float32


# Stephen checked these values on a Tephigram.
# Start at point t, p. Move down dry adiabat until you reach q, then up saturated adiabat back to p
# This should coincide with expected_t and expected_q.
@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "t, p, q, expected_t, expected_q",
    (
        (220, 30000, 5.6e-4, 221.2935, 6.48412e-5),
        (280, 90000, 6.976e-3, 280.0804, 6.9435e-3),
        (271, 85000, 6.8369e-3, 275.118, 5.1856e-3),
        (271, 85000, 3.0e-3, 271, 3.0e-3),  # Subsaturated value
        (289, 100000, 1.2828e-2, 290.291, 1.23013e-2),
        (294, 90000, 2.7e-2, 299.811, 2.46185e-2),
        (292.6, 85000, 2.2e-2, 295.943, 2.06319e-2),
        (220, 10000, 2.7e-2, 259.6123, 1.18363e-2),
        # Last item is so extreme that doesn't converge in 6 iterations
    ),
)
def test_saturated_latent_heat(shape, t, p, q, expected_t, expected_q):
    """Test the saturated_latent_heat method"""
    result_t, result_q = adjust_for_latent_heat(
        np.full(shape, t, dtype=np.float32),
        np.full(shape, q, dtype=np.float32),
        np.full(shape, p, dtype=np.float32),
    )
    assert np.isclose(result_t, expected_t).all()
    assert np.isclose(result_q, expected_q).all()
    for r in result_t, result_q:
        assert r.shape == shape
        assert r.dtype == np.float32


def test_saturated_latent_heat_with_large_array():
    """Test the saturated_latent_heat method with a large array, thus triggering 6 iterations.
    This demonstrates that for arrays, so long as at least one point converges, all succeed,
    and a warning is issued."""
    shape = 150

    # These values converge in six iterations
    t, p, q = 220, 30000, 5.6e-4

    t = np.full(shape, t, dtype=np.float32)
    q = np.full(shape, q, dtype=np.float32)
    p = np.full(shape, p, dtype=np.float32)
    expected_t = np.full(shape, 221.2935, dtype=np.float32)
    expected_q = np.full(shape, 6.48412e-5, dtype=np.float32)

    # These values require more than six iterations to fully converge, so the result does
    # not match the equivalent values in test_saturated_latent_heat()
    t[0] = 220
    p[0] = 10000
    q[0] = 2.7e-2
    expected_t[0] = 253.4863
    expected_q[0] = 1.41813e-2

    with pytest.warns(
        RuntimeWarning, match="some failed to converge after 6 iterations"
    ):
        result_t, result_q = adjust_for_latent_heat(t, q, p)

    assert np.isclose(result_t, expected_t).all()
    assert np.isclose(result_q, expected_q).all()
    for r in result_t, result_q:
        assert r.shape[0] == shape
        assert r.dtype == np.float32


@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "t, expected",
    (
        (185.0, 2707271.0),
        (260.65, 2530250.0),
        (273.15, consts.LH_CONDENSATION_WATER),
        (338.15, 2348900.0),
    ),
)
def test_calculate_latent_heat(t, expected, shape):
    """Test latent heat calculation"""
    result = _calculate_latent_heat(np.full(shape, t, dtype=np.float32))
    assert np.isclose(result, expected).all()
    assert result.dtype == np.float32

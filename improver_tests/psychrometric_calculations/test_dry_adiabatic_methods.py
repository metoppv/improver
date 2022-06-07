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
"""Tests for the dry_adiabatic_temperature and dry_adiabatic_pressure methods."""
import numpy as np
import pytest

from improver.psychrometric_calculations.psychrometric_calculations import (
    dry_adiabatic_pressure,
    dry_adiabatic_temperature,
    saturated_humidity,
    saturated_latent_heat,
)

t_1 = 280.0
p_1 = 100000.0
t_2 = 271.7008
p_2 = 90000.0


@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "method, t1, p1, n2, expected",
    (
        (dry_adiabatic_temperature, t_1, p_1, p_2, t_2),
        (dry_adiabatic_temperature, t_2, p_2, p_1, t_1),
        (dry_adiabatic_pressure, t_1, p_1, t_2, p_2),
        (dry_adiabatic_pressure, t_2, p_2, t_1, p_1),
    ),
)
def test_dry_adiabatic_methods(shape, method, t1, p1, n2, expected):
    """Test that we can move between pairs of points in both directions with both methods.
    Point pairs are t_1,p_1 and t_2,p_2. t1,p1 is the starting point for a test and
    n2 is the target point, either temperature or pressure depending on the method being tested."""
    result = method(
        np.full(shape, t1, dtype=np.float32),
        np.full(shape, p1, dtype=np.float32),
        np.full(shape, n2, dtype=np.float32),
    )
    assert np.isclose(result, expected).all()
    assert result.shape == shape


@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "t, p, expected",
    (
        (243.15, 30000, 0.78787e-3),  # Tephigram value is about 1.1e-3
        (273.15, 60000, 6.3074e-3),
        (273.15, 90000, 4.2194e-3),
        (273.15, 100000, 3.8008e-3),
        (293.15, 100000, 1.43954e-2),
        (290, 100000, 1.18451e-2),
    ),
)
def test_saturated_humidity(shape, t, p, expected):
    """Test the saturated_humidity method"""
    result = saturated_humidity(
        np.full(shape, t, dtype=np.float32), np.full(shape, p, dtype=np.float32)
    )
    assert np.isclose(result, expected).all()
    assert result.shape == shape


# Stephen checked these values on a Tephigram.
# Start at point t, p. Move down dry adiabat until you reach q, then up saturated adiabat back to p
# This should coincide with expected_t and expected_q.
@pytest.mark.parametrize("shape", ((1,), (2, 2)))
@pytest.mark.parametrize(
    "t, p, q, expected_t, expected_q",
    (
        (220, 30000, 5.6e-4, 220.6760, 5.99874e-5),
        (280, 90000, 6.9e-3, 280.0584, 6.8568e-3),
        (271, 85000, 6.8369e-3, 273.879, 4.7071e-3),
        (289, 100000, 1.25845e-2, 290, 1.1845e-2),
        (294, 90000, 2.7e-2, 299.341, 2.30493e-2),
    ),
)
def test_saturated_latent_heat(shape, t, p, q, expected_t, expected_q):
    """Test the saturated_latent_heat method"""
    result_t, result_q = saturated_latent_heat(
        np.full(shape, t, dtype=np.float32),
        np.full(shape, q, dtype=np.float32),
        np.full(shape, p, dtype=np.float32),
    )
    assert np.isclose(result_t, expected_t).all()
    assert np.isclose(result_q, expected_q).all()
    assert result_t.shape == shape
    assert result_q.shape == shape

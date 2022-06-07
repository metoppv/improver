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
def test_round_trip(shape, method, t1, p1, n2, expected):
    """Test that we can move between pairs of points in both directions with both methods"""
    result = method(
        np.full(shape, t1, dtype=np.float32),
        np.full(shape, p1, dtype=np.float32),
        np.full(shape, n2, dtype=np.float32),
    )
    assert np.isclose(result, expected).all()
    assert result.shape == shape

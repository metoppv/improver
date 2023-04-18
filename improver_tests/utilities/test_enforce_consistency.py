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

from improver.synthetic_data.set_up_test_cubes import set_up_percentile_cube
from improver.utilities.enforce_consistency import EnforceConsistentPercentiles


def get_percs(value, shape):
    """Correctly shape the value provided into the desired shape. The value is
    used as the middle of three percentiles. The value is subtracted by 10 or added by
    10 to create the bounding percentiles. These values are clipped to ensure that
    the values are positive."""
    return np.broadcast_to(
        np.expand_dims(np.clip([value - 10, value, value + 10], 0, None), axis=(1, 2),),
        shape,
    )


def get_wind_speed_cube(wind_speed, shape):
    """Create a wind speed percentile cube."""
    wind_speed_percs = get_percs(wind_speed, shape)
    wind_speed_cube = set_up_percentile_cube(
        np.full(shape, wind_speed_percs, dtype=np.float32),
        [10, 50, 90],
        name="wind_speed_at_10m",
        units="m s-1",
    )
    return wind_speed_cube


def get_wind_gust_cube(wind_gust, shape):
    """Create a wind gust percentile cube."""
    wind_gust_percs = get_percs(wind_gust, shape)
    wind_gust_cube = set_up_percentile_cube(
        np.full(shape, wind_gust_percs, dtype=np.float32),
        [10, 50, 90],
        name="wind_gust_at_10m_max-PT01H",
        units="m s-1",
    )
    return wind_gust_cube


@pytest.mark.parametrize(
    "min_percentage_exceedance,wind_speed,wind_gust,expected_values",
    (
        (0, 10, 10, [0.0, 10, 20]),  # Values unchanged. 0 percentage.
        (10, 10, 10, [0.0, 11, 22]),  # 10% increase.
        (10, 10, 20, [10, 20, 30],),  # Percentage exceedance has no effect.
        (10, 20, 10, [11, 22, 33],),  # Gust speed 10% greater than wind speed.
        (20, 10, 10, [0.0, 12, 24]),  # 20% increase.
        (100, 10, 10, [0.0, 20, 40]),  # 100% increase.
    ),
)
def test_basic(min_percentage_exceedance, wind_speed, wind_gust, expected_values):
    """Test that consistency between percentiles is enforced for a variety of
    percentages, wind speeds and wind gusts."""
    shape = (3, 2, 2)
    wind_speed_cube = get_wind_speed_cube(wind_speed, shape)
    wind_gust_cube = get_wind_gust_cube(wind_gust, shape)

    result = EnforceConsistentPercentiles(
        min_percentage_exceedance=min_percentage_exceedance
    )(wind_speed_cube, wind_gust_cube.copy())

    expected = wind_gust_cube.copy()
    expected.data = np.broadcast_to(np.expand_dims(expected_values, axis=(1, 2)), shape)

    assert isinstance(result, Cube)
    assert result.name() == wind_gust_cube.name()
    np.testing.assert_array_almost_equal(result.data, expected.data)


@pytest.mark.parametrize(
    "min_percentage_exceedance", ((-10, 110)),
)
def test_exception(min_percentage_exceedance):
    """Test that an exception is raised if the percentage representing the
    minimum that the reference forecast must be exceeded by is outside the
    range 0 to 100"""
    msg = "The percentage representing the minimum"
    with pytest.raises(ValueError, match=msg):
        EnforceConsistentPercentiles(
            min_percentage_exceedance=min_percentage_exceedance
        )

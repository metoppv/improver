#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Unit tests for the FillRadarHoles plugin"""

import pytest
import numpy as np
from numpy.ma import MaskedArray
from iris.cube import Cube

from improver.nowcasting.utilities import FillRadarHoles
from ...set_up_test_cubes import set_up_variable_cube


def create_masked_rainrate_data():
    """Create a masked data array of rain rates in mm/h"""
    nonzero_data = np.array(
        [[0.03, 0.1, 0.1, 0.1, 0.03],
         [0.1, 0.2, 0.2, np.nan, 0.1],
         [0.2, 0.5, np.nan, np.nan, 0.2],
         [0.1, 0.5, np.nan, np.nan, 0.1],
         [0.03, 0.2, 0.2, 0.1, 0.03]])
    data = np.zeros((16, 16), dtype=np.float32)
    data[5:12, 5:12] = np.full((7, 7), 0.03, dtype=np.float32)
    data[6:11, 6:11] = nonzero_data.astype(np.float32)
    mask = np.where(np.isfinite(data), False, True)
    return MaskedArray(data, mask=mask)


RAIN_DATA = create_masked_rainrate_data()

INTERPOLATED_RAIN_DATA = RAIN_DATA.copy()
INTERPOLATED_RAIN_DATA.data[7:10, 8:10] = [
    [0.2, 0.07138586],
    [0.11366593, 0.09165306],
    [0.09488520, 0.07650946]
]
INTERPOLATED_RAIN_DATA.mask = np.full((16, 16), False)

RAIN_CUBE = set_up_variable_cube(
    RAIN_DATA, name="lwe_precipitation_rate", units="mm/h",
    spatial_grid="equalarea"
)

MASKED_RAIN_CUBE = RAIN_CUBE.copy()
MASKED_RAIN_CUBE.data[:11, :11] = np.nan
MASKED_RAIN_CUBE.data.mask = np.where(
    np.isfinite(MASKED_RAIN_CUBE.data.data), False, True
)

# set up alternate cases for which interpolation should ("speckle") and
# should not ("masked") be triggered
CASES = ["speckle", "masked"]
INPUT_CUBES = {"speckle": RAIN_CUBE,
               "masked": MASKED_RAIN_CUBE}
OUTPUT_DATA = {"speckle": INTERPOLATED_RAIN_DATA,
               "masked": MASKED_RAIN_CUBE.data.copy()}

PLUGIN = FillRadarHoles()


def test_basic():
    """Test that the plugin returns a masked cube"""
    result = PLUGIN(RAIN_CUBE)
    assert isinstance(result, Cube)
    assert isinstance(result.data, MaskedArray)


@pytest.mark.parametrize("case", CASES)
def test_values(case):
    """Test that the data and mask contain the expected values"""
    result = PLUGIN(INPUT_CUBES[case])
    assert np.allclose(result.data, OUTPUT_DATA[case])
    assert np.array_equal(result.data.mask, OUTPUT_DATA[case].mask)








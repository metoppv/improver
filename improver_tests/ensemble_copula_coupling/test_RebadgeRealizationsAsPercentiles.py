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
"""
Unit tests for the `ensemble_copula_coupling.RebadgeRealizationsAsPercentiles` class.

"""
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgeRealizationsAsPercentiles,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_variable_cube,
)

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS


@pytest.mark.parametrize("scalar_realization", (True, False,))
@pytest.mark.parametrize("optimal_crps_percentiles", (True, False,))
@pytest.mark.parametrize(
    "data", (ECC_TEMPERATURE_REALIZATIONS, ECC_TEMPERATURE_REALIZATIONS[::-1])
)
def test_process(data, optimal_crps_percentiles, scalar_realization):
    """Check that rebadging realizations as percentiles gives the desired output."""
    if scalar_realization:
        cube = set_up_variable_cube(ECC_TEMPERATURE_REALIZATIONS[1])
        realization_coord = DimCoord([0], standard_name="realization", units="1")
        cube.add_aux_coord(realization_coord)
        expected_data = ECC_TEMPERATURE_REALIZATIONS[1]
        percentiles = [50]
    else:
        cube = set_up_variable_cube(data, realizations=[0, 1, 2])
        expected_data = ECC_TEMPERATURE_REALIZATIONS
        if optimal_crps_percentiles:
            percentiles = [16.6666, 50, 83.3333]
        else:
            percentiles = [25, 50, 75]

    result = RebadgeRealizationsAsPercentiles(
        optimal_crps_percentiles=optimal_crps_percentiles
    )(cube)

    assert isinstance(result, Cube)
    assert len(result.coords("percentile")) == 1
    assert len(result.coords("realization")) == 0
    assert result.coord("percentile").dtype == np.float32
    assert result.coord("percentile").units == "%"
    np.testing.assert_allclose(
        result.coord("percentile").points, percentiles, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(result.data, expected_data)


def test_exception():
    """Check that an exception is raised, if the input does not have a
    realization coordinate."""
    cube = set_up_percentile_cube(
        ECC_TEMPERATURE_REALIZATIONS, percentiles=[25, 50, 75]
    )

    with pytest.raises(
        CoordinateNotFoundError, match="No realization coordinate is present"
    ):
        RebadgeRealizationsAsPercentiles()(cube)

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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

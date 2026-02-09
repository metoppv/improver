# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.RebadgePercentilesAsRealizations` class.
"""

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import InvalidCubeError

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_percentile_cube

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS


@pytest.fixture
def percentile_cube():
    """Fixture to set up a temperature percentile cube for testing."""
    return set_up_percentile_cube(
        np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
        np.array([25, 50, 75], dtype=np.float32),
    )


def test_basic(percentile_cube):
    """Test that a cube is produced with a realization dimension."""
    result = Plugin().process(percentile_cube)
    assert isinstance(result, Cube)
    assert isinstance(result.coord("realization"), DimCoord)
    assert result.coord("realization").units == "1"


def test_specify_realization_numbers(percentile_cube):
    """Test specifying particular values for the ensemble realization numbers."""
    ensemble_realization_numbers = [12, 13, 14]
    result = Plugin(ensemble_realization_numbers=ensemble_realization_numbers).process(
        percentile_cube
    )
    np.testing.assert_array_equal(
        result.coord("realization").points, ensemble_realization_numbers
    )


@pytest.mark.parametrize("ensure_evenly_spaced_percentiles", [True, False])
def test_ensure_evenly_spaced_percentiles(
    percentile_cube, ensure_evenly_spaced_percentiles
):
    """Test specifying particular values for the ensemble realization numbers."""
    percentile_cube.coord("percentile").points = np.array(
        [10, 50, 90], dtype=np.float32
    )
    if ensure_evenly_spaced_percentiles:
        msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
        with pytest.raises(ValueError, match=msg):
            Plugin(
                ensure_evenly_spaced_percentiles=ensure_evenly_spaced_percentiles
            ).process(percentile_cube)
    else:
        result = Plugin(
            ensure_evenly_spaced_percentiles=ensure_evenly_spaced_percentiles
        ).process(percentile_cube)
        np.testing.assert_array_equal(
            result.coord("realization").points, np.array([0, 1, 2])
        )


def test_number_of_realizations(percentile_cube):
    """Check the values for the realization coordinate generated without specifying numbers."""
    result = Plugin().process(percentile_cube)
    np.testing.assert_array_almost_equal(
        result.coord("realization").points, np.array([0, 1, 2])
    )


def test_raises_exception_if_realization_already_exists(percentile_cube):
    """Check that an exception is raised if a realization coordinate already exists."""
    percentile_cube.add_aux_coord(AuxCoord(0, "realization"))
    msg = r"Cannot rebadge percentile coordinate to realization.*"
    with pytest.raises(InvalidCubeError, match=msg):
        Plugin().process(percentile_cube)


def test_raises_exception_if_percentiles_unevenly_spaced(percentile_cube):
    """Check that an exception is raised if the input percentiles are not evenly spaced."""
    percentile_cube.coord("percentile").points = np.array(
        [25, 50, 90], dtype=np.float32
    )
    msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
    with pytest.raises(ValueError, match=msg):
        Plugin().process(percentile_cube)


def test_raises_exception_if_percentiles_not_centred(percentile_cube):
    """Check that an exception is raised if the input percentiles are not centred on 50th percentile."""
    percentile_cube.coord("percentile").points = np.array(
        [30, 60, 90], dtype=np.float32
    )
    msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
    with pytest.raises(ValueError, match=msg):
        Plugin().process(percentile_cube)


def test_raises_exception_if_percentiles_unequal_partition_percentile_space(
    percentile_cube,
):
    """Check that an exception is raised if the input percentiles don't evenly partition percentile space."""
    percentile_cube.coord("percentile").points = np.array(
        [10, 50, 90], dtype=np.float32
    )
    msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
    with pytest.raises(ValueError, match=msg):
        Plugin().process(percentile_cube)

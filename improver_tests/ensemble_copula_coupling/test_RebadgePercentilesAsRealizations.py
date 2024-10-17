# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.RebadgePercentilesAsRealizations` class.

"""
import unittest

import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_percentile_cube

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS


class Test_process(IrisTest):

    """Test the process method of the
    RebadgePercentilesAsRealizations plugin."""

    def setUp(self):
        """Set up temperature percentile cube for testing"""
        self.cube = set_up_percentile_cube(
            np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
            np.array([25, 50, 75], dtype=np.float32),
        )

    def test_basic(self):
        """Test that a cube is produced with a realization dimension"""
        result = Plugin().process(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.coord("realization"), DimCoord)
        self.assertEqual(result.coord("realization").units, "1")

    def test_specify_realization_numbers(self):
        """Use the ensemble_realization_numbers optional argument to specify
        particular values for the ensemble realization numbers."""
        ensemble_realization_numbers = [12, 13, 14]
        result = Plugin().process(self.cube, ensemble_realization_numbers)
        self.assertArrayEqual(
            result.coord("realization").points, ensemble_realization_numbers
        )

    def test_number_of_realizations(self):
        """Check the values for the realization coordinate generated without
        specifying the ensemble_realization_numbers argument."""
        result = Plugin().process(self.cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, np.array([0, 1, 2])
        )

    def test_raises_exception_if_realization_already_exists(self):
        """Check that an exception is raised if a realization
        coordinate already exists."""
        self.cube.add_aux_coord(AuxCoord(0, "realization"))
        msg = r"Cannot rebadge percentile coordinate to realization.*"
        with self.assertRaisesRegex(InvalidCubeError, msg):
            Plugin().process(self.cube)

    def test_raises_exception_if_percentiles_unevenly_spaced(self):
        """Check that an exception is raised if the input percentiles
        are not evenly spaced."""
        cube = set_up_percentile_cube(
            np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
            np.array([25, 50, 90], dtype=np.float32),
        )
        msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(cube)

    def test_raises_exception_if_percentiles_not_centred(self):
        """Check that an exception is raised if the input percentiles
        are not centred on 50th percentile."""
        cube = set_up_percentile_cube(
            np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
            np.array([30, 60, 90], dtype=np.float32),
        )
        msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(cube)

    def test_raises_exception_if_percentiles_unequal_partition_percentile_space(self):
        """Check that an exception is raised if the input percentiles
        don't evenly partition percentile space."""
        cube = set_up_percentile_cube(
            np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
            np.array([10, 50, 90], dtype=np.float32),
        )
        msg = r"The percentile cube provided cannot be rebadged as ensemble realizations.*"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(cube)


if __name__ == "__main__":
    unittest.main()

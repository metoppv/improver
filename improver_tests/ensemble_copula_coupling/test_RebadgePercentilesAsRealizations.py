# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
            np.array([10, 50, 90], dtype=np.float32),
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
        """Check that we raise an exception if a realization coordinate already
        exists."""
        self.cube.add_aux_coord(AuxCoord(0, "realization"))
        msg = r"Cannot rebadge percentile coordinate to realization.*"
        with self.assertRaisesRegex(InvalidCubeError, msg):
            Plugin().process(self.cube)


if __name__ == "__main__":
    unittest.main()

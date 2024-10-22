# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the generate_ancillary.CorrectLandSeaMask plugin."""

import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_ancillary import (
    CorrectLandSeaMask as CorrectLand,
)


class Test_process(IrisTest):
    """Test the land-sea mask correction plugin."""

    def setUp(self):
        """setting up paths to test ancillary files"""
        landmask_data = np.array([[0.2, 0.0, 0.0], [0.7, 0.5, 0.05], [1, 0.95, 0.7]])
        self.landmask = Cube(landmask_data, long_name="test land")
        self.expected_mask = np.array(
            [[False, False, False], [True, True, False], [True, True, True]]
        )

    def test_landmaskcorrection(self):
        """Test landmask correction. Note that the name land_binary_mask is
        enforced to reflect the change that has been made."""
        result = CorrectLand().process(self.landmask)
        self.assertEqual(result.name(), "land_binary_mask")
        self.assertArrayEqual(result.data, self.expected_mask)
        self.assertTrue(result.dtype == np.int8)


if __name__ == "__main__":
    unittest.main()

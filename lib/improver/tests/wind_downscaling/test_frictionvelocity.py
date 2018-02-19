# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for plugin wind_downscaling.FrictionVelocity."""


import unittest

from cf_units import Unit
import numpy as np

import iris
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.grids import STANDARD_GRID_CCRS
from improver.constants import RMDI
from improver.wind_downscaling import FrictionVelocity


class Test_calc_ustar(IrisTest):

    """Test the calc_ustar method."""

    """
    Args:
        u_href: 2D np.array (float)
            wind speed at h_ref
        h_ref:  2D np.array (float)
            reference height
        z_0:    2D np.array (float)
            vegetative roughness length
        mask:   2D np.array (logical)
            where True, calculate u*

    comments:
        * z_0 and h_ref need to have identical units.
        * the calculated friction velocity will have the units of the
            supplied velocity u_href.

    """

    def setUp(self):
        """Create 2D arrays for testing."""
        self.u_href = u_href_array
        self.h_ref = h_ref_array
        self.z_0 = z_0_array
        self.mask = mask_array

    def test_basic(self):
        """Test that the function returns a 2D array. """

        result = FrictionVelocity(u_href, h_ref, z_0)
        self.assertIsInstance(result, numpy array)


    def test_without_mask(self):
        """Test that the function returns the expected values without
           a mask applied. """


        result = FrictionVelocity(u_href, h_ref, z_0)
        self.assertIsInstance(result, numpy array)



    def test_with_mask(self):
        """Test that the function returns the expected values with a mask."""

        result = FrictionVelocity(u_href, h_ref, z_0, mask)
        self.assertIsInstance(result, numpy array)


if __name__ == '__main__':
    unittest.main()

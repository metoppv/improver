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
"""Unit tests for nbhood.circular_kernel.check_radius_against_distance."""

import unittest

from iris.tests import IrisTest
import numpy as np

from improver.nbhood.circular_kernel import check_radius_against_distance
from improver_tests.set_up_test_cubes import set_up_variable_cube


class Test_check_radius_against_distance(IrisTest):

    """Test check_radius_against_distance function."""

    def setUp(self):
        """Set up the cube."""
        data = np.ones((4, 4), dtype=np.float32)
        self.cube = set_up_variable_cube(data, spatial_grid='equalarea')

    def test_error(self):
        """Test correct exception raised when the distance is larger than the
        corner-to-corner distance of the domain."""
        distance = 550000.0
        msg = "Distance of 550000.0m exceeds max domain distance of "
        with self.assertRaisesRegex(ValueError, msg):
            check_radius_against_distance(self.cube, distance)

    def test_passes(self):
        """Test no exception raised when the distance is smaller than the
        corner-to-corner distance of the domain."""
        distance = 6100
        check_radius_against_distance(self.cube, distance)


if __name__ == '__main__':
    unittest.main()

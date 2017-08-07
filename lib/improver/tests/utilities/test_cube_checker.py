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
"""Unit tests for the cube_checker utility."""


import unittest

from iris.tests import IrisTest

from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)


class Test_check_for_x_and_y_axes(IrisTest):

    """Test whether the cube has an x and y axis."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_no_y_coordinate(self):
        """Test that the expected exception is raised, if there is no
        y coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_x_coordinate"]):
            break
        sliced_cube.remove_coord("projection_y_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegexp(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)

    def test_no_x_coordinate(self):
        """Test that the expected exception is raised, if there is no
        x coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_y_coordinate"]):
            break
        sliced_cube.remove_coord("projection_x_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegexp(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)


if __name__ == '__main__':
    unittest.main()

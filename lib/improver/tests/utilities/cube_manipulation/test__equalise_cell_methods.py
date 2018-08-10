# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
Unit tests for the utilities within the "cube_manipulation" module.

"""
import unittest

import iris
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import _equalise_cell_methods

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube

from improver.utilities.warnings_handler import ManageWarnings


class Test__equalise_cell_methods(IrisTest):

    """Test the_equalise_cube_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cell_method1 = iris.coords.CellMethod("mean", "realization")
        self.cell_method2 = iris.coords.CellMethod("mean", "time")
        self.cell_method3 = iris.coords.CellMethod("max", "neighbourhood")

    def test_basic(self):
        """Test returns an iris.cube.CubeList."""
        result = _equalise_cell_methods(iris.cube.CubeList([self.cube,
                                                            self.cube]))
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].is_compatible(result[1]))

    @ManageWarnings(record=True)
    def test_single_cube_in_cubelist(self, warning_list=None):
        """Test single cube in CubeList returns CubeList and raises warning."""
        result = _equalise_cell_methods(iris.cube.CubeList([self.cube]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = ("Only a single cube so no differences "
                       "will be found in cell methods")
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_different_cell_methods(self):
        """Test returns an iris.cube.CubeList with matching cell methods."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube3 = self.cube.copy()
        cube1.cell_methods = tuple([self.cell_method1, self.cell_method2])
        cube2.cell_methods = tuple([self.cell_method1, self.cell_method2,
                                    self.cell_method3])
        cube3.cell_methods = tuple([self.cell_method1, self.cell_method3])
        result = _equalise_cell_methods(iris.cube.CubeList([cube1,
                                                            cube2,
                                                            cube3]))
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result[0].cell_methods), 1)
        check = result[1].cell_methods[0] == self.cell_method1
        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()

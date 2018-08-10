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

from improver.utilities.cube_manipulation import add_renamed_cell_method

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


class Test_add_renamed_cell_method(IrisTest):
    """Class to test the add_renamed_cell_method function"""

    def setUp(self):
        """Set up input cube for tests"""
        self.cube = set_up_temperature_cube()
        self.cell_method = iris.coords.CellMethod(method='mean', coords='time')
        self.cube.add_cell_method(self.cell_method)

    def test_basic(self):
        """Basic test for one cell method on input cube"""
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods, (expected_cell_method,))

    def test_only_difference_is_name(self):
        """Testing that the input cell method and the new cell method only
        differ by name"""
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods, (expected_cell_method,))
        new_cell_method = self.cube.cell_methods[0]
        self.assertEqual(self.cell_method.coord_names,
                         new_cell_method.coord_names)
        self.assertEqual(self.cell_method.intervals, new_cell_method.intervals)
        self.assertEqual(self.cell_method.comments, new_cell_method.comments)

    def test_no_cell_method_in_input_cube(self):
        """Testing that when there are no cell methods on the input cube then
        the new cell method still gets added as expected."""
        self.cube.cell_methods = ()
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods, (expected_cell_method,))

    def test_wrong_input(self):
        """Test a sensible error is raised when the wrong input is passed in"""
        self.cube.cell_methods = ()
        message = ('Input Cell_method is not an instance of '
                   'iris.coord.CellMethod')
        with self.assertRaisesRegex(TypeError, message):
            add_renamed_cell_method(self.cube, 'not_a_cell_method',
                                    'weighted_mean')

    def test_multiple_cell_methods_in_input_cube(self):
        """Test that other cell methods are preserved."""
        extra_cell_method = iris.coords.CellMethod(method='max',
                                                   coords='realization')
        self.cube.cell_methods = (self.cell_method, extra_cell_method)
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods,
                         (extra_cell_method, expected_cell_method,))


if __name__ == '__main__':
    unittest.main()

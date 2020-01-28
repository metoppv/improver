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
"""
Unit tests for the function collapsed.
"""

import unittest

import iris
import numpy as np

from improver.utilities.cube_manipulation import collapsed
from ...set_up_test_cubes import set_up_variable_cube


class Test_collapsed(unittest.TestCase):

    """Test the collapsed utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 281*np.ones((3, 3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(data, realizations=[0, 1, 2])

    def test_single_method(self):
        """Test that a collapsed cube is returned with no cell method added"""
        result = collapsed(self.cube, 'realization', iris.analysis.MEAN)
        self.assertTupleEqual(result.cell_methods, ())
        self.assertTrue((result.data == self.cube.collapsed(
            'realization', iris.analysis.MEAN).data).all())

    def test_two_methods(self):
        """Test that a cube keeps its original cell method but another
        isn't added.
        """
        cube = self.cube
        method = iris.coords.CellMethod('test')
        cube.add_cell_method(method)
        result = collapsed(cube, 'realization', iris.analysis.MEAN)
        self.assertTupleEqual(result.cell_methods, (method,))
        self.assertTrue((result.data == cube.collapsed(
            'realization', iris.analysis.MEAN).data).all())


if __name__ == '__main__':
    unittest.main()

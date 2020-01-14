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
"""Unit tests for the uv_index function."""

import unittest

import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.uv_index import calculate_uv_index

from ..set_up_test_cubes import set_up_variable_cube


class Test_uv_index(IrisTest):
    """ Tests that the uv_index plugin calculates the UV index
    correctly. """

    def setUp(self):
        """Set up the cubes for upward and downward uv fluxes,
        and also one with different units."""
        data_up = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                           dtype=np.float32)
        uv_up_name = 'surface_upwelling_ultraviolet_flux_in_air'
        self.cube_uv_up = set_up_variable_cube(data_up,
                                               name=uv_up_name,
                                               units='W m-2')
        self.cube_up_badname = set_up_variable_cube(data_up,
                                                    name='Wrong name',
                                                    units='W m-2')
        data_down = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                             dtype=np.float32)
        uv_down_name = 'surface_downwelling_ultraviolet_flux_in_air'
        self.cube_uv_down = set_up_variable_cube(data_down,
                                                 name=uv_down_name,
                                                 units='W m-2')
        self.cube_diff_units = set_up_variable_cube(data_down,
                                                    name=uv_down_name,
                                                    units='m')
        self.cube_down_badname = set_up_variable_cube(data_down,
                                                      name='Wrong name',
                                                      units='W m-2')

    def test_basic(self):
        """ Test that the a basic uv calculation works, using the
        default scaling factor. Make sure the output is a cube
        with the expected data."""
        scale_factor = 1.0
        expected = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
                            dtype=np.float32)
        result = calculate_uv_index(self.cube_uv_up, self.cube_uv_down,
                                    scale_factor)
        self.assertArrayEqual(result.data, expected)

    def test_scale_factor(self):
        """ Test the uv calculation works when changing the scale factor. Make
        sure the output is a cube with the expected data."""
        expected = np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
                            dtype=np.float32)
        result = calculate_uv_index(self.cube_uv_up, self.cube_uv_down,
                                    scale_factor=10)
        self.assertArrayEqual(result.data, expected)

    def test_metadata(self):
        """ Tests that the uv index output has the correct metadata (no units,
        and name = ultraviolet index)."""
        result = calculate_uv_index(self.cube_uv_up, self.cube_uv_down)
        self.assertEqual(str(result.standard_name), 'ultraviolet_index')
        self.assertIsNone(result.var_name)
        self.assertIsNone(result.long_name)
        self.assertEqual((result.units), Unit("1"))

    def test_diff_units(self):
        """Tests that a ValueError is raised if the input uv files have
        different units. """
        msg = 'The input uv files do not have the same units.'
        with self.assertRaisesRegex(ValueError, msg):
            calculate_uv_index(self.cube_uv_up, self.cube_diff_units,
                               scale_factor=1.0)

    def test_badname_down(self):
        """Tests that a ValueError is raised if the input uv down
        file has the wrong name. """
        msg = 'The radiation flux in UV downward'
        with self.assertRaisesRegex(ValueError, msg):
            calculate_uv_index(self.cube_uv_up,
                               self.cube_down_badname)

    def test_badname_up(self):
        """Tests that a ValueError is raised if the input uv up
        file has the wrong name. """
        msg = 'The radiation flux in UV upward'
        with self.assertRaisesRegex(ValueError, msg):
            calculate_uv_index(self.cube_up_badname,
                               self.cube_uv_down)


if __name__ == '__main__':
    unittest.main()

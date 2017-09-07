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
"""Unit tests for the threshold.BasicThreshold plugin."""


import unittest

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.nowcast.lightning import NowcastLightning as Plugin
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import set_up_cube


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(Plugin())
        msg = ('<NowcastLightning: radius=10000.0, debug=False>')
        self.assertEqual(result, msg)


class Test__process_haloes(IrisTest):

    """Test the _process_haloes method."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.cube = set_up_cube()

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin._process_haloes(self.cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        incube = self.cube.copy()
        result = plugin._process_haloes(incube)
        self.assertArrayAlmostEqual(incube.data, self.cube.data)

    def test_data(self):
        """Test that the method returns the expected data"""
        plugin = Plugin(4000.)
        expected = self.cube.data.copy()
        expected[0, 0, 6, :] = [1., 1., 1., 1., 1., 1., 11./12., 0.875, 11./12., 1., 1., 1., 1., 1., 1., 1.]
        expected[0, 0, 7, :] = [1., 1., 1., 1., 1., 1., 0.875, 5./6., 0.875, 1., 1., 1., 1., 1., 1., 1.]
        expected[0, 0, 8, :] = [1., 1., 1., 1., 1., 1., 11./12., 0.875, 11./12., 1., 1., 1., 1., 1., 1., 1.]
        result = plugin._process_haloes(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test__update_meta(IrisTest):

    """Test the _update_meta method."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.cube = set_up_cube()

    def test_basic(self):
        """Test that the method returns the expected cube type
        and that the metadata are as expected.
        We expect a new name and an empty dictionary of attributes."""
        plugin = Plugin()
        self.cube.attributes = {'source': 'testing'}
        result = plugin._update_meta(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "lightning_probability")
        self.assertEqual(result.attributes, {})

    def test_input(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        incube = self.cube.copy()
        result = plugin._update_meta(incube)
        self.assertArrayAlmostEqual(incube.data, self.cube.data)


class Test__modify_first_guess(IrisTest):

    """Test the _modify_first_guess method."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.cube = set_up_cube()
        self.fg_cube = set_up_cube()
        self.ltng_cube = set_up_cube()
        self.precip_cube = set_up_cube()

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        cube_a = self.cube.copy()
        cube_b = self.fg_cube.copy()
        cube_c = self.ltng_cube.copy()
        cube_d = self.precip_cube.copy()
        result = plugin._modify_first_guess(cube_a, cube_b, cube_c, cube_d)
        self.assertArrayAlmostEqual(cube_a.data, self.cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_c.data, self.ltng_cube.data)
        self.assertArrayAlmostEqual(cube_d.data, self.precip_cube.data)


#class Test_process(IrisTest):

    #"""Test the thresholding plugin."""

    #def setUp(self):
        #"""Create a cube with a single non-zero point."""
        #self.cube = def_cube()

    #def test_basic(self):
        #"""Test that the plugin returns an iris.cube.Cube."""
        #fuzzy_factor = 0.95
        #threshold = 0.1
        #plugin = Plugin()
        #result = plugin.process(self.cube)
        #self.assertIsInstance(result, Cube)

    #def test_metadata_changes(self):
        #"""Test the metadata altering functionality"""
        ## Copy the cube as the cube.data is used as the basis for comparison.
        #cube = self.cube.copy()
        #plugin = Plugin()
        #result = plugin.process(cube)
        ## The single 0.5-valued point => 1.0, so cheat by * 2.0 vs orig data.
        #name = "probability_of_{}"
        #expected_name = name.format(self.cube.name())
        #expected_attribute = "above"
        #expected_units = 1
        #expected_coord = DimCoord(0.1,
                                  #long_name='threshold',
                                  #units=self.cube.units)
        #self.assertEqual(result.name(), expected_name)
        #self.assertEqual(result.attributes['relative_to_threshold'],
                         #expected_attribute)
        #self.assertEqual(result.units, expected_units)
        #self.assertEqual(result.coord('threshold'),
                         #expected_coord)

if __name__ == '__main__':
    unittest.main()

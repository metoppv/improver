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
"""Unit tests for the nowcast.convection.handle_vii.ApplyIce plugin."""

import unittest
from iris.tests import IrisTest

from iris.util import squeeze
import cf_units
import numpy as np
from iris.cube import Cube
from improver.nowcast.convection.handle_vii import ApplyIce as Plugin
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_with_no_realizations)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(Plugin())
        msg = ("""<ApplyIce:
 VII (ice) mapping (kg/m2):
   upper:  VII {viiu} => max lightning prob {lviiu}
   middle: VII {viim} => max lightning prob {lviim}
   lower:  VII {viil} => max lightning prob {lviil}
>""".format(viiu=2.0, viim=1.0, viil=0.5,
            lviiu=0.9, lviim=0.5, lviil=0.1)
            )
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the nowcast convection handle_vii ApplyIce plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]))
        self.fg_cube.rename("probability_of_lightning")
        self.fg_cube.coord('forecast_period').points = [0.]
        self.ice_cube = squeeze(
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3,
                            zero_point_indices=[]),
                fp_point=0.0))
        threshold_coord = self.ice_cube.coord('realization')
        threshold_coord.points = [0.5, 1.0, 2.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.ice_cube.data = np.zeros_like(self.ice_cube.data)
        self.ice_cube.rename("probability_of_vertical_integral_of_ice")

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin.process(self.fg_cube, self.ice_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cubes."""
        plugin = Plugin()
        cube_a = self.fg_cube.copy()
        cube_b = self.ice_cube.copy()
        plugin.process(cube_a, cube_b)
        self.assertArrayAlmostEqual(cube_a.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.ice_cube.data)

    def test_ice_null(self):
        """Test that small VII probs do not increase lightning risk"""
        self.ice_cube.data[:, 7, 7] = 0.
        self.ice_cube.data[0, 7, 7:9] = 0.5
        self.fg_cube.data[0, 7, 7] = 0.25
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.25
        result = plugin.process(self.fg_cube,
                                self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_zero(self):
        """Test that zero VII probs do not increase lightning risk"""
        self.ice_cube.data[:, 7, 7] = 0.
        self.fg_cube.data[0, 7, 7] = 0.
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.
        result = plugin.process(self.fg_cube,
                                self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_small(self):
        """Test that small VII probs do increase lightning risk"""
        self.ice_cube.data[:, 7, 7] = 0.
        self.ice_cube.data[0, 7, 7] = 0.5
        self.fg_cube.data[0, 7, 7] = 0.
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.05
        result = plugin.process(self.fg_cube,
                                self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_large(self):
        """Test that large VII probs do increase lightning risk"""
        self.ice_cube.data[:, 7, 7] = 1.
        self.fg_cube.data[0, 7, 7] = 0.
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.9
        result = plugin.process(self.fg_cube,
                                self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_large_long_fc(self):
        """Test that large VII probs do not increase lightning risk when
        forecast lead time is large"""
        self.ice_cube.data[:, 7, 7] = 1.
        self.fg_cube.data[0, 7, 7] = 0.
        self.fg_cube.coord('forecast_period').points = [3.]
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.0
        result = plugin.process(self.fg_cube,
                                self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


if __name__ == '__main__':
    unittest.main()

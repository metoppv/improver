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
"""Unit tests for the nowcast.convection.handle_precip.ApplyPrecip plugin."""

import unittest
from iris.tests import IrisTest
from iris.cube import Cube

import cf_units
from improver.nowcast.convection.handle_precip import ApplyPrecip as Plugin
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_with_no_realizations)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(Plugin())
        msg = ("""<ApplyPrecip:
 precipitation mapping:
   upper:  precip probability {precu} => max lightning prob {lprecu}
   middle: precip probability {precm} => max lightning prob {lprecm}
   lower:  precip probability {precl} => max lightning prob {lprecl}

   heavy:  prob(precip>7mm/hr)  {pphvy} => min lightning prob {lprobl}
   intense:prob(precip>35mm/hr) {ppint} => min lightning prob {lprobu}
>""".format(lprobu=1., lprobl=0.25,
            precu=0.1, precm=0.05, precl=0.0,
            lprecu=1., lprecm=0.2, lprecl=0.0067,
            pphvy=0.4, ppint=0.2))
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the nowcast convection handle_vii ApplyIce plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]))
        self.fg_cube.rename("probability_of_lightning")
        self.fg_cube.coord('forecast_period').points = [0.]
        self.precip_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3), fp_point=0.0))
        threshold_coord = self.precip_cube.coord('realization')
        threshold_coord.points = [0.5, 7.0, 35.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.precip_cube.rename("probability_of_precipitation")
        self.precip_cube.attributes.update({'relative_to_threshold': 'above'})
        self.precip_cube.data[1:, 0, ...] = 0.
        self.precip_cube.coord('forecast_period').points = [4.]

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cubes."""
        plugin = Plugin()
        cube_a = self.fg_cube.copy()
        cube_b = self.precip_cube.copy()
        plugin.process(cube_a, cube_b)
        self.assertArrayAlmostEqual(cube_a.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.precip_cube.data)

    def test_precip_zero(self):
        """Test that zero precip probs reduce lightning risk"""
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.0067
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_small(self):
        """Test that small precip probs reduce lightning risk"""
        self.precip_cube.data[:, 0, 7, 7] = 0.
        self.precip_cube.data[0, 0, 7, 7] = 0.075
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.6
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_heavy(self):
        """Test that prob of heavy precip increases lightning risk"""
        self.precip_cube.data[0, 0, 7, 7] = 1.0
        self.precip_cube.data[1, 0, 7, 7] = 0.5
        # Set first-guess to zero
        self.fg_cube.data[0, 7, 7] = 0.0
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.25
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_heavy_null(self):
        """Test that low prob of heavy precip does not increase
        lightning risk"""
        self.precip_cube.data[0, 0, 7, 7] = 1.0
        self.precip_cube.data[1, 0, 7, 7] = 0.3
        # Set first-guess to zero
        self.fg_cube.data[0, 7, 7] = 0.1
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.1
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_intense(self):
        """Test that prob of intense precip increases lightning risk"""
        self.precip_cube.data[0, 0, 7, 7] = 1.0
        self.precip_cube.data[1, 0, 7, 7] = 1.0
        self.precip_cube.data[2, 0, 7, 7] = 0.5
        # Set first-guess to zero
        self.fg_cube.data[0, 7, 7] = 0.0
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 1.0
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_intense_null(self):
        """Test that low prob of intense precip does not increase
        lightning risk"""
        self.precip_cube.data[0, 0, 7, 7] = 1.0
        self.precip_cube.data[1, 0, 7, 7] = 1.0
        self.precip_cube.data[2, 0, 7, 7] = 0.1
        # Set first-guess to zero
        self.fg_cube.data[0, 7, 7] = 0.1
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.25  # Heavy-precip result only
        result = plugin.process(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


if __name__ == '__main__':
    unittest.main()

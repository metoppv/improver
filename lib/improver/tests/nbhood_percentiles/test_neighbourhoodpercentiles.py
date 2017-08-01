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
"""Unit tests for the nbhood.NeighbourhoodProcessing plugin."""


import unittest

import iris
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np


from improver.nbhood_percentiles import NeighbourhoodPercentiles as NBHood
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period
from improver.tests.nbhood.test_neighbourhoodprocessing import (
    set_up_cube, set_up_cube_with_no_realizations)
from improver.percentile import PercentileConverter


class Test__init__(IrisTest):

    """Test the __init__ method of NeighbourhoodProcessing"""

    def test_radii_varying_with_lead_time_mismatch(self):
        """
        Test that the desired error message is raised, if there is a mismatch
        between the number of radii and the number of lead times.
        """
        radii = [10000, 20000, 30000]
        lead_times = [2, 3]
        msg = "There is a mismatch in the number of radii"
        with self.assertRaisesRegexp(ValueError, msg):
            method = 'circular'
            NBHood(method, radii, lead_times=lead_times)

    def test_method_does_not_exist(self):
        """
        Test that desired error message is raised, if the neighbourhood method
        does not exist.
        """
        method = 'nonsense'
        radii = 10000
        msg = 'The method requested: '
        with self.assertRaisesRegexp(KeyError, msg):
            NBHood(method, radii)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(NBHood("circular", 10000))
        msg = ('<NeighbourhoodPercentiles: method: circular; '
               'radii: 10000.0; lead_times: None; '
               'ens_factor: 1.0; percentile-count: {}>'.format(
                   len(PercentileConverter.DEFAULT_PERCENTILES)))
        self.assertEqual(result, msg)


class Test__find_radii(IrisTest):

    """Test the internal _find_radii function is working correctly."""

    def test_basic_float_cube_lead_times_is_none(self):
        """Test _find_radii returns a float with the correct value."""
        method = "circular"
        ens_factor = 0.8
        num_ens = 2.0
        radius = 6300
        plugin = NBHood(method,
                        radius,
                        ens_factor=ens_factor)
        result = plugin._find_radii(num_ens)
        expected_result = 3563.8181771801998
        self.assertIsInstance(result, float)
        self.assertAlmostEquals(result, expected_result)

    def test_basic_array_cube_lead_times_an_array(self):
        """Test _find_radii returns an array with the correct values."""
        method = "circular"
        ens_factor = 0.9
        num_ens = 2.0
        fp_points = np.array([2, 3, 4])
        radii = [10000, 20000, 30000]
        lead_times = [2, 3, 4]
        plugin = NBHood(method,
                        radii,
                        lead_times=lead_times,
                        ens_factor=ens_factor)
        result = plugin._find_radii(num_ens,
                                    cube_lead_times=fp_points)
        expected_result = np.array([6363.961031, 12727.922061, 19091.883092])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolation(self):
        """Test that interpolation is working as expected in _find_radii."""
        fp_points = np.array([2, 3, 4])
        method = "circular"
        ens_factor = 0.8
        num_ens = 4.0
        fp_points = np.array([2, 3, 4])
        radii = [10000, 30000]
        lead_times = [2, 4]
        plugin = NBHood(method,
                        radii,
                        lead_times=lead_times,
                        ens_factor=ens_factor)
        result = plugin._find_radii(num_ens,
                                    cube_lead_times=fp_points)
        expected_result = np.array([4000., 8000., 12000.])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_process(IrisTest):

    """Tests for the process method of NeighbourhoodProcessing."""

    RADIUS = 6300  # Gives 3 grid cells worth.

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        cube = set_up_cube()
        method = "circular"
        result = NBHood(method, self.RADIUS).process(cube)
        self.assertIsInstance(result, Cube)

    def test_single_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        cube = set_up_cube()
        cube.data[0][0][6][7] = np.NAN
        msg = "NaN detected in input cube data"
        with self.assertRaisesRegexp(ValueError, msg):
            method = "circular"
            NBHood(method, self.RADIUS).process(cube)

    def test_realizations_and_source_realizations_fails(self):
        """Raises error if realizations and source realizations both set."""
        cube = set_up_cube()
        cube.attributes.update({'source_realizations': [0, 1, 2, 3]})
        msg = ('Realizations and attribute source_realizations should not'
               ' both be set')
        with self.assertRaisesRegexp(ValueError, msg):
            method = "circular"
            NBHood(method, self.RADIUS).process(cube)

    def test_multiple_realizations(self):
        """Test when the cube has a realization dimension that same coord is returned."""
        cube = set_up_cube(num_realization_points=4)
        radii = 15000
        method = "circular"
        ens_factor = 0.8
        result = NBHood(method, radii,
                        ens_factor=ens_factor).process(cube)
        self.assertIsInstance(result, Cube)
        expected = cube.coord('realization').points
        self.assertArrayEqual(result.coord('realization').points, expected)

    def test_multiple_realizations_and_times(self):
        """Test when the cube has a realization and time dimension that both are returned."""
        cube = set_up_cube(num_time_points=3,
                           num_realization_points=4)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii = [15000, 15000, 15000]
        lead_times = [2, 3, 4]
        method = "circular"
        ens_factor = 0.8
        result = NBHood(method, radii,
                        lead_times=lead_times,
                        ens_factor=ens_factor).process(cube)
        self.assertIsInstance(result, Cube)
        expected = cube.coord('realization').points
        self.assertArrayEqual(result.coord('realization').points, expected)
        expected = cube.coord('time').points
        self.assertArrayEqual(result.coord('time').points, expected)

    def test_returns_percentiles_coord(self):
        """Test the expected percentiles coord exists."""
        cube = set_up_cube_with_no_realizations()
        radii = 6000
        method = "circular"
        result = NBHood(method, radii).process(cube)
        self.assertIsInstance(result.coord('percentiles'), iris.coords.Coord)
        self.assertArrayEqual(result.coord('percentiles').points,
                              PercentileConverter.DEFAULT_PERCENTILES)

    def test_no_realizations(self):
        """Test when the array has no realization coord."""
        cube = set_up_cube_with_no_realizations()
        radii = 6000
        method = "circular"
        result = NBHood(method, radii).process(cube)
        self.assertIsInstance(result, Cube)

    def test_source_realizations(self):
        """Test when the array has source_realization attribute."""
        member_list = [0, 1, 2, 3]
        cube = (
            set_up_cube_with_no_realizations(source_realizations=member_list))
        radii = 15000
        ens_factor = 0.8
        method = "circular"
        plugin = NBHood(method, radii,
                        ens_factor=ens_factor)
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time(self):
        """
        Test that a cube is returned when the radius varies with lead time.
        """
        cube = set_up_cube(num_time_points=3)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii = [10000, 20000, 30000]
        lead_times = [2, 3, 4]
        method = "circular"
        plugin = NBHood(method, radii, lead_times)
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time_with_interpolation(self):
        """
        Test that a cube is returned for the following conditions:
        1. The radius varies with lead time.
        2. Linear interpolation is required to create values for the radii
        which are required but were not specified within the 'radii'
        argument.
        """
        cube = set_up_cube(num_time_points=3)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii = [10000, 30000]
        lead_times = [2, 4]
        method = "circular"
        plugin = NBHood(method, radii, lead_times)
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()

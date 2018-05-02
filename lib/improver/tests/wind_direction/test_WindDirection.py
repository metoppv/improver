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
"""Unit tests for the wind_direction.WindDirection plugin."""

import unittest
import numpy as np

import iris
from iris.tests import IrisTest
from iris.cube import Cube
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError
from cf_units import Unit

from improver.wind_direction import WindDirection
from improver.utilities.warnings_handler import ManageWarnings


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WindDirection())
        msg = ('<WindDirection>')
        self.assertEqual(result, msg)


# The two degree/complex arrays are directly equivalent.
degree_angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                          130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
                          230, 240, 250, 260, 270, 280, 290, 300, 310, 320,
                          330, 340, 350])

complex_angles = np.array([1.0+0j, 0.984807753+0.173648178j,
                           0.939692621+0.342020143j, 0.866025404+0.5j,
                           0.766044443+0.642787610j,
                           0.642787610+0.766044443j,
                           0.5+0.866025404j, 0.342020143+0.939692621j,
                           0.173648178+0.984807753j, 0.0+1.0j,
                           -0.173648178+0.984807753j,
                           -0.342020143+0.939692621j,
                           -0.5+0.866025404j, -0.642787610+0.766044443j,
                           -0.766044443+0.642787610j, -0.866025404+0.5j,
                           -0.939692621+0.342020143j,
                           -0.984807753+0.173648178j,
                           -1.0 + 0.0j, -0.984807753-0.173648178j,
                           -0.939692621-0.342020143j, -0.866025404-0.5j,
                           -0.766044443-0.642787610j,
                           -0.642787610-0.766044443j,
                           -0.5-0.866025404j, -0.342020143-0.939692621j,
                           -0.173648178-0.984807753j, -0.0-1.0j,
                           0.173648178-0.984807753j, 0.342020143-0.939692621j,
                           0.5-0.866025404j, 0.642787610-0.766044443j,
                           0.766044443-0.642787610j, 0.866025404-0.5j,
                           0.939692621-0.342020143j, 0.984807753-0.173648178j])


# Test the complex number handling functions.
class Test_deg_to_complex(IrisTest):
    """Test the deg_to_complex function."""

    def test_converts_single(self):
        """Tests that degree angle value is converted to complex."""
        expected_out = 0.707106781187+0.707106781187j
        result = WindDirection().deg_to_complex(45.0)
        self.assertAlmostEqual(result, expected_out)

    def test_handles_angle_wrap(self):
        """Test that code correctly handles 360 and 0 degrees."""
        expected_out = 1+0j
        result = WindDirection().deg_to_complex(0)
        self.assertAlmostEqual(result, expected_out)

        expected_out = 1-0j
        result = WindDirection().deg_to_complex(360)
        self.assertAlmostEqual(result, expected_out)

    def test_converts_array(self):
        """Tests that array of floats is converted to complex array."""
        result = WindDirection().deg_to_complex(degree_angles)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, complex_angles)


class Test_complex_to_deg(IrisTest):
    """Test the complex_to_deg function."""

    def test_converts_single(self):
        """Tests that complex value is converted to degrees."""
        expected_out = 270.0
        result = WindDirection().complex_to_deg(0-1j)
        self.assertEqual(result, expected_out)

    def test_handles_angle_wrap(self):
        """Test that code correctly handles 360 and 0 degrees."""
        expected_out = 0.0
        result = WindDirection().complex_to_deg(1+0j)  # Complex for 0 deg.
        self.assertAlmostEqual(result, expected_out)

        expected_out = 0.0
        result = WindDirection().complex_to_deg(1-0j)  # Complex for 360 deg.
        self.assertAlmostEqual(result, expected_out)

    def test_converts_array(self):
        """Tests that array of complex values are converted to degrees."""
        result = WindDirection().complex_to_deg(complex_angles)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, degree_angles)


class Test_find_r_values(IrisTest):
    """Test the find_r_values function."""

    def test_converts_single(self):
        """Tests that r-value correctly extracted from complex value."""
        expected_out = 2.0
        complex_in = 1.4142135624+1.4142135624j  # Complex for angle=45 and r=2
        result = WindDirection().find_r_values(complex_in)
        self.assertAlmostEqual(result, expected_out)

    def test_converts_array(self):
        """Test that code can find r-values from array of complex numbers."""
        expected_out = np.ones(complex_angles.shape, dtype=np.float32)
        result = WindDirection().find_r_values(complex_angles)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_out)


# Test the ensemble averaging codes.
# First element - two angles set at 90 and 270 degrees.
wind_dir_deg = np.array([[[90.0, 50.0],
                          [270.0, 350.0]],
                         [[270.0, 60.0],
                          [290.0, 10.0]]])

wind_dir_deg_mean = np.array([[180.0, 55.0],
                              [280.0, 0.0]])

wind_dir_complex = np.array([
                          [[6.12323400e-17+1.0j, 0.642787610+0.76604444j],
                           [-1.83697020e-16-1.0j, 0.984807753-0.17364818j]],
                          [[-1.83697020e-16-1.0j, 0.5+0.8660254j],
                           [0.342020143-0.93969262j, 0.984807753+0.17364818j]]
                             ])

wind_dir_complex_mean = np.array([
                         [-6.12323400e-17+0j, 0.571393805+0.816034923j],
                         [0.171010072-0.969846310j, 0.984807753-0j]
                                 ])


class Test_calc_polar_mean_std_dev(IrisTest):
    """Test the calc_avg_dist_mean function returns confidence values."""

    def test_returns_confidence(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless. This code calculates the standard deviation about
        mean point."""

        expected_out = np.array([[0.29289322, 0.95638061],
                                 [0.91284426,  0.91284426]])

        result = WindDirection().calc_polar_mean_std_dev(wind_dir_complex,
                                                         wind_dir_deg_mean)

        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_out)


class Test_wind_dir_decider(IrisTest):

    def test_runs_function(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless with an r value of nearly zero. So the code subistites the
        wind direction taken from the first ensemble value in its place."""

        expected_out = np.array([[90.0, 55.0],
                                 [280.0, 0.0]])

        result = WindDirection.wind_dir_decider(wind_dir_deg,
                                                wind_dir_deg_mean,
                                                wind_dir_complex_mean)

        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_out)


class Test_process(IrisTest):

    def setUp(self):
        """Create a cube with collapsable coordinates."""

        # 5x3x4 3D Array containing wind direction in angles.
        data = np.array([[[[170.0, 50.0, 90.0, 90.0],
                           [170.0, 170.0, 47.0, 350.0],
                           [10.0, 309.0, 10.0, 10.0]]],
                         [[[170.0, 50.0, 90.0, 90.0],
                           [170.0, 170.0, 47.0, 47.0],
                           [10.0, 10.0, 10.0, 10.0]]],
                         [[[10.0, 50.0, 90.0, 90.0],
                           [170.0, 170.0, 47.0, 47.0],
                           [310.0, 309.0, 10.0, 10.0]]],
                         [[[190.0, 40.0, 270.0, 90.0],
                           [170.0, 170.0, 47.0, 47.0],
                           [310.0, 309.0, 10.0, 10.0]]],
                         [[[190.0, 40.0, 270.0, 270.0],
                           [170.0, 170.0, 47.0, 47.0],
                           [310.0, 309.0, 10.0, 10.0]]]])

        realization = DimCoord([0, 1, 2, 3, 4], 'realization', units=1)
        time = DimCoord([402192.5], standard_name='time',
                        units=Unit('hours since 1970-01-01 00:00:00',
                                   calendar='gregorian'))
        latitude = DimCoord(np.linspace(-90, 90, 3),
                            standard_name='latitude', units='degrees')
        longitude = DimCoord(np.linspace(-180, 180, 4),
                             standard_name='longitude', units='degrees')

        cube = Cube(data, standard_name="wind_from_direction",
                    dim_coords_and_dims=[(realization, 0),
                                         (time, 1),
                                         (latitude, 2),
                                         (longitude, 3)],
                                        units="degree")

        self.cube = cube

    def test_basic(self):
        """Test that the plugin returns expected data types. """
        result_cube, r_vals, std_dev = WindDirection().process(self.cube)
        self.assertIsInstance(result_cube, Cube)
        self.assertIsInstance(r_vals, np.ndarray)
        self.assertIsInstance(std_dev, np.ndarray)

    def test_fails_if_data_is_not_cube(self):
        """Test code raises a Type Error if input cube is not a cube."""
        input_data = 50.0
        msg = ('Wind direction input is not a cube, but'
               ' {0:s}'.format(type(input_data)))
        with self.assertRaisesRegexp(TypeError, msg):
            result = WindDirection().process(input_data)

    def test_return_single_precision(self):
        """Test that the function returns data of float32."""

        result_cube, r_vals, std_dev = WindDirection().process(self.cube)
        result = result_cube.data
        result = result[0]
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(r_vals.dtype, np.float32)
        self.assertEqual(std_dev.dtype, np.float32)

    def test_returns_expected_values(self):
        """Test that the function returns correct 2D arrays of floats. """

        expected_wind_mean = np.array([[176.63627625, 46.00244522, 90.0, 90.0],
                                       [170.0, 170.0, 47.0, 36.54423141],
                                       [333.41320801, 320.03521729, 10.0,
                                        10.0]])

        expected_r_vals = np.array([[0.5919044, 0.99634719, 0.2, 0.6],
                                    [1.0, 1.0, 1.0, 0.92427504],
                                    [0.87177974, 0.91385943, 1.0, 1.0]])

        expected_std_dev = np.array([[0.73166388, 0.95813018, 0.6, 0.8],
                                     [1.0, 1.0, 1.0, 0.84808648],
                                     [0.75270665, 0.83861077, 1.0, 1.0]])

        result_cube, r_vals, std_dev = WindDirection().process(self.cube)
        result = result_cube.data
        wind_mean = result[0]  # Extracts first element (time) from list.

        self.assertIsInstance(result, np.ndarray)
        self.assertIsInstance(r_vals, np.ndarray)
        self.assertIsInstance(std_dev, np.ndarray)
        self.assertArrayAlmostEqual(wind_mean, expected_wind_mean)
        self.assertArrayAlmostEqual(r_vals, expected_r_vals)
        self.assertArrayAlmostEqual(std_dev, expected_std_dev)

if __name__ == '__main__':
    unittest.main()

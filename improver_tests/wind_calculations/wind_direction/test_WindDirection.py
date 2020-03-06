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
"""Unit tests for the wind_direction.WindDirection plugin."""

import unittest

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.wind_calculations.wind_direction import WindDirection

from ...calibration.ensemble_calibration.helper_functions import (
    set_up_temperature_cube)
from ...nbhood.nbhood.test_BaseNeighbourhoodProcessing import set_up_cube
from ...set_up_test_cubes import set_up_variable_cube


# Data to test complex/degree handling functions.
# Complex angles equivalent to np.arange(0., 360, 10) degrees.
COMPLEX_ANGLES = np.array([1.0+0j, 0.984807753+0.173648178j,
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

# Data to test the ensemble averaging codes.
WIND_DIR_COMPLEX = np.array([[[6.12323400e-17+1.0j, 0.642787610+0.76604444j],
                              [-1.83697020e-16-1.0j, 0.984807753-0.17364818j]],
                             [[-1.83697020e-16-1.0j, 0.5+0.8660254j],
                              [0.342020143-0.93969262j,
                               0.984807753+0.17364818j]]])


def make_wdir_cube_222():
    """Make a wind direction cube for testing this plugin"""
    # 2x2x2 3D Array containing wind direction in angles.
    # First element - two angles set at 90 and 270 degrees.
    data = np.array([[[90.0, 50.0],
                      [270.0, 350.0]],
                     [[270.0, 60.0],
                      [290.0, 10.0]]])
    cube = set_up_cube(num_grid_points=2,
                       num_realization_points=2,
                       zero_point_indices=[[0, 0, 0, 0]])
    cube = cube[:, 0, :, :]  # Demotes time dimension.

    cube.data = data
    cube.units = Unit('degrees')
    return cube


def make_wdir_cube_534():
    """Make a wind direction cube for testing this plugin"""
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
                       [310.0, 309.0, 10.0, 10.0]]]], dtype=np.float32)

    cube = set_up_cube(num_grid_points=4,
                       num_realization_points=5,
                       zero_point_indices=[[0, 0, 0, 0]])
    cube = cube[:, :, 0:-1, :]  # (reduce y from 4 to 3)
    cube.data = data
    cube.rename("wind_from_direction")
    cube.units = Unit('degrees')
    return cube


def pad_wdir_cube_222():
    """Make a padded wind direction cube for testing this plugin"""
    # 2x2x2 3D Array containing wind direction in angles.
    # Padded in x and y to 2x10x10 for use with nbhood option
    data = np.array([[[90.0, 50.0],
                      [270.0, 350.0]],
                     [[270.0, 60.0],
                      [290.0, 10.0]]])
    cube = set_up_cube(num_grid_points=10,
                       num_realization_points=2,
                       zero_point_indices=[[0, 0, 0, 0]])
    cube = cube[:, 0, :, :]  # Demotes time dimension.
    cube.data = np.pad(data,
                       ((0, 0), (4, 4), (4, 4)),
                       "constant",
                       constant_values=(0.0, 0.0))
    cube.rename("wind_from_direction")
    cube.units = Unit('degrees')
    return cube


class Test__init__(IrisTest):
    """Test the init method."""

    def test_basic(self):
        """Test that the __init__ does not fail."""
        result = WindDirection()
        self.assertIsInstance(result, WindDirection)

    def test_backup_method(self):
        """Test that the __init__ accepts this keyword."""
        result = WindDirection(backup_method='neighbourhood')
        self.assertIsInstance(result, WindDirection)

    def test_invalid_method(self):
        """Test that the __init__ fails when an unrecognised option is given"""
        msg = ('Invalid option for keyword backup_method ')
        with self.assertRaisesRegex(ValueError, msg):
            WindDirection(backup_method='invalid')


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WindDirection())
        msg = ('<WindDirection: backup_method "neighbourhood"; neighbourhood '
               'radius "6000.0"m>')
        self.assertEqual(result, msg)


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
        result = WindDirection().deg_to_complex(np.arange(0., 360, 10))
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, COMPLEX_ANGLES)


class Test_complex_to_deg(IrisTest):
    """Test the complex_to_deg function."""

    def test_fails_if_data_is_not_array(self):
        """Test code raises a Type Error if input data not an array."""
        input_data = 0-1j
        msg = ('Input data is not a numpy array, but'
               ' {}'.format(type(input_data)))
        with self.assertRaisesRegex(TypeError, msg):
            WindDirection().complex_to_deg(input_data)

    def test_handles_angle_wrap(self):
        """Test that code correctly handles 360 and 0 degrees."""
        # Input is complex for 0 and 360 deg - both should return 0.0.
        input_data = np.array([1+0j, 1-0j])
        result = WindDirection().complex_to_deg(input_data)
        self.assertTrue((result == 0.0).all())

    def test_converts_array(self):
        """Tests that array of complex values are converted to degrees."""
        result = WindDirection().complex_to_deg(COMPLEX_ANGLES)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, np.arange(0., 360, 10))


class Test_complex_to_deg_roundtrip(IrisTest):
    """Test the complex_to_deg and deg_to_complex functions together."""

    def setUp(self):
        """Initialise plugin and supply data for tests"""
        self.plugin = WindDirection()
        self.cube = make_wdir_cube_534()

    def test_from_deg(self):
        """Tests that array of values are converted to complex and back."""
        tmp_complex = self.plugin.deg_to_complex(self.cube.data)
        result = self.plugin.complex_to_deg(tmp_complex)
        self.assertArrayAlmostEqual(result, self.cube.data, decimal=4)

    def test_from_complex(self):
        """Tests that array of values are converted to degrees and back."""
        tmp_degrees = self.plugin.complex_to_deg(COMPLEX_ANGLES)
        result = self.plugin.deg_to_complex(tmp_degrees)
        self.assertArrayAlmostEqual(result, COMPLEX_ANGLES)


class Test_calc_wind_dir_mean(IrisTest):
    """Test the calc_wind_dir_mean function."""

    def setUp(self):
        """Initialise plugin and supply data for tests"""
        self.plugin = WindDirection()
        # 5x3x4 3D Array containing wind direction in angles.
        cube = make_wdir_cube_534()
        self.plugin.wdir_complex = self.plugin.deg_to_complex(
            cube.data)
        self.plugin.wdir_slice_mean = (
            next(cube.slices_over("realization")))
        self.plugin.realization_axis = 0

        self.expected_wind_mean = (
            np.array([[[176.636276, 46.002445, 90.0, 90.0],
                       [170.0, 170.0, 47.0, 36.544231],
                       [333.413239, 320.035217, 10.0, 10.0]]]))

    def test_complex(self):
        """Test that the function defines correct complex mean."""
        self.plugin.calc_wind_dir_mean()
        result = self.plugin.wdir_mean_complex
        expected_complex = (
            self.plugin.deg_to_complex(self.expected_wind_mean,
                                       radius=np.absolute(result)))
        self.assertArrayAlmostEqual(result, expected_complex)

    def test_degrees(self):
        """Test that the function defines correct degrees cube."""
        self.plugin.calc_wind_dir_mean()
        result = self.plugin.wdir_slice_mean
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.data, np.ndarray)
        self.assertArrayAlmostEqual(result.data, self.expected_wind_mean,
                                    decimal=4)


class Test_find_r_values(IrisTest):
    """Test the find_r_values function."""

    def setUp(self):
        """Initialise plugin and supply data for tests"""
        self.plugin = WindDirection()

    def test_converts_single(self):
        """Tests that r-value is correctly extracted from complex value."""
        # Attach a cube for the plugin to copy in creating the resulting cube:
        self.plugin.wdir_slice_mean = make_wdir_cube_222()[0][0][0]
        expected_out = 2.0
        # Set-up complex values for angle=45 and r=2
        self.plugin.wdir_mean_complex = 1.4142135624+1.4142135624j
        self.plugin.find_r_values()
        self.assertAlmostEqual(self.plugin.r_vals_slice.data, expected_out)

    def test_converts_array(self):
        """Test that code can find r-values from array of complex numbers."""
        longitude = DimCoord(np.linspace(-180, 180, 36),
                             standard_name='longitude', units='degrees')

        cube = Cube(COMPLEX_ANGLES, standard_name="wind_from_direction",
                    dim_coords_and_dims=[(longitude, 0)],
                    units="degree")
        # Attach a cube for the plugin to copy in creating the resulting cube:
        self.plugin.wdir_slice_mean = cube
        self.plugin.wdir_mean_complex = COMPLEX_ANGLES
        expected_out = np.ones(COMPLEX_ANGLES.shape, dtype=np.float32)
        self.plugin.find_r_values()
        result = self.plugin.r_vals_slice.data
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_out)


class Test_calc_confidence_measure(IrisTest):
    """Test the calc_avg_dist_mean function returns confidence values."""

    def setUp(self):
        """Initialise plugin and supply data for tests"""
        self.plugin = WindDirection()
        self.plugin.wdir_complex = WIND_DIR_COMPLEX
        self.plugin.realization_axis = 0
        self.plugin.r_vals_slice = make_wdir_cube_222()[0]
        self.plugin.r_vals_slice.data = (
            np.array([[6.12323400e-17, 0.996194698],
                      [0.984807753, 0.984807753]]))
        self.plugin.wdir_slice_mean = make_wdir_cube_222()[0]
        self.plugin.wdir_slice_mean.data = np.array([[180.0, 55.0],
                                                     [280.0, 0.0]])

    def test_returns_confidence(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless. This code calculates a confidence measure based on how
        far the individual ensemble realizationss are away from
        the mean point."""
        expected_out = np.array([[0.0, 0.95638061],
                                 [0.91284426, 0.91284426]])
        self.plugin.calc_confidence_measure()
        result = self.plugin.confidence_slice.data

        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_out)


class Test_wind_dir_decider(IrisTest):
    """Test the wind_dir_decider function."""

    def test_runs_function_1st_member(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless with an r value of nearly zero. So the code substitutes the
        wind direction taken from the first ensemble value in its place."""
        cube = make_wdir_cube_222()
        self.plugin = WindDirection(backup_method="first_realization")
        self.plugin.wdir_complex = WIND_DIR_COMPLEX
        self.plugin.realization_axis = 0
        self.plugin.wdir_slice_mean = make_wdir_cube_222()[0]
        self.plugin.wdir_slice_mean.data = np.array([[180.0, 55.0],
                                                     [280.0, 0.0]])
        self.plugin.wdir_mean_complex = (
            self.plugin.deg_to_complex(self.plugin.wdir_slice_mean.data))
        expected_out = np.array([[90.0, 55.0],
                                 [280.0, 0.0]])
        where_low_r = np.array([[True, False],
                                [False, False]])
        self.plugin.wind_dir_decider(where_low_r, cube)
        result = self.plugin.wdir_slice_mean.data

        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_out)

    def test_runs_function_nbhood(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless with an r value of nearly zero. So the code substitutes the
        wind direction taken using the neighbourhood method."""
        expected_out = np.array([[354.91, 55.],
                                 [280., 0.]])

        cube = pad_wdir_cube_222()
        where_low_r = np.pad(
            np.array([[True, False],
                      [False, False]]),
            ((4, 4), (4, 4)), "constant", constant_values=(True, True))

        wind_dir_deg_mean = np.array([[180.0, 55.0],
                                      [280.0, 0.0]])

        self.plugin = WindDirection(backup_method="neighbourhood")
        self.plugin.realization_axis = 0
        self.plugin.n_realizations = 1
        self.plugin.wdir_mean_complex = np.pad(
            self.plugin.deg_to_complex(wind_dir_deg_mean),
            ((4, 4), (4, 4)),
            "constant",
            constant_values=(0.0, 0.0))
        self.plugin.wdir_complex = np.pad(WIND_DIR_COMPLEX,
                                          ((0, 0), (4, 4), (4, 4)),
                                          "constant",
                                          constant_values=(0.0 + 0.0j))
        self.plugin.wdir_slice_mean = pad_wdir_cube_222()[0]
        self.plugin.wdir_slice_mean.data = np.pad(wind_dir_deg_mean,
                                                  ((4, 4), (4, 4)),
                                                  "constant",
                                                  constant_values=0.0)

        self.plugin.wind_dir_decider(where_low_r, cube)
        result = self.plugin.wdir_slice_mean.data
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result[4:6, 4:6], expected_out, decimal=2)


class Test_process(IrisTest):
    """Test entire code handles a cube correctly."""

    def setUp(self):
        """Create a cube with collapsable coordinates."""
        self.cube = make_wdir_cube_534()

        self.expected_wind_mean = (
            np.array([[[176.63627625, 46.00244522, 90.0, 90.0],
                       [170.0, 170.0, 47.0, 36.54423141],
                       [333.41320801, 320.03521729, 10.0, 10.0]]],
                     dtype=np.float32))

        self.expected_r_vals = np.array([[0.5919044, 0.99634719, 0.2, 0.6],
                                         [1.0, 1.0, 1.0, 0.92427504],
                                         [0.87177974, 0.91385943, 1.0, 1.0]],
                                        dtype=np.float32)

        self.expected_confidence_measure = (
            np.array([[0.73166388, 0.95813018, 0.6, 0.8],
                      [1.0, 1.0, 1.0, 0.84808648],
                      [0.75270665, 0.83861077, 1.0, 1.0]],
                     dtype=np.float32))

    def test_basic(self):
        """Test that the plugin returns expected data types. """
        result_cube, r_vals_cube, confidence_measure_cube = (
            WindDirection().process(self.cube))

        self.assertIsInstance(result_cube, Cube)
        self.assertIsInstance(r_vals_cube, Cube)
        self.assertIsInstance(confidence_measure_cube, Cube)

    def test_fails_if_data_is_not_cube(self):
        """Test code raises a Type Error if input cube is not a cube."""
        input_data = 50.0
        msg = ('Wind direction input is not a cube, but'
               ' {0}'.format(type(input_data)))
        with self.assertRaisesRegex(TypeError, msg):
            WindDirection().process(input_data)

    def test_fails_if_data_is_not_convertible_to_degrees(self):
        """Test code raises a ValueError if input cube is not convertible to
        degrees."""
        cube = set_up_temperature_cube()
        msg = 'Input cube cannot be converted to degrees'
        with self.assertRaisesRegex(ValueError, msg):
            WindDirection().process(cube)

    def test_return_single_precision(self):
        """Test that the function returns data of float32."""

        result_cube, r_vals_cube, confidence_measure_cube = (
            WindDirection().process(self.cube))

        self.assertEqual(result_cube.dtype, np.float32)
        self.assertEqual(r_vals_cube.dtype, np.float32)
        self.assertEqual(confidence_measure_cube.dtype, np.float32)

    def test_returns_expected_values(self):
        """Test that the function returns correct 2D arrays of floats. """

        result_cube, r_vals_cube, confidence_measure_cube = (
            WindDirection().process(self.cube))

        result = result_cube.data
        r_vals = r_vals_cube.data
        confidence_measure = confidence_measure_cube.data

        self.assertIsInstance(result, np.ndarray)
        self.assertIsInstance(r_vals, np.ndarray)
        self.assertIsInstance(confidence_measure, np.ndarray)
        self.assertArrayAlmostEqual(result, self.expected_wind_mean, decimal=4)
        self.assertArrayAlmostEqual(r_vals, self.expected_r_vals)
        self.assertArrayAlmostEqual(
            confidence_measure, self.expected_confidence_measure)

    def test_with_backup(self):
        """Test that wind_dir_decider is invoked to select a better value for
        a low-confidence point."""
        # create a low-confidence point
        self.cube.data[:, 0, 1, 1] = [0., 72., 144., 216., 288.]

        # set up a larger cube using a "neutral" pad value so that
        # neighbourhood processing does not fail
        data = np.full((5, 9, 10), 30., dtype=np.float32)
        data[:, 3:6, 3:7] = self.cube.data[:, 0, :, :].copy()

        cube = set_up_variable_cube(
            data, name='wind_from_direction', units='degrees',
            spatial_grid='equalarea')
        cube.coord(axis='x').points = np.arange(-50000., -31000., 2000.)
        cube.coord(axis='y').points = np.arange(0., 17000., 2000.)

        self.expected_wind_mean[0, 1, 1] = 30.0870
        self.expected_r_vals[1, 1] = 2.665601e-08
        self.expected_confidence_measure[1, 1] = 0.0

        result_cube, r_vals_cube, confidence_measure_cube = (
            WindDirection().process(cube))

        result = result_cube.data[3:6, 3:7]
        r_vals = r_vals_cube.data[3:6, 3:7]
        confidence_measure = confidence_measure_cube.data[3:6, 3:7]

        self.assertIsInstance(result, np.ndarray)
        self.assertIsInstance(r_vals, np.ndarray)
        self.assertIsInstance(confidence_measure, np.ndarray)
        self.assertArrayAlmostEqual(
            result, self.expected_wind_mean[0], decimal=4)
        self.assertArrayAlmostEqual(
            confidence_measure, self.expected_confidence_measure)
        self.assertArrayAlmostEqual(r_vals, self.expected_r_vals)


if __name__ == '__main__':
    unittest.main()

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
"""Unit tests for the convection.DiagnoseConvectivePrecipitation plugin."""


import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.convection import DiagnoseConvectivePrecipitation

# Fraction to convert from mm/hr to m/s.
# m/s are SI units, however, mm/hr values are easier to handle.
mm_hr_to_m_s = 2.7778e-7


def set_up_precipitation_rate_cube():
    """Create a cube with metadata and values suitable for
    precipitation rate."""
    data = np.zeros((1, 1, 4, 4))
    # Convert from mm/hr to m/s.
    data[0, 0, 0, :] = 2.0 * mm_hr_to_m_s
    data[0, 0, 1, :] = 4.0 * mm_hr_to_m_s
    data[0, 0, 2, :] = 8.0 * mm_hr_to_m_s
    data[0, 0, 3, :] = 16.0 * mm_hr_to_m_s
    data[0, 0, 0, 2] = 0.0
    data[0, 0, 2, 1] = 0.0
    data[0, 0, 3, 0] = 0.0
    return set_up_cube(data, "lwe_precipitation_rate", "m s-1")


def set_up_cube(data, phenomenon_standard_name, phenomenon_units,
                realizations=np.array([0]),
                timesteps=np.array([402192.5]),
                y_dimension_values=np.array([0., 2000., 4000., 6000.]),
                x_dimension_values=np.array([0., 2000., 4000., 6000.])):
    """Create a cube containing the required realizations, timesteps,
    y-dimension values and x-dimension values."""
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    cube.add_dim_coord(DimCoord(realizations, 'realization',
                                units='1'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(timesteps, "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(y_dimension_values,
                                'projection_y_coordinate', units='m'), 2)
    cube.add_dim_coord(DimCoord(x_dimension_values,
                                'projection_x_coordinate', units='m'), 3)
    return cube


def apply_threshold(cube, threshold):
    """Apply threshold and convert to binary, rather than logical values."""
    cube.data = cube.data > threshold
    cube.data = cube.data.astype(int)
    return cube


def lower_higher_threshold_cubelist(
        lower_cube, higher_cube, lower_threshold, higher_threshold):
    """Apply low and high thresholds and put into a cube list."""
    lower_cube = (
        apply_threshold(lower_cube, lower_threshold))
    higher_cube = (
        apply_threshold(higher_cube, higher_threshold))
    return iris.cube.CubeList([lower_cube, higher_cube])


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        lower_threshold = 0.001 * mm_hr_to_m_s
        higher_threshold = 5 * mm_hr_to_m_s
        neighbourhood_method = "square"
        radii = 2000.0
        result = str(DiagnoseConvectivePrecipitation(
            lower_threshold, higher_threshold,
            neighbourhood_method, radii))
        msg = ('<DiagnoseConvectivePrecipitation: lower_threshold 2.7778e-10; '
               'higher_threshold 1.3889e-06; neighbourhood_method: square; '
               'radii: 2000.0; fuzzy_factor None; comparison_operator: >; '
               'lead_times: None; weighted_mode: True;'
               'use_adjacent_grid_square_differences: True>')
        self.assertEqual(str(result), msg)


class Test__calculate_convective_ratio(IrisTest):

    """Test the _calculate_convective_ratio method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii = 2000.0
        self.cube = set_up_precipitation_rate_cube()
        self.lower_cube = self.cube.copy()
        self.higher_cube = self.cube.copy()
        self.cubelist = lower_higher_threshold_cubelist(
            self.lower_cube, self.higher_cube, self.lower_threshold,
            self.higher_threshold)
        self.threshold_list = [self.lower_threshold, self.higher_threshold]

    def test_basic(self):
        """Test a basic example using the default values for the keyword
        arguments. Make sure that the output is a cube with the expected
        data."""
        expected = np.array(
            [[[[0., 0., 0., 0.],
               [0.2, 0.28571429, 0.28571429, 0.4],
               [0.5, 0.57142857, 0.625, 0.66666667],
               [1., 1., 1., 1.]]]])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            self.radii)._calculate_convective_ratio(
                self.cubelist, self.threshold_list)
        self.assertArrayAlmostEqual(result, expected)

    def test_no_precipitation(self):
        """If there is no precipitation, then the convective ratio will try
        to do a 0/0 division, which will result in NaN values. Check that
        the output array works as intended."""
        expected = np.array(
            [[[[np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan]]]])
        data = np.zeros((1, 1, 4, 4))
        cube = set_up_cube(data, "lwe_precipitation_rate", "m s-1")
        cubelist = lower_higher_threshold_cubelist(
            cube.copy(), cube.copy(), self.lower_threshold,
            self.higher_threshold)
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            self.radii)._calculate_convective_ratio(
                cubelist, self.threshold_list)
        self.assertArrayAlmostEqual(result, expected)

    def test_catch_infinity_values(self):
        """Test an example where the infinity values are generated.
        Ensure these are caught as intended."""
        lower_threshold = 5 * mm_hr_to_m_s
        higher_threshold = 0.001 * mm_hr_to_m_s
        cubelist = lower_higher_threshold_cubelist(
            self.cube.copy(), self.cube.copy(), lower_threshold,
            higher_threshold)
        msg = "A value of infinity was found"
        with self.assertRaisesRegex(ValueError, msg):
            DiagnoseConvectivePrecipitation(
                self.lower_threshold, self.higher_threshold,
                self.neighbourhood_method,
                self.radii
                )._calculate_convective_ratio(cubelist, self.threshold_list)

    def test_catch_greater_than_1_values(self):
        """Test an example where the greater than 1 values are generated.
        Ensure these are caught as intended."""
        lower_threshold = 5 * mm_hr_to_m_s
        higher_threshold = 0.001 * mm_hr_to_m_s
        cubelist = lower_higher_threshold_cubelist(
            self.cube.copy(), self.cube.copy(), lower_threshold,
            higher_threshold)
        radii = 4000.0
        msg = "A value of greater than 1.0 was found"
        with self.assertRaisesRegex(ValueError, msg):
            DiagnoseConvectivePrecipitation(
                self.lower_threshold, self.higher_threshold,
                self.neighbourhood_method, radii,
                )._calculate_convective_ratio(cubelist, self.threshold_list)

    def test_multiple_lead_times_neighbourhooding(self):
        """Test where neighbourhood is applied for multiple lead times, where
        different radii are applied at each lead time."""
        expected = np.array(
            [[[[0.25, 0.166667, 0.166667, 0.],
               [0.166667, 0.11111111, 0.11111111, 0.],
               [0.166667, 0.11111111, 0.11111111, 0.],
               [0., 0., 0., 0.]],
              [[0.1111111, 0.0833333, 0.0833333, 0.1111111],
               [0.0833333, 0.0625, 0.0625, 0.0833333],
               [0.0833333, 0.0625, 0.0625, 0.0833333],
               [0.1111111, 0.0833333, 0.0833333, 0.1111111]]]])
        data = np.full((1, 2, 4, 4), 1.0 * mm_hr_to_m_s)
        data[0, 0, 1, 1] = 20.0 * mm_hr_to_m_s
        data[0, 1, 1, 1] = 20.0 * mm_hr_to_m_s
        lead_times = [3, 6]
        cube = set_up_cube(
            data, "lwe_precipitation_rate", "m s-1",
            timesteps=np.array([402192.5, 402195.5]))
        cube.add_aux_coord(AuxCoord(
            lead_times, "forecast_period", units="hours"), 1)
        radii = [2000.0, 4000.0]
        cubelist = lower_higher_threshold_cubelist(
            cube.copy(), cube.copy(), self.lower_threshold,
            self.higher_threshold)
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            radii, lead_times=lead_times
            )._calculate_convective_ratio(cubelist, self.threshold_list)
        self.assertArrayAlmostEqual(result, expected)

    def test_circular_neighbourhood(self):
        """Test a circular neighbourhood."""
        expected = np.array(
            [[[[0., 0., np.nan, 0.],
               [0., 0., 0., 0.],
               [1., np.nan, 1., 1.],
               [np.nan, 1., 1., 1.]]]])
        neighbourhood_method = "circular"
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            neighbourhood_method, self.radii
            )._calculate_convective_ratio(self.cubelist, self.threshold_list)
        self.assertArrayAlmostEqual(result, expected)

    def test_circular_neighbourhood_weighted_mode(self):
        """Test a circular neighbourhood with the weighted_mode
        set to True."""
        expected = np.array(
            [[[[0., 0., 0., 0.],
               [0.2, 0., 0.25, 0.2],
               [0.666667, 0.75, 0.75, 0.8],
               [1., 1., 1., 1.]]]])
        neighbourhood_method = "circular"
        weighted_mode = False
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            neighbourhood_method,
            self.radii, weighted_mode=weighted_mode
            )._calculate_convective_ratio(self.cubelist, self.threshold_list)
        self.assertArrayAlmostEqual(result, expected)


class Test_absolute_differences_between_adjacent_grid_squares(IrisTest):

    """Test the absolute_differences_between_adjacent_grid_squares method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii = 2000.0
        self.cube = set_up_precipitation_rate_cube()

    def test_basic(self):
        """Test that differences are calculated correctly between adjacent
        grid squares along x and y. Check that absolute values are returned."""
        expected_x = np.array(
            [[[[0.000000e+00, 5.555600e-07, 5.555600e-07],
               [0.000000e+00, 0.000000e+00, 0.000000e+00],
               [2.222240e-06, 2.222240e-06, 0.000000e+00],
               [4.444480e-06, 0.000000e+00, 0.000000e+00]]]])
        expected_y = np.array(
            [[[[5.555600e-07, 5.555600e-07, 1.111120e-06, 5.555600e-07],
               [1.111120e-06, 1.111120e-06, 1.111120e-06, 1.111120e-06],
               [2.222240e-06, 4.444480e-06, 2.222240e-06, 2.222240e-06]]]])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii
            ).absolute_differences_between_adjacent_grid_squares(
                self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, expected_x)
        self.assertArrayAlmostEqual(result[1].data, expected_y)


class Test_iterate_over_threshold(IrisTest):

    """Test the iterate_over_threshold method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii = 2000.0
        self.cube = set_up_precipitation_rate_cube()

    def test_basic(self):
        """Test an example for iterating over a list of thresholds."""
        expected = np.array(
            [[[[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [1., 0., 1., 1.],
               [0., 1., 1., 1.]]]])
        cubelist = iris.cube.CubeList([self.cube, self.cube])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            self.radii).iterate_over_threshold(cubelist, self.higher_threshold)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, expected)

    def test_fuzzy_factor(self):
        """Test an example where a fuzzy_factor is specified."""
        expected = np.array(
            [[[[0., 0., 0., 0.],
               [0.166667, 0.166667, 0.166667, 0.166667],
               [1., 0., 1., 1.],
               [0., 1., 1., 1.]]]])
        fuzzy_factor = 0.7
        cubelist = iris.cube.CubeList([self.cube, self.cube])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            self.radii, fuzzy_factor=fuzzy_factor
            ).iterate_over_threshold(cubelist, self.higher_threshold)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, expected)
        self.assertArrayAlmostEqual(result[1].data, expected)

    def test_below_threshold(self):
        """Test an example where the points below the specified threshold
        are regarded as significant."""
        expected = np.array(
            [[[[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [0., 1., 0., 0.],
               [1., 0., 0., 0.]]]])
        comparison_operator = '<='
        lower_threshold = 5 * mm_hr_to_m_s
        higher_threshold = 0.001 * mm_hr_to_m_s
        cubelist = iris.cube.CubeList([self.cube, self.cube])
        result = DiagnoseConvectivePrecipitation(
            lower_threshold, higher_threshold,
            self.neighbourhood_method,
            self.radii, comparison_operator=comparison_operator
            ).iterate_over_threshold(cubelist, self.higher_threshold)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, expected)
        self.assertArrayAlmostEqual(result[1].data, expected)


class Test_sum_differences_between_adjacent_grid_squares(IrisTest):

    """Test the sum_differences_between_adjacent_grid_squares method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii = 2000.0
        self.cube = set_up_precipitation_rate_cube()

    def test_basic(self):
        """Test that the sum of differences between adjacent grid squares,
        when accounting for the offset between the grid of the difference
        cube and the original grid is as expected."""
        expected = np.array(
            [[[[0., 2., 1., 0.],
               [2., 2., 0., 1.],
               [1., 2., 3., 3.],
               [1., 2., 1., 1.]]]])
        # Set up threshold_cube_x.
        threshold_cube_x_data = np.array(
            [[[[0., 1., 0.],
               [1., 0., 0.],
               [0., 1., 1.],
               [1., 0., 0.]]]])
        threshold_cube_x = self.cube.copy()
        threshold_cube_x = threshold_cube_x[:, :, :, :-1]
        threshold_cube_x.data = threshold_cube_x_data
        # Set up threshold_cube_y.
        threshold_cube_y_data = np.array(
            [[[[0., 1., 0., 0.],
               [1., 0., 0., 1.],
               [0., 1., 1., 1.]]]])
        threshold_cube_y = self.cube.copy()
        threshold_cube_y = threshold_cube_y[:, :, :-1, :]
        threshold_cube_y.data = threshold_cube_y_data
        thresholded_cube = iris.cube.CubeList(
            [threshold_cube_x, threshold_cube_y])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii
            ).sum_differences_between_adjacent_grid_squares(
                self.cube, thresholded_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_2d_input_cube(self):
        """Test that the sum of differences between adjacent grid squares,
        when accounting for the offset between the grid of the difference
        cube and the original grid is as expected for a 2d cube."""
        expected = np.array(
            [[0., 2., 1., 0.],
             [2., 2., 0., 1.],
             [1., 2., 3., 3.],
             [1., 2., 1., 1.]])
        cube = self.cube[0, 0, :, :]
        # Set up threshold_cube_x.
        threshold_cube_x_data = np.array(
            [[0., 1., 0.],
             [1., 0., 0.],
             [0., 1., 1.],
             [1., 0., 0.]])
        threshold_cube_x = cube.copy()
        threshold_cube_x = threshold_cube_x[:, :-1]
        threshold_cube_x.data = threshold_cube_x_data
        # Set up threshold_cube_y.
        threshold_cube_y_data = np.array(
            [[0., 1., 0., 0.],
             [1., 0., 0., 1.],
             [0., 1., 1., 1.]])
        threshold_cube_y = cube.copy()
        threshold_cube_y = threshold_cube_y[:-1, :]
        threshold_cube_y.data = threshold_cube_y_data
        thresholded_cube = iris.cube.CubeList(
            [threshold_cube_x, threshold_cube_y])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii
            ).sum_differences_between_adjacent_grid_squares(
                cube, thresholded_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii = 2000.0
        self.cube = set_up_precipitation_rate_cube()

    def test_use_adjacent_grid_square_differences(self):
        """Diagnose convective precipitation using the differences between
        adjacent grid squares."""
        expected = np.array(
            [[[[0., 0., 0., 0.],
               [0.357143, 0.318182, 0.272727, 0.214286],
               [0.6, 0.571429, 0.526316, 0.454545],
               [0.818182, 0.8, 0.769231, 0.714286]]]])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii).process(self.cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_does_not_use_adjacent_grid_square_differences(self):
        """Diagnose convective precipitation using the precipitation rate
        field directly, rather than calculating differences between adjacent
        grid squares."""
        expected = np.array(
            [[[[0., 0., 0., 0.],
               [0.2, 0.28571429, 0.28571429, 0.4],
               [0.5, 0.57142857, 0.625, 0.66666667],
               [1., 1., 1., 1.]]]])
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii,
            use_adjacent_grid_square_differences=False).process(self.cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()

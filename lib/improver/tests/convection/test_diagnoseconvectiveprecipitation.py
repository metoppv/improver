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
"""Unit tests for the convection.DiagnoseConvectivePrecipitation plugin."""


import unittest


from cf_units import Unit
import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.convection import DiagnoseConvectivePrecipitation


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
                realizations=np.array([0]), timesteps=1,
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
    cube.add_dim_coord(DimCoord(np.linspace(402192.5, 402292.5, timesteps),
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(y_dimension_values,
                                'projection_y_coordinate', units='m'), 2)
    cube.add_dim_coord(DimCoord(x_dimension_values,
                                'projection_x_coordinate', units='m'), 3)
    return cube


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii_in_km = 2
        result = str(DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii_in_km))
        msg = ('<DiagnoseConvectivePrecipitation: lower_threshold 2.7778e-10; '
               'higher_threshold 1.3889e-06; neighbourhood_method: square; '
               'radii_in_km: 2; fuzzy_factor None; below_thresh_ok: False; '
               'lead_times: None; unweighted_mode: False; ens_factor: 1.0; '
               'use_adjacent_grid_square_differences: True>')
        self.assertEqual(str(result), msg)


class Test__calculate_convective_ratio(IrisTest):

    """Test the _calculate_convective_ratio method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii_in_km = 2.0
        self.cube = set_up_precipitation_rate_cube()

    def test_basic(self):
        """Test a basic example using the default values for the keyword
        arguments. Make sure that the output is a cube with the expected
        data."""
        expected = np.array(
            [[0., 0., 0., 0.],
             [0.2, 0.28571429, 0.28571429, 0.4],
             [0.5, 0.57142857, 0.625, 0.66666667],
             [1., 1., 1., 1.]])

        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            self.radii_in_km)._calculate_convective_ratio(self.cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_no_precipitation(self):
        """If there is no precipitation, then the convective ratio will try
        to do a 0/0 division, which will result in NaN values. Check that
        that output array works as intended."""
        expected = np.array(
            [[np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan]])
        data = np.zeros((1, 1, 4, 4))
        cube = set_up_cube(data, "lwe_precipitation_rate", "m s-1")
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            self.radii_in_km)._calculate_convective_ratio(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_multiple_realizations_and_multiple_times(self):
        """Test a basic example using the default values for the keyword
        arguments. Make sure that the output is a cube with the expected
        data."""
        expected = np.array(
            [[[[0., 0.,  0., 0.],
               [0., 0.,  0., 0.],
               [0., 0.,  0., 0.],
               [0., 0.,  0., 0.]],
              [[0., 0.,  0., 0.],
               [0., 0.,  0., 0.],
               [0., 0.,  0., 0.],
               [0., 0.,  0., 0.]]],
             [[[0., 0.,  0., 0.],
               [0., 0.,  0., 0.],
               [0., 0.,  0., 0.],
               [0., 0.,  0., 0.]],
              [[0.5, 0.33333333,  0., 0.],
               [0.5, 0.33333333,  0., 0.],
               [0.5, 0.33333333,  0., 0.],
               [0.5, 0.33333333,  0., 0.]]]])
        data = np.full((2, 2, 4, 4), 1.0 * mm_hr_to_m_s)
        data[1, 1, :, 0] = 20.0 * mm_hr_to_m_s
        radii_in_km = 4.0
        cube = set_up_cube(
            data, "lwe_precipitation_rate", "m s-1",
            realizations=np.array([0, 1]), timesteps=2)
        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method,
            radii_in_km)._calculate_convective_ratio(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up the cube."""
        self.lower_threshold = 0.001 * mm_hr_to_m_s
        self.higher_threshold = 5 * mm_hr_to_m_s
        self.neighbourhood_method = "square"
        self.radii_in_km = 2
        self.cube = set_up_precipitation_rate_cube()

    def test_use_adjacent_grid_square_differences(self):
        """Diagnose convective precipitation using the differences between
        adjacent grid squares."""
        expected_x = np.array(
            [[np.nan, 0., 0.],
             [1., 0.5, 0.5],
             [1., 1., 1.],
             [1., 1., 1.]])
        expected_y = np.array(
            [[0., 0., 0., 0.],
             [0.25, 0.285714, 0.375, 0.333333],
             [0.5, 0.5, 0.6, 0.5]])

        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii_in_km).process(self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, expected_x)
        self.assertArrayAlmostEqual(result[1].data, expected_y)

    def test_does_not_use_adjacent_grid_square_differences(self):
        """Diagnose convective precipitation using the precipitation rate
        field directly, rather than calculating differences between adjacent
        grid squares."""
        expected = np.array(
            [[0., 0., 0., 0.],
             [0.2, 0.28571429, 0.28571429, 0.4],
             [0.5, 0.57142857, 0.625, 0.66666667],
             [1., 1., 1., 1.]])

        result = DiagnoseConvectivePrecipitation(
            self.lower_threshold, self.higher_threshold,
            self.neighbourhood_method, self.radii_in_km,
            use_adjacent_grid_square_differences=False).process(self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, expected)


if __name__ == '__main__':
    unittest.main()

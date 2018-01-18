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
"""
Unit tests for the plugins and functions within statistical_operations.py
"""

import unittest
import numpy as np

import iris
from iris.tests import IrisTest
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_checker import find_percentile_coordinate
from improver.utilities.statistical_operations import \
    ProbabilitiesFromPercentiles2D


def set_up_percentiles_cube():
    """ Set up 3D cube with percentiles of height """

    test_data = np.full((5, 4, 4), -1)
    for i in xrange(5):
        test_data[i].fill(100*i + 200)

    percentiles = DimCoord(np.linspace(0, 100, 5), long_name="percentiles",
                           units="%")
    grid_x = DimCoord(np.arange(4), standard_name="projection_x_coordinate",
                      units="km")
    grid_y = DimCoord(np.arange(4), standard_name="projection_y_coordinate",
                      units="km")
    test_cube = iris.cube.Cube(test_data, long_name="snow_level", units="m",
                               dim_coords_and_dims=[(percentiles, 0),
                                                    (grid_y, 1), (grid_x, 2)])
    return test_cube


def set_up_threshold_cube():
    """
    Set up 2D cube with "orography" data on which to threshold percentiles
    """
    test_data = 50*np.arange(16).reshape(4, 4)
    grid_x = DimCoord(np.arange(4), standard_name="projection_x_coordinate",
                      units="km")
    grid_y = DimCoord(np.arange(4), standard_name="projection_y_coordinate",
                      units="km")
    test_cube = iris.cube.Cube(test_data, long_name="topography", units="m",
                               dim_coords_and_dims=[(grid_y, 0), (grid_x, 1)])
    return test_cube


def set_reference_probabilities():
    """
    Define the array of probabilities correct for the percentile and threshold
    cubes above.
    """
    reference_array = np.zeros(shape=(16,))
    reference_array[:4] = 0.
    reference_array[4:12] = np.linspace(0., 0.875, 8)
    reference_array[12:] = 1.
    return reference_array.reshape(4, 4)


class Test__init__(IrisTest):
    """ Test initialisation of the ProbabilitiesFromPercentiles2D class """
    def setUp(self):
        """ Set up test cube """
        self.test_cube = set_up_percentiles_cube()
        self.new_name = "probability"

    def test_basic(self):
        """ Test name and default ordering are correctly set """
        pfp_instance = ProbabilitiesFromPercentiles2D(self.test_cube,
                                                      self.new_name)
        self.assertEqual(pfp_instance.output_name, self.new_name)

    def test_naming(self):
        """ Test default naming """
        pfp_instance = ProbabilitiesFromPercentiles2D(self.test_cube)
        self.assertEqual(pfp_instance.output_name,
                         "probability_of_{}".format(self.test_cube.name()))


class Test__repr__(IrisTest):
    """ Test string representation """
    def test_basic(self):
        """ Compare __repr__ string with expectation """
        new_name = "probability"
        test_cube = set_up_percentiles_cube()
        expected = ('<ProbabilitiesFromPercentiles2D: percentiles_'
                    'cube: {}, output_name: {}'.format(
                        test_cube, new_name))
        result = str(ProbabilitiesFromPercentiles2D(test_cube,
                                                    new_name))
        self.assertEqual(result, expected)


class Test_create_probability_cube(IrisTest):
    """
    Test creation of new probability cube based on percentile and orography
    input cubes
    """
    def setUp(self):
        """ Set up a probability cube from percentiles and orography """
        self.percentiles_cube = set_up_percentiles_cube()
        self.percentile_coordinate = find_percentile_coordinate(
            self.percentiles_cube)
        self.new_name = "probability"
        self.pfp_instance = ProbabilitiesFromPercentiles2D(
            self.percentiles_cube, self.new_name)
        self.orography_cube = set_up_threshold_cube()

    def test_attributes(self):
        """ Test name and units are correctly set """
        result = self.pfp_instance.create_probability_cube(
            self.percentiles_cube, self.percentile_coordinate)
        self.assertEqual(result.units, "1")
        self.assertEqual(result.name(), self.new_name)

    def test_coordinate_collapse(self):
        """ Test probability cube has no percentile coordinate """
        result = self.pfp_instance.create_probability_cube(
            self.percentiles_cube, self.percentile_coordinate)
        with self.assertRaises(CoordinateNotFoundError):
            find_percentile_coordinate(result)


class Test_percentile_interpolation(IrisTest):
    """
    Metadata and quantitative tests of function that performs the percentile
    interpolation
    """
    def setUp(self):
        """ Set up orography cube """
        self.orography_cube = set_up_threshold_cube()

    def test_values(self):
        """
        Test that interpolated probabilities at given topography heights are
        sensible.  Includes out-of-range values (P=0 and P=1).
        """
        percentiles_cube = set_up_percentiles_cube()
        expected = set_reference_probabilities()

        probability_cube = ProbabilitiesFromPercentiles2D(
            percentiles_cube).percentile_interpolation(
                self.orography_cube, percentiles_cube)
        self.assertArrayAlmostEqual(probability_cube.data, expected)

    def test_values_inverse_ordering(self):
        """
        Test that interpolated probabilities at given topography heights are
        sensible when we use the inverse_ordering set to True. This is for
        situations in which the values associated with the percentiles increase
        in the opposite direction, e.g. 0 % = 100m, 20% = 50m, etc.

        In this situation we expect the lowest points to have a probability of
        1, and the highest points to have probabilities of 0. The probabilities
        between should be the inverse of what is found in the usual case.
        """
        percentiles_cube = set_up_percentiles_cube()
        # Invert the values associated with the percentiles.
        percentiles_cube.data = np.flipud(percentiles_cube.data)
        expected = set_reference_probabilities()
        expected = 1.0 - expected

        probability_cube = ProbabilitiesFromPercentiles2D(
            percentiles_cube, inverse_ordering=True).percentile_interpolation(
                self.orography_cube, percentiles_cube)
        self.assertArrayAlmostEqual(probability_cube.data, expected)

    def test_equal_percentiles(self):
        """
        Test for sensible behaviour when some percentile levels are equal
        """
        percentiles_cube = set_up_percentiles_cube()
        percentiles_cube.data[0].fill(300)
        expected = set_reference_probabilities()
        expected[np.where(expected <= 0.25)] = 0.
        probability_cube = ProbabilitiesFromPercentiles2D(
            percentiles_cube).percentile_interpolation(
                self.orography_cube, percentiles_cube)

        self.assertArrayAlmostEqual(probability_cube.data, expected)


class Test_process(IrisTest):
    """
    Test top level processing function that calls percentile_interpolation()
    """
    def setUp(self):
        """ Set up class instance and orography cube """
        percentiles_cube = set_up_percentiles_cube()
        self.pfp_instance = ProbabilitiesFromPercentiles2D(percentiles_cube)
        self.orography_cube = set_up_threshold_cube()

    def test_basic(self):
        """
        Test the "process" function returns a single cube whose shape matches
        that of the input threshold (orography) field
        """
        probability_cube = self.pfp_instance.process(self.orography_cube)
        self.assertIsInstance(probability_cube, iris.cube.Cube)
        self.assertSequenceEqual(probability_cube.shape,
                                 self.orography_cube.shape)

    def test_threshold_dimensions(self):
        """
        Test threshold data is correctly sliced and processed if eg a 2-field
        orography cube is passed into the "process" function
        """
        threshold_data_3d = np.broadcast_to(self.orography_cube.data,
                                            (2, 4, 4))
        grid_x = self.orography_cube.coord("projection_x_coordinate")
        grid_y = self.orography_cube.coord("projection_y_coordinate")
        realization = DimCoord(np.arange(2), standard_name="realization",
                               units="1")
        threshold_cube = iris.cube.Cube(threshold_data_3d,
                                        long_name="topography", units="m",
                                        dim_coords_and_dims=[(realization, 0),
                                                             (grid_y, 1),
                                                             (grid_x, 2)])

        probability_cube = self.pfp_instance.process(threshold_cube)
        print probability_cube
        self.assertSequenceEqual(probability_cube.shape,
                                 self.orography_cube.shape)
        self.assertArrayAlmostEqual(probability_cube.data,
                                    set_reference_probabilities())


if __name__ == '__main__':
    unittest.main()

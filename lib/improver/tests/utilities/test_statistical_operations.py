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

    test_data = np.full((5, 10, 10), -1)
    test_data[0].fill(500)
    test_data[1].fill(750)
    test_data[2].fill(1000)
    test_data[3].fill(1250)
    test_data[4].fill(1500)

    percentiles = DimCoord(np.linspace(0, 100, 5), long_name="percentiles",
                           units="%")
    grid_x = DimCoord(np.arange(10), standard_name="projection_x_coordinate",
                      units="km")
    grid_y = DimCoord(np.arange(10), standard_name="projection_y_coordinate",
                      units="km")
    test_cube = iris.cube.Cube(test_data, long_name="test data", units="m",
                               dim_coords_and_dims=[(percentiles, 0),
                                                    (grid_y, 1), (grid_x, 2)])
    return test_cube


def set_up_threshold_cube():
    """
    Set up 2D cube with "orography" data on which to threshold percentiles
    """
    test_data = 20.*np.arange(100).reshape(10, 10)
    grid_x = DimCoord(np.arange(10), standard_name="projection_x_coordinate",
                      units="km")
    grid_y = DimCoord(np.arange(10), standard_name="projection_y_coordinate",
                      units="km")
    test_cube = iris.cube.Cube(test_data, long_name="topography", units="m",
                               dim_coords_and_dims=[(grid_y, 0), (grid_x, 1)])
    return test_cube


def set_reference_array():
    """
    Define a linear array of probabilities correct for the percentile and
    threshold cubes above, in all but shape.
    """
    reference_array = np.zeros(shape=(100,))
    reference_array[:25] = 0.
    reference_array[25:75] = np.arange(0., 1., 0.02)
    reference_array[75:] = 1.
    return reference_array


def set_reference_probabilities():
    """
    Define the array of probabilities correct for the percentile and threshold
    cubes above.
    """
    reference_array = set_reference_array()
    return reference_array.reshape(10, 10)


def set_up_percentiles_cube_2():
    """ Set up 3D cube with some percentiles at the same height """

    test_data = np.full((5, 10, 10), -1)
    test_data[0].fill(750)
    test_data[1].fill(750)
    test_data[2].fill(1000)
    test_data[3].fill(1250)
    test_data[4].fill(1500)

    percentiles = DimCoord(np.linspace(0, 100, 5), long_name="percentiles",
                           units="%")
    grid_x = DimCoord(np.arange(10), standard_name="projection_x_coordinate",
                      units="km")
    grid_y = DimCoord(np.arange(10), standard_name="projection_y_coordinate",
                      units="km")
    test_cube = iris.cube.Cube(test_data, long_name="test data", units="m",
                               dim_coords_and_dims=[(percentiles, 0),
                                                    (grid_y, 1), (grid_x, 2)])
    return test_cube

def set_reference_probabilities_2():
    """
    Define the array of probabilities correct for a modified percentile cube
    where the 0th and 25th percentiles are colocated at 750 m.
    """
    reference_array = set_reference_array()
    reference_array[np.where(reference_array < 0.25)] = 0.
    return reference_array.reshape(10, 10)


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
        self.assertFalse(pfp_instance.inverse_ordering)

    def test_inverse_ordering(self):
        """ Test set inverse ordering """
        pfp_instance = ProbabilitiesFromPercentiles2D(self.test_cube,
                                                      self.new_name,
                                                      inverse_ordering=True)
        self.assertTrue(pfp_instance.inverse_ordering)
    

class Test__repr__(IrisTest):
    """ Test string representation """

    def setUp(self):
        """ Set up test cube """
        self.test_cube = set_up_percentiles_cube()
        self.new_name = "probability"
        self.pfp_instance = ProbabilitiesFromPercentiles2D(self.test_cube,
                                                           self.new_name)
        self.reference_repr = ('<ProbabilitiesFromPercentiles2D: percentiles_'
                               'cube: {}, output_name: {}, inverse_ordering: '
                               '{}'.format(self.test_cube, self.new_name,
                                           False))

    def test_basic(self):
        """ Compare __repr__ string with expectation """
        result = str(self.pfp_instance)
        self.assertEqual(result, self.reference_repr)
        

class Test_create_probability_cube(IrisTest):
    """
    Test creation of new probability cube based on percentile and orography
    input cubes
    """

    def setUp(self):
        """ Set up a probability cube from percentiles and orography """
        self.percentiles_cube = set_up_percentiles_cube()
        self.new_name = "probability"
        self.pfp_instance = ProbabilitiesFromPercentiles2D(self.percentiles_cube,
                                                           self.new_name)
        self.orography_cube = set_up_threshold_cube()
        self.probability_cube = \
             self.pfp_instance.create_probability_cube(self.orography_cube)

    def test_attributes(self):
        """ Test name and units are correctly set """
        self.assertEqual(self.probability_cube.units, "1")
        self.assertEqual(self.probability_cube.name(), self.new_name)

    def test_coordinate_collapse(self):
        """ Test probability cube has no percentile coordinate """
        with self.assertRaises(CoordinateNotFoundError):
            percentile_coordinate = \
                find_percentile_coordinate(self.probability_cube)


class Test_percentile_interpolation(IrisTest):

    def setUp(self):
        """ Set up a probability cube from percentiles and orography """
        self.new_name = "probability"
        self.orography_cube = set_up_threshold_cube()

    def test_values(self):
        """
        Test that interpolated probabilities at given topography heights are
        sensible.  Includes out-of-range values (P=0 and P=1).
        """
        percentiles_cube = set_up_percentiles_cube()
        pfp_instance = ProbabilitiesFromPercentiles2D(percentiles_cube, 
                                                      self.new_name)
        probability_cube = \
            pfp_instance.percentile_interpolation(self.orography_cube,
                                                  percentiles_cube)
        self.assertArrayAlmostEqual(probability_cube.data,
                                    set_reference_probabilities())

    def test_equal_percentiles(self):
        """
        Test for sensible behaviour when some percentile levels are equal.
        """
        percentiles_cube = set_up_percentiles_cube_2()
        pfp_instance = ProbabilitiesFromPercentiles2D(percentiles_cube,
                                                      self.new_name)
        probability_cube = \
            pfp_instance.percentile_interpolation(self.orography_cube,
                                                  percentiles_cube)
        self.assertArrayAlmostEqual(probability_cube.data,
                                    set_reference_probabilities_2())


if __name__ == '__main__':
    unittest.main()


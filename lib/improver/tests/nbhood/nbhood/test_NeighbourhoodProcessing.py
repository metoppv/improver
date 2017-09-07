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
"""Unit tests for the nbhood.NeighbourhoodProcessing plugin."""


import unittest

from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.nbhood.nbhood import NeighbourhoodProcessing as NBHood
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


SINGLE_POINT_RANGE_3_CENTROID = np.array([
    [0.992, 0.968, 0.96, 0.968, 0.992],
    [0.968, 0.944, 0.936, 0.944, 0.968],
    [0.96, 0.936, 0.928, 0.936, 0.96],
    [0.968, 0.944, 0.936, 0.944, 0.968],
    [0.992, 0.968, 0.96, 0.968, 0.992]
])

SINGLE_POINT_RANGE_X_6_Y_3_CENTROID = np.array([
    [1.0, 1.0, 0.99799197, 0.99598394, 0.99799197, 1.0, 1.0],
    [1.0, 0.98995984, 0.98393574, 0.98192771, 0.98393574, 0.98995984, 1.0],
    [0.98995984, 0.97991968, 0.97389558, 0.97188755, 0.97389558, 0.97991968,
     0.98995984],
    [0.98393574, 0.97389558, 0.96787149, 0.96586345, 0.96787149, 0.97389558,
     0.98393574],
    [0.98192771, 0.97188755, 0.96586345, 0.96385542, 0.96586345, 0.97188755,
     0.98192771],
    [0.98393574, 0.97389558, 0.96787149, 0.96586345, 0.96787149, 0.97389558,
     0.98393574],
    [0.98995984, 0.97991968, 0.97389558, 0.97188755, 0.97389558, 0.97991968,
     0.98995984],
    [1.0, 0.98995984, 0.98393574, 0.98192771, 0.98393574, 0.98995984, 1.0],
    [1.0, 1.0, 0.99799197, 0.99598394, 0.99799197, 1.0, 1.0]
])

SINGLE_POINT_RANGE_2_CENTROID_FLAT = np.array([
    [1.0, 1.0, 0.92307692, 1.0, 1.0],
    [1.0, 0.92307692, 0.92307692, 0.92307692, 1.0],
    [0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692],
    [1.0, 0.92307692, 0.92307692, 0.92307692, 1.0],
    [1.0, 1.0, 0.92307692, 1.0, 1.0]
])

SINGLE_POINT_RANGE_5_CENTROID = np.array([
    [1.0, 1.0, 0.99486125, 0.99177801, 0.99075026, 0.99177801,
     0.99486125, 1.0, 1.0],
    [1.0, 0.99280576, 0.98766701, 0.98458376, 0.98355601,
     0.98458376, 0.98766701, 0.99280576, 1.0],
    [0.99486125, 0.98766701, 0.98252826, 0.97944502, 0.97841727,
     0.97944502, 0.98252826, 0.98766701, 0.99486125],
    [0.99177801, 0.98458376, 0.97944502, 0.97636177, 0.97533402,
     0.97636177, 0.97944502, 0.98458376, 0.99177801],
    [0.99075026, 0.98355601, 0.97841727, 0.97533402, 0.97430627,
     0.97533402, 0.97841727, 0.98355601, 0.99075026],
    [0.99177801, 0.98458376, 0.97944502, 0.97636177, 0.97533402,
     0.97636177, 0.97944502, 0.98458376, 0.99177801],
    [0.99486125, 0.98766701, 0.98252826, 0.97944502, 0.97841727,
     0.97944502, 0.98252826, 0.98766701, 0.99486125],
    [1.0, 0.99280576, 0.98766701, 0.98458376, 0.98355601,
     0.98458376, 0.98766701, 0.99280576, 1.0],
    [1.0, 1.0, 0.99486125, 0.99177801, 0.99075026, 0.99177801,
     0.99486125, 1.0, 1.0]
])


def set_up_cube_with_no_realizations(zero_point_indices=((0, 7, 7),),
                                     num_time_points=1,
                                     num_grid_points=16,
                                     source_realizations=None):
    """Set up a cube with equal intervals along the x and y axis."""

    zero_point_indices = list(zero_point_indices)
    for index, indices in enumerate(zero_point_indices):
        if len(indices) == 2:
            indices = (0,) + indices
        zero_point_indices[index] = indices
    zero_point_indices = tuple(zero_point_indices)

    data = np.ones((num_time_points,
                    num_grid_points,
                    num_grid_points))
    for indices in zero_point_indices:
        time_index, lat_index, lon_index = indices
        data[time_index][lat_index][lon_index] = 0

    cube = Cube(data, standard_name="precipitation_amount",
                units="kg m^-2")

    if source_realizations is not None:
        if isinstance(source_realizations, list):
            cube.attributes.update(
                {'source_realizations': source_realizations})

    tunit = Unit("hours since 1970-01-01 00:00:00", "gregorian")
    time_points = [402192.5 + _ for _ in range(num_time_points)]
    cube.add_dim_coord(DimCoord(time_points,
                                standard_name="time", units=tunit), 0)

    step_size = 2000
    y_points = np.arange(0, step_size*num_grid_points, step_size)
    cube.add_dim_coord(
        DimCoord(
            y_points,
            'projection_y_coordinate',
            units='m'
        ),
        1
    )

    x_points = np.arange(-50000, (step_size*num_grid_points)-50000, step_size)
    cube.add_dim_coord(
        DimCoord(
            x_points,
            'projection_x_coordinate',
            units='m'
        ),
        2
    )

    return cube


def set_up_cube_lat_long(zero_point_indices=((0, 7, 7),), num_time_points=1,
                         num_grid_points=16):
    """Set up a lat-long coord cube."""
    data = np.ones((num_time_points, num_grid_points, num_grid_points))
    for time_index, lat_index, lon_index in zero_point_indices:
        data[time_index][lat_index][lon_index] = 0
    cube = Cube(data, standard_name="precipitation_amount",
                units="kg m^-2")
    tunit = Unit("hours since 1970-01-01 00:00:00", "gregorian")
    time_points = [402192.5 + _ for _ in range(num_time_points)]
    cube.add_aux_coord(
        AuxCoord(time_points, "time", units=tunit), 0)
    cube.add_dim_coord(
        DimCoord(np.linspace(0.0, float(num_grid_points - 1),
                             num_grid_points),
                 'latitude',
                 units='degrees'),
        1
    )
    cube.add_dim_coord(
        DimCoord(np.linspace(0.0, float(num_grid_points - 1),
                             num_grid_points),
                 'longitude',
                 units='degrees'),
        2
    )
    return cube


class Test__init__(IrisTest):

    """Test the __init__ method of NeighbourhoodProcessing."""

    def test_neighbourhood_method_exists(self):
        """Test that no exception is raised if the requested neighbourhood
         method exists."""
        neighbourhood_method = 'circular'
        radii = 10000
        result = NBHood(neighbourhood_method, radii)
        msg = ('<CircularNeighbourhood: weighted_mode: True, '
               'sum_or_fraction: fraction>')
        self.assertEqual(str(result.neighbourhood_method), msg)

    def test_neighbourhood_method_does_not_exist(self):
        """Test that desired error message is raised, if the neighbourhood
        method does not exist."""
        neighbourhood_method = 'nonsense'
        radii = 10000
        msg = 'The neighbourhood_method requested: '
        with self.assertRaisesRegex(KeyError, msg):
            NBHood(neighbourhood_method, radii)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(NBHood("circular", 10000))
        msg = ('<BaseNeighbourhoodProcessing: neighbourhood_method: '
               '<CircularNeighbourhood: weighted_mode: True, '
               'sum_or_fraction: fraction>; '
               'radii: 10000.0; lead_times: None>')
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)

    def test_weighted_mode_is_true(self):
        """Test that the circular neighbourhood processing is successful, if
        the weighted mode is True."""
        expected = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.91666667, 0.875, 0.91666667, 1.],
               [1., 0.875, 0.83333333, 0.875, 1.],
               [1., 0.91666667, 0.875, 0.91666667, 1.],
               [1., 1., 1., 1., 1.]]]])
        neighbourhood_method = 'circular'
        radii = 4000
        result = NBHood(neighbourhood_method, radii).process(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_weighted_mode_is_false(self):
        """Test that the circular neighbourhood processing is successful, if
        the weighted mode is False."""
        expected = np.array(
            [[[[1., 1., 0.92307692, 1., 1.],
               [1., 0.92307692, 0.92307692, 0.92307692, 1.],
               [0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692],
               [1., 0.92307692, 0.92307692, 0.92307692, 1.],
               [1., 1., 0.92307692, 1., 1.]]]])
        neighbourhood_method = 'circular'
        radii = 4000
        result = NBHood(neighbourhood_method, radii,
                        weighted_mode=False).process(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_square_neighbourhood(self):
        """Test that the square neighbourhood processing is successful."""
        expected = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        neighbourhood_method = 'square'
        radii = 2000
        result = NBHood(neighbourhood_method, radii).process(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()

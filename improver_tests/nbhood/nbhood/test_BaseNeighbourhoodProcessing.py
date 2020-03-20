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
"""Unit tests for the nbhood.BaseNeighbourhoodProcessing plugin."""


import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.grids import ELLIPSOID, STANDARD_GRID_CCRS
from improver.nbhood.circular_kernel import CircularNeighbourhood
from improver.nbhood.nbhood import BaseNeighbourhoodProcessing as NBHood
from improver.nbhood.nbhood import SquareNeighbourhood

from ...calibration.ensemble_calibration.helper_functions import (
    add_forecast_reference_time_and_forecast_period)

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


def set_up_cube(zero_point_indices=((0, 0, 7, 7),), num_time_points=1,
                num_grid_points=16, num_realization_points=1):
    """Set up a cube with equal intervals along the x and y axis."""

    zero_point_indices = list(zero_point_indices)
    for index, indices in enumerate(zero_point_indices):
        if len(indices) == 3:
            indices = (0,) + indices
        zero_point_indices[index] = indices
    zero_point_indices = tuple(zero_point_indices)

    data = np.ones((
        num_realization_points, num_time_points,
        num_grid_points, num_grid_points), dtype=np.float32)
    for indices in zero_point_indices:
        realization_index, time_index, lat_index, lon_index = indices
        data[realization_index][time_index][lat_index][lon_index] = 0

    cube = Cube(data, standard_name="precipitation_amount",
                units="kg m^-2")

    cube.add_dim_coord(
        DimCoord(
            range(num_realization_points),
            standard_name='realization'), 0)
    tunit = Unit("hours since 1970-01-01 00:00:00", "gregorian")
    time_points = [402192.5 + _ for _ in range(num_time_points)]
    cube.add_dim_coord(DimCoord(time_points,
                                standard_name="time", units=tunit), 1)

    step_size = 2000
    y_points = np.arange(0., step_size*num_grid_points, step_size,
                         dtype=np.float32)
    cube.add_dim_coord(
        DimCoord(
            y_points,
            'projection_y_coordinate',
            units='m',
            coord_system=STANDARD_GRID_CCRS
        ),
        2
    )

    x_points = np.arange(-50000., (step_size*num_grid_points)-50000, step_size,
                         dtype=np.float32)
    cube.add_dim_coord(
        DimCoord(
            x_points,
            'projection_x_coordinate',
            units='m',
            coord_system=STANDARD_GRID_CCRS
        ),
        3
    )
    return cube


def set_up_cube_with_no_realizations(zero_point_indices=((0, 7, 7),),
                                     num_time_points=1,
                                     num_grid_points=16):
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
            units='m',
            coord_system=STANDARD_GRID_CCRS
        ),
        1
    )

    x_points = np.arange(-50000, (step_size*num_grid_points)-50000, step_size)
    cube.add_dim_coord(
        DimCoord(
            x_points,
            'projection_x_coordinate',
            units='m',
            coord_system=STANDARD_GRID_CCRS
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
                 units='degrees',
                 coord_system=ELLIPSOID),
        1
    )
    cube.add_dim_coord(
        DimCoord(np.linspace(0.0, float(num_grid_points - 1),
                             num_grid_points),
                 'longitude',
                 units='degrees',
                 coord_system=ELLIPSOID),
        2
    )
    return cube


class Test_set_up_cube(IrisTest):

    """Test the set_up_cube method in this module"""

    def test_regrid(self):
        """
        Test that set_up_cube returns a cube that can be regridded
        as this proves that it has a coordinate reference systems.
        """
        result = set_up_cube().regrid(set_up_cube_lat_long(),
                                      iris.analysis.Linear())
        self.assertIsInstance(result, Cube)


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
        with self.assertRaisesRegex(ValueError, msg):
            neighbourhood_method = CircularNeighbourhood()
            NBHood(neighbourhood_method, radii, lead_times=lead_times)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_callable(self):
        """Test that the __repr__ returns the expected string."""
        result = str(NBHood(CircularNeighbourhood(), 10000))
        msg = ('<BaseNeighbourhoodProcessing: neighbourhood_method: '
               '<CircularNeighbourhood: weighted_mode: True, '
               'sum_or_fraction: fraction>; '
               'radii: 10000.0; lead_times: None>')
        self.assertEqual(result, msg)

    def test_not_callable(self):
        """Test that the __repr__ returns the expected string."""
        result = str(NBHood("circular", 10000))
        msg = ('<BaseNeighbourhoodProcessing: neighbourhood_method: '
               'circular; radii: 10000.0; lead_times: None>')
        self.assertEqual(result, msg)


class Test__find_radii(IrisTest):

    """Test the internal _find_radii function is working correctly."""

    def test_basic_float_cube_lead_times_is_none(self):
        """Test _find_radii returns an unaltered radius if
        the lead times are none, and this radius is a float."""
        neighbourhood_method = CircularNeighbourhood()
        radius = 6300
        plugin = NBHood(neighbourhood_method,
                        radius)
        result = plugin._find_radii(cube_lead_times=None)
        expected_result = 6300.0
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected_result)

    def test_basic_array_cube_lead_times_an_array(self):
        """Test _find_radii returns an array with the correct values."""
        neighbourhood_method = CircularNeighbourhood
        fp_points = np.array([2, 3, 4])
        radii = [10000, 20000, 30000]
        lead_times = [1, 3, 5]
        plugin = NBHood(neighbourhood_method(),
                        radii,
                        lead_times=lead_times)
        result = plugin._find_radii(cube_lead_times=fp_points)
        expected_result = np.array([15000., 20000., 25000.])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolation(self):
        """Test that interpolation is working as expected in _find_radii."""
        neighbourhood_method = CircularNeighbourhood
        fp_points = np.array([2, 3, 4])
        radii = [10000, 30000]
        lead_times = [2, 4]
        plugin = NBHood(neighbourhood_method(),
                        radii=radii, lead_times=lead_times)
        result = plugin._find_radii(cube_lead_times=fp_points)
        expected_result = np.array([10000., 20000., 30000.])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_process(IrisTest):

    """Tests for the process method of NeighbourhoodProcessing."""

    RADIUS = 6300  # Gives 3 grid cells worth.

    def setUp(self):
        """Set up cube."""
        self.cube = set_up_cube()

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        neighbourhood_method = CircularNeighbourhood()
        result = NBHood(neighbourhood_method, self.RADIUS)(self.cube)
        self.assertIsInstance(result, Cube)

    def test_neighbourhood_method_does_not_exist(self):
        """
        Test that desired error message is raised, if the neighbourhood method
        does not exist.
        """
        neighbourhood_method = 'nonsense'
        radii = 10000
        msg = "is not valid as a neighbourhood_method"
        with self.assertRaisesRegex(ValueError, msg):
            NBHood(neighbourhood_method, radii)(self.cube)

    def test_single_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        self.cube.data[0][0][6][7] = np.NAN
        msg = "NaN detected in input cube data"
        with self.assertRaisesRegex(ValueError, msg):
            neighbourhood_method = CircularNeighbourhood
            NBHood(neighbourhood_method, self.RADIUS)(self.cube)

    def test_multiple_realizations(self):
        """Test when the cube has a realization dimension."""
        cube = set_up_cube(num_realization_points=4)
        radii = 5600
        neighbourhood_method = CircularNeighbourhood()
        result = NBHood(neighbourhood_method, radii)(cube)
        self.assertIsInstance(result, Cube)
        expected = np.ones([4, 1, 16, 16])
        expected[0, 0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667])
        self.assertArrayAlmostEqual(result.data, expected)

    def test_no_realizations(self):
        """Test when the array has no realization coord."""
        cube = set_up_cube_with_no_realizations()
        radii = 5600
        neighbourhood_method = CircularNeighbourhood()
        result = NBHood(neighbourhood_method, radii)(cube)
        self.assertIsInstance(result, Cube)
        expected = np.ones([1, 16, 16])
        expected[0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667])
        self.assertArrayAlmostEqual(result.data, expected)

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
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(cube.coord("forecast_period").units, "hours")

    def test_radii_varying_with_lead_time_fp_seconds(self):
        """
        Test that a cube fp coord is unchanged by the lead time calculation.
        """
        cube = set_up_cube(num_time_points=3)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        cube.coord("forecast_period").convert_units("seconds")
        radii = [10000, 20000, 30000]
        lead_times = [2, 3, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(cube.coord("forecast_period").units, "seconds")

    def test_radii_varying_with_lead_time_check_data(self):
        """
        Test that the expected data is produced when the radius
        varies with lead time.
        """
        cube = set_up_cube(
            zero_point_indices=((0, 0, 7, 7), (0, 1, 7, 7,), (0, 2, 7, 7)),
            num_time_points=3)
        expected = np.ones_like(cube.data)
        expected[0, 0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667])

        expected[0, 1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID

        expected[0, 2, 4:11, 4:11] = (
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9825, 0.97, 0.9625, 0.96, 0.9625, 0.97, 0.9825],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1])

        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii = [5600, 7600, 9500]
        lead_times = [2, 3, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_radii_varying_with_lead_time_with_interpolation(self):
        """Test that a cube is returned for the following conditions:
        1. The radius varies with lead time.
        2. Linear interpolation is required to create values for the radii
        which are required but were not specified within the 'radii'
        argument."""
        cube = set_up_cube(num_time_points=3)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii = [10000, 30000]
        lead_times = [2, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(cube)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time_with_interpolation_check_data(self):
        """Test that a cube with the correct data is returned for the
        following conditions:
        1. The radius varies with lead time.
        2. Linear interpolation is required to create values for the radii
        which are required but were not specified within the 'radii'
        argument."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 7, 7), (0, 1, 7, 7,), (0, 2, 7, 7)),
            num_time_points=3)
        expected = np.ones_like(cube.data)
        expected[0, 0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667])

        expected[0, 1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID

        expected[0, 2, 4:11, 4:11] = (
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9825, 0.97, 0.9625, 0.96, 0.9625, 0.97, 0.9825],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1])

        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii = [5600, 9500]
        lead_times = [2, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_use_mask_cube_occurrences_not_masked(self):
        """Test that the plugin returns an iris.cube.Cube with the correct
        data array if a mask cube is used and the mask cube does not mask
        out the occurrences."""
        expected = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),),
            num_grid_points=5, num_time_points=1)
        cube = iris.util.squeeze(cube)
        mask_cube = cube.copy()
        mask_cube.data = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        radius = 2000
        neighbourhood_method = SquareNeighbourhood()
        result = NBHood(neighbourhood_method, radius)(
            cube, mask_cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_use_mask_cube_occurrences_masked(self):
        """Test that the plugin returns an iris.cube.Cube with the correct
        data array if a mask cube is used and the mask cube does mask
        out the occurrences."""
        expected = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),),
            num_grid_points=5, num_time_points=1)
        cube = iris.util.squeeze(cube)
        mask_cube = cube.copy()
        mask_cube.data = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        radius = 2000
        neighbourhood_method = SquareNeighbourhood()
        result = NBHood(neighbourhood_method, radius)(
            cube, mask_cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_use_mask_cube_occurrences_masked_irregular(self):
        """Test that the plugin returns an iris.cube.Cube with the correct
        data array if a mask cube is used and the mask cube does mask
        out the occurrences. In this case, an irregular mask is applied."""
        expected = np.array(
            [[1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
             [1.000000, 0.000000, 0.833333, 0.000000, 1.000000],
             [1.000000, 0.000000, 0.833333, 0.875000, 1.000000],
             [1.000000, 0.857143, 0.833333, 0.857143, 1.000000],
             [1.000000, 0.000000, 1.000000, 0.000000, 0.000000]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),),
            num_grid_points=5, num_time_points=1)
        cube = iris.util.squeeze(cube)
        mask_cube = cube.copy()
        mask_cube.data = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1.],
             [1., 0., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 0.]])
        radius = 2000
        neighbourhood_method = SquareNeighbourhood()
        result = NBHood(neighbourhood_method, radius)(
            cube, mask_cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()

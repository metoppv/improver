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

from cf_units import Unit
import iris
from iris.coords import AuxCoord, DimCoord
from iris.coord_systems import OSGB
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np


from improver.grids.osgb import OSGBGRID
from improver.nbhood import NeighbourhoodProcessing as NBHood
from improver.tests.helper_functions_ensemble_calibration import (
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
    """Set up a normal OSGB UK National Grid cube."""

    zero_point_indices = list(zero_point_indices)
    for index, indices in enumerate(zero_point_indices):
        if len(indices) == 3:
            indices = (0,) + indices
        zero_point_indices[index] = indices
    zero_point_indices = tuple(zero_point_indices)

    data = np.ones((
        num_realization_points, num_time_points,
        num_grid_points, num_grid_points))
    for indices in zero_point_indices:
        realization_index, time_index, lat_index, lon_index = indices
        data[realization_index][time_index][lat_index][lon_index] = 0

    cube = Cube(data, standard_name="precipitation_amount",
                units="kg m^-2 s^-1")
    coord_system = OSGB()
    scaled_y_coord = OSGBGRID.coord('projection_y_coordinate')
    cube.add_dim_coord(
        DimCoord(
            range(num_realization_points), 'realization',
            units='degrees'), 0)
    tunit = Unit("hours since 1970-01-01 00:00:00", "gregorian")
    time_points = [402192.5 + _ for _ in range(num_time_points)]
    cube.add_aux_coord(AuxCoord(time_points,
                                "time", units=tunit), 1)
    cube.add_dim_coord(
        DimCoord(
            scaled_y_coord.points[:num_grid_points],
            'projection_y_coordinate',
            units='m', coord_system=coord_system
        ),
        2
    )
    scaled_x_coord = OSGBGRID.coord('projection_x_coordinate')
    cube.add_dim_coord(
        DimCoord(
            scaled_x_coord.points[:num_grid_points],
            'projection_x_coordinate',
            units='m', coord_system=coord_system
        ),
        3
    )
    return cube


def set_up_cube_lat_long(zero_point_indices=((0, 7, 7),), num_time_points=1,
                         num_grid_points=16):
    """Set up a lat-long coord cube."""
    data = np.ones((num_time_points, num_grid_points, num_grid_points))
    for time_index, lat_index, lon_index in zero_point_indices:
        data[time_index][lat_index][lon_index] = 0
    cube = Cube(data, standard_name="precipitation_amount",
                units="kg m^-2 s^-1")
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

    def test_radii_varying_with_lead_time_mismatch(self):
        """
        Test that the desired error message is raised, if there is a mismatch
        between the number of radii and the number of lead times.
        """
        radii_in_km = [10, 20, 30]
        lead_times = [2, 3]
        msg = "There is a mismatch in the number of radii"
        with self.assertRaisesRegexp(ValueError, msg):
            kernel_method = 'circular'
            NBHood(kernel_method, radii_in_km, lead_times=lead_times)

    def test_kernel_method_does_not_exist(self):
        """
        Test that desired error message is raised, if the kernel method
        does not exist.
        """
        kernel_method = 'nonsense'
        radii_in_km = 10
        msg = 'The requested kernel method: '
        with self.assertRaisesRegexp(AttributeError, msg):
            NBHood(kernel_method, radii_in_km)


class Test_process(IrisTest):

    """Tests for the process method of BasicNeighbourhoodProcessing."""

    RADIUS_IN_KM = 6.3  # Gives 3 grid cells worth.

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        cube = set_up_cube()
        kernel_method = "circular"
        plugin = NBHood(kernel_method, self.RADIUS_IN_KM)
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)

    def test_single_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        cube = set_up_cube()
        cube.data[0][0][6][7] = np.NAN
        msg = "NaN detected in input cube data"
        with self.assertRaisesRegexp(ValueError, msg):
            kernel_method = "circular"
            NBHood(kernel_method, self.RADIUS_IN_KM).process(cube)

    def test_fail_multiple_realisations(self):
        """Test failing when the array has a realisation dimension."""
        data = np.ones((14, 1, 16, 16))
        data[0][0][7][7] = 0.0

        cube = Cube(data, standard_name="precipitation_amount",
                    units="kg m^-2 s^-1")
        num_grid_points = 16
        coord_system = OSGB()
        scaled_y_coord = OSGBGRID.coord('projection_y_coordinate')
        cube.add_dim_coord(
            DimCoord(
                scaled_y_coord.points[:num_grid_points],
                'projection_y_coordinate',
                units='m', coord_system=coord_system
            ),
            2
        )
        scaled_x_coord = OSGBGRID.coord('projection_x_coordinate')
        cube.add_dim_coord(
            DimCoord(
                scaled_x_coord.points[:num_grid_points],
                'projection_x_coordinate',
                units='m', coord_system=coord_system
            ),
            3
        )
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(AuxCoord([402192.5],
                                    "time", units=tunit), 1)
        cube.add_aux_coord(AuxCoord(np.array(range(14)),
                                    standard_name="realization"), 0)
        msg = "Does not operate across realizations"
        with self.assertRaisesRegexp(ValueError, msg):
            kernel_method = "circular"
            NBHood(kernel_method, self.RADIUS_IN_KM).process(cube)

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
        radii_in_km = [10, 20, 30]
        lead_times = [2, 3, 4]
        kernel_method = "circular"
        plugin = NBHood(kernel_method, radii_in_km, lead_times)
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time_check_data(self):
        """
        Test that the expected data is produced when the radius
        varies with lead time.
        """
        cube = set_up_cube(
            zero_point_indices=((0, 0, 7, 7), (0, 1, 7, 7,), (0, 2, 7, 7)),
            num_time_points=3)
        expected = np.ones_like(cube.data[0])
        expected[0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667])

        expected[1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID

        expected[2, 4:11, 4:11] = (
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
        radii_in_km = [6, 8, 10]
        lead_times = [2, 3, 4]
        kernel_method = "circular"
        plugin = NBHood(kernel_method, radii_in_km, lead_times)
        result = plugin.process(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_radii_varying_with_lead_time_with_interpolation(self):
        """
        Test that a cube is returned when the radius varies with lead time
        and linearly interpolation is required, in order to .
        """
        cube = set_up_cube(num_time_points=3)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        time_points = cube.coord("time").points
        fp_points = [2, 3, 4]
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=time_points, fp_point=fp_points)
        radii_in_km = [10, 30]
        lead_times = [2, 4]
        kernel_method = "circular"
        plugin = NBHood(kernel_method, radii_in_km, lead_times)
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time_with_interpolation_check_data(self):
        """Test behaviour when the radius varies with lead time."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 7, 7), (0, 1, 7, 7,), (0, 2, 7, 7)),
            num_time_points=3)
        expected = np.ones_like(cube.data[0])
        expected[0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667])

        expected[1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID

        expected[2, 4:11, 4:11] = (
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
        radii_in_km = [6, 10]
        lead_times = [2, 4]
        kernel_method = "circular"
        plugin = NBHood(kernel_method, radii_in_km, lead_times)
        result = plugin.process(cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()

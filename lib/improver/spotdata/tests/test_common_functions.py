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
"""Unit tests for the spotdata.common_functions."""


import unittest
import cf_units
from datetime import datetime as dt

from iris.coords import (DimCoord,
                         AuxCoord)
from iris import coord_systems
from iris.coord_systems import GeogCS
from iris import Constraint
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime
import cartopy.crs as ccrs
import numpy as np
from iris import FUTURE

from improver.spotdata.common_functions import *

FUTURE.cell_datetime_objects = True


class TestCommonFunctions(IrisTest):

    """Test functions in common_functions."""

    def setUp(self):
        """
        Create a cube containing a regular lat-lon grid.

        Data is striped horizontally,
        e.g.
              1 1 1 1 1 1
              1 1 1 1 1 1
              2 2 2 2 2 2
              2 2 2 2 2 2
              3 3 3 3 3 3
              3 3 3 3 3 3
        """
        data = np.ones((12, 12))
        data[0:4, :] = 1
        data[4:8, :] = 2
        data[8:, :] = 3

        latitudes = np.linspace(-90, 90, 12)
        longitudes = np.linspace(-180, 180, 12)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees', coord_system=GeogCS(6371229.0))
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees', coord_system=GeogCS(6371229.0),
                             circular=True)

        # Use time of 2017-02-17 06:00:00
        time = DimCoord(
            [1487311200], standard_name='time',
            units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                calendar='gregorian'))

        time_dt = dt(2017, 2, 17, 6, 0)
        time_extract = Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))

        cube = Cube(data.reshape((1, 12, 12)),
                    long_name="air_temperature",
                    dim_coords_and_dims=[(time, 0),
                                         (latitude, 1),
                                         (longitude, 2)],
                    units="K")

        orography = Cube(np.ones((12, 12)),
                         long_name="surface_altitude",
                         dim_coords_and_dims=[(latitude, 0),
                                              (longitude, 1)],
                         units="m")

        # Western half of grid at altitude 0, eastern half at 10.
        # Note that the pressure_on_height_levels data is left unchanged,
        # so it is as if there is a sharp front running up the grid with
        # differing pressures on either side at equivalent heights above
        # the surface (e.g. east 1000hPa at 0m AMSL, west 1000hPa at 10m AMSL).
        # So there is higher pressure in the west.
        orography.data[0:10] = 0
        orography.data[10:] = 10
        ancillary_data = {}
        ancillary_data['orography'] = orography

        additional_data = {}
        ad = CubeList()
        ad.append(cube)
        additional_data['air_temperature'] = ad

        data_indices = [list(data.nonzero()[0]),
                        list(data.nonzero()[1])]

        self.cube = cube
        self.data = data
        self.time_dt = time_dt
        self.time_extract = time_extract
        self.data_indices = data_indices
        self.ancillary_data = ancillary_data
        self.additional_data = additional_data


class TestConditionalListExtract(TestCommonFunctions):
    """
    Test numerical comparison of 2D arrays using ConditionalListExtract.

    """

    def test_less_than(self):
        """
        Test conditional less than test. Should return the indices of all
        values less than 2 (the top band of the data array).

        """
        plugin = ConditionalListExtract('less_than')
        expected = [sorted(range(0, 4)*12), range(0, 12)*4]
        result = plugin.process(self.data, self.data_indices, 2)
        self.assertArrayEqual(expected, result)

    def test_greater_than(self):
        """
        Test conditional greater than test. Should return the indices of all
        values greater than 2 (the bottom band of the data array).

        """
        plugin = ConditionalListExtract('greater_than')
        expected = [sorted(range(8, 12)*12), range(0, 12)*4]
        result = plugin.process(self.data, self.data_indices, 2)
        self.assertArrayEqual(expected, result)

    def test_equal_to(self):
        """
        Test conditional equal to test. Should return the indices of all
        values equal to 2 (the middle band of the data array).

        """
        plugin = ConditionalListExtract('equal_to')
        expected = [sorted(range(4, 8)*12), range(0, 12)*4]
        result = plugin.process(self.data, self.data_indices, 2)
        self.assertArrayEqual(expected, result)

    def test_not_equal_to(self):
        """
        Test conditional not equal to test. Should return the indices of all
        values not equal to 2 (the top and bottom bands of the data array).

        """
        plugin = ConditionalListExtract('not_equal_to')
        expected = [sorted((range(0, 4) + range(8, 12))*12), range(0, 12)*8]
        result = plugin.process(self.data, self.data_indices, 2)
        self.assertArrayEqual(expected, result)

    def test_unknown_method(self):
        """Test that the plugin copes with an unknown method."""
        plugin = ConditionalListExtract('not_to')
        msg = 'Unknown method'
        with self.assertRaisesRegexp(AttributeError, msg):
            plugin.process(self.data, self.data_indices, 2)


class TestNearestNNeighbours(TestCommonFunctions):
    """
    Test generation of neighbour grids using nearest_n_neighbours.

    """

    def test_indices_return(self):
        """Test that the correct indices are returned."""
        plugin = nearest_n_neighbours
        expected = [[9, 9, 9, 10, 10, 10, 11, 11, 11],
                    [9, 10, 11, 9, 10, 11, 9, 10, 11]]
        result = plugin(10, 10, 9)
        self.assertArrayEqual(expected, result)

    def test_indices_exclude_self(self):
        """Test that the correct indices are returned."""
        plugin = nearest_n_neighbours
        expected = [[9, 9, 9, 10, 10, 11, 11, 11],
                    [9, 10, 11, 9, 11, 9, 10, 11]]
        result = plugin(10, 10, 9, exclude_self=True)
        self.assertArrayEqual(expected, result)

    def test_invalid_no_neighbours(self):
        """Test function rejects invalid numbers of neighbours."""
        plugin = nearest_n_neighbours
        msg = 'Invalid nearest'
        with self.assertRaisesRegexp(ValueError, msg):
            plugin(10, 10, 7)


class TestNodeEdgeTest(TestCommonFunctions):
    """
    Test functions ability to modify neighbours lists to reflect domain edges
    and domain wrapping on cylindrical global grids.

    """

    def test_lower_bounds_i(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some i (latitude) neighbour nodes are < 0 and should
        be removed.

        """
        plugin = node_edge_test
        # i = 0
        # j = 10
        node_list = [[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                     [9, 10, 11, 9, 10, 11, 9, 10, 11]]
        expected = [[0, 0, 0, 1, 1, 1],
                    [9, 10, 11, 9, 10, 11]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_lower_bounds_j(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some j (longitude) neighbour nodes are < 0 and should
        be wrapped around as we are assuming a global cylindrical grid.

        """
        plugin = node_edge_test
        # i = 10
        # j = 0
        node_list = [[9, 9, 9, 10, 10, 10, 11, 11, 11],
                     [-1, 0, 1, -1, 0, 1, -1, 0, 1]]
        expected = [[9, 9, 9, 10, 10, 10, 11, 11, 11],
                    [11, 0, 1, 11, 0, 1, 11, 0, 1]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_upper_bounds_i(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some i (latitude) neighbour nodes are > imax and should
        be removed.

        """
        plugin = node_edge_test
        # i = 11
        # j = 10
        node_list = [[10, 10, 10, 11, 11, 11, 12, 12, 12],
                     [9, 10, 11, 9, 10, 11, 9, 10, 11]]
        expected = [[10, 10, 10, 11, 11, 11],
                    [9, 10, 11, 9, 10, 11]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_upper_bounds_j(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some j (longitude) neighbour nodes are > jmax and should
        be wrapped around as we are assuming a global cylindrical grid.

        """
        plugin = node_edge_test
        # i = 10
        # j = 11
        node_list = [[9, 9, 9, 10, 10, 10, 11, 11, 11],
                     [10, 11, 12, 10, 11, 12, 10, 11, 12]]
        expected = [[9, 9, 9, 10, 10, 10, 11, 11, 11],
                    [10, 11, 0, 10, 11, 0, 10, 11, 0]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_lower_bounds_ij(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some i & j neighbour nodes are < 0 and should
        be removed or wrapped as appropriate.

        """
        plugin = node_edge_test
        # i = 0
        # j = 0
        node_list = [[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                     [-1, 0, 1, -1, 0, 1, -1, 0, 1]]
        expected = [[0, 0, 0, 1, 1, 1],
                    [11, 0, 1, 11, 0, 1]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_upper_bounds_ij(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some i & j neighbour nodes are > imax/jmax and should
        be removed or wrapped as appropriate.

        """
        plugin = node_edge_test
        # i = 11
        # j = 11
        node_list = [[10, 10, 10, 11, 11, 11, 12, 12, 12],
                     [10, 11, 12, 10, 11, 12, 10, 11, 12]]
        expected = [[10, 10, 10, 11, 11, 11],
                    [10, 11, 0, 10, 11, 0]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_upper_i_lower_j(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some some neighbour nodes i < 0 & j > jmax and should
        be removed or wrapped as appropriate.

        """
        plugin = node_edge_test
        # i = 0
        # j = 11
        node_list = [[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                     [10, 11, 12, 10, 11, 12, 10, 11, 12]]
        expected = [[0, 0, 0, 1, 1, 1],
                    [10, 11, 0, 10, 11, 0]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)

    def test_upper_j_lower_i(self):
        """
        Test nodes spilling over grid boundaries are adjusted appropriately.
        In this case some some neighbour nodes i > imax & j < 0 and should
        be removed or wrapped as appropriate.

        """
        plugin = node_edge_test
        # i = 11
        # j = 0
        node_list = [[10, 10, 10, 11, 11, 11, 12, 12, 12],
                     [-1, 0, 1, -1, 0, 1, -1, 0, 1]]
        expected = [[10, 10, 10, 11, 11, 11],
                    [11, 0, 1, 11, 0, 1]]
        result = plugin(node_list, self.cube)
        self.assertArrayEqual(expected, result)


class TestGetNearestCoords(TestCommonFunctions):
    """Test wrapper for iris.cube.Cube.nearest_neighbour_index."""

    def test_nearest_coords(self):
        """Test correct indices are returned."""
        plugin = get_nearest_coords
        longitude = 80
        latitude = -25
        expected = (4, 8)
        result = plugin(self.cube, latitude, longitude,
                        'latitude', 'longitude')
        self.assertEqual(expected, result)


class TestIndexOfMinimumDifference(TestCommonFunctions):
    """
    Test ability to identify the index of a minimum value in an array,
    where the sign of the value is disregarded (hence minimum_difference).

    """

    def test_whole_list(self):
        """Test function for finding the index in a complete list."""
        plugin = index_of_minimum_difference
        test_list = [-10, 10, -1.5, 1.5, -0.6, 0.4]
        expected = 5  # index of value 0.4
        result = plugin(test_list)
        self.assertEqual(expected, result)

    def test_subset_list(self):
        """Test function when finding the index in a subset of the list."""
        plugin = index_of_minimum_difference
        test_list = [-10, 10, -1.5, 1.5, -0.6, 0.4]
        subset = range(0, 5)
        expected = 4  # index of value -0.6 as 0.4 is not in the sublist.
        result = plugin(test_list, subset_list=subset)
        self.assertEqual(expected, result)


class TestListEntryFromIndex(TestCommonFunctions):
    """
    Test ability to extract corresponding list entries from multi-dimensional
    lists using a single index.

    e.g. [[0,1,2],[3,4,5]]
         index = 1  -->  [1,4]

    """

    def test_2D_list(self):
        """Test in a 2D list."""
        plugin = list_entry_from_index
        test_list = [[10, 9, 8, 7, 6],
                     [1, 2, 3, 4, 5]]
        index = 2
        expected = [8, 3]
        result = plugin(test_list, index)
        self.assertEqual(expected, result)

    def test_3D_list(self):
        """Test in a 3D list."""
        plugin = list_entry_from_index
        test_list = [[10, 9, 8, 7, 6],
                     [1, 2, 3, 4, 5],
                     [10, 9, 8, 7, 6]]
        index = 1
        expected = [9, 2, 9]
        result = plugin(test_list, index)
        self.assertEqual(expected, result)

    def test_ND_list(self):
        """Test in a ND list, where here N=10."""
        plugin = list_entry_from_index
        test_list = [range(0, 5)]*10
        index = 3
        expected = [3]*10
        result = plugin(test_list, index)
        self.assertEqual(expected, result)


class TestDateTimeConstraint(TestCommonFunctions):
    """
    Test construction of an iris.Constraint from a python.datetime.datetime
    object.

    """

    def test_constraint_equality(self):
        """Check constraint is as expected."""
        plugin = datetime_constraint
        dt_constraint = plugin(dt(2017, 2, 17, 6, 0))
        self.assertEqual(self.time_extract._coord_values,
                         dt_constraint._coord_values)

    def test_constraint_type(self):
        """Check type is iris.Constraint."""
        plugin = datetime_constraint
        dt_constraint = plugin(dt(2017, 2, 17, 6, 0))
        self.assertIsInstance(dt_constraint, Constraint)

    def test_valid_constraint(self):
        """Test use of constraint at a time valid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(dt(2017, 2, 17, 6, 0))
        result = self.cube.extract(dt_constraint)
        self.assertIsInstance(result, Cube)

    def test_invalid_constraint(self):
        """Test use of constraint at a time invalid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(dt(2017, 2, 17, 18, 0))
        result = self.cube.extract(dt_constraint)
        self.assertNotIsInstance(result, Cube)


class TestApplyBias(TestCommonFunctions):
    """
    Test subsetting of an array to extract the indices of positive or negative
    values as required by the provided bias condition.

    """

    def test_bias_above(self):
        """Extract indices of negative values from the list."""
        bias = 'above'
        dzs = [-2., -1., 0., 1., 2.]
        plugin = apply_bias
        expected = [0, 1, 2]
        result = plugin(bias, np.array(dzs))
        self.assertArrayEqual(expected, result)

    def test_bias_below(self):
        """Extract indices of positive values from the list."""
        bias = 'below'
        dzs = [-2., -1., 0., 1., 2.]
        plugin = apply_bias
        expected = [2, 3, 4]
        result = plugin(bias, np.array(dzs))
        self.assertArrayEqual(expected, result)

    def test_bias_none(self):
        """Test indices returned with no bias provided."""
        bias = None
        dzs = [-2., -1., 0., 1., 2.]
        plugin = apply_bias
        expected = [0, 1, 2, 3, 4]
        result = plugin(bias, np.array(dzs))
        self.assertArrayEqual(expected, result)

    def test_bias_below_all_above(self):
        """Test indices returned when no values match the bias provided."""
        bias = 'below'
        dzs = [-5., -4., -3., -2., -1.]
        plugin = apply_bias
        expected = [0, 1, 2, 3, 4]
        result = plugin(bias, np.array(dzs))
        self.assertArrayEqual(expected, result)


class TestXYTest(TestCommonFunctions):
    """Test function that tests projections used in diagnostic cubes."""

    def test_projection_test(self):
        """Test identification of non-lat/lon projections."""
        src_crs = ccrs.PlateCarree()
        trg_crs = ccrs.LambertConformal(central_longitude=50,
                                        central_latitude=10)
        trg_crs_iris = coord_systems.LambertConformal(
            central_lon=50, central_lat=10)
        lons = self.cube.coord('longitude').points
        lats = self.cube.coord('latitude').points
        x, y = [], []
        for lon, lat in zip(lons, lats):
            x_trg, y_trg = trg_crs.transform_point(lon, lat, src_crs)
            x.append(x_trg)
            y.append(y_trg)

        new_x = AuxCoord(x, standard_name='projection_x_coordinate',
                         units='m', coord_system=trg_crs_iris)
        new_y = AuxCoord(y, standard_name='projection_y_coordinate',
                         units='m', coord_system=trg_crs_iris)

        cube = Cube(self.cube.data,
                    long_name="air_temperature",
                    dim_coords_and_dims=[(self.cube.coord('time'), 0)],
                    aux_coords_and_dims=[(new_y, 1), (new_x, 2)],
                    units="K")

        plugin = xy_test
        expected = trg_crs
        result = plugin(cube)
        self.assertEqual(expected, result)


class TestXYTransform(TestCommonFunctions):
    """
    Test function that transforms the lookup latitude and longitude into the
    projection used in a diagnostic cube.

    """
    def test_projection_transform(self):
        """
        Test transformation of lookup coordinates to the projections in
        which the diagnostic is provided.

        """
        trg_crs = ccrs.LambertConformal(central_longitude=50,
                                        central_latitude=10)

        plugin = xy_transform
        expected_x, expected_y = 0., 0.
        result_x, result_y = plugin(trg_crs, 10, 50)
        self.assertAlmostEqual(expected_x, result_x)
        self.assertAlmostEqual(expected_y, result_y)


class TestExtractCubeAtTime(TestCommonFunctions):
    """
    Test wrapper for iris cube extraction at desired times.

    """

    def test_valid_time(self):
        """Case for a time that is available within the diagnostic cube."""
        plugin = extract_cube_at_time
        cubes = CubeList([self.cube])
        result = plugin(cubes, self.time_dt, self.time_extract)
        self.assertIsInstance(result, Cube)

    def test_invalid_time(self):
        """Case for a time that is unavailable within the diagnostic cube."""
        plugin = extract_cube_at_time
        time_dt = dt(2017, 2, 18, 6, 0)
        time_extract = Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))
        cubes = CubeList([self.cube])
        with warnings.catch_warnings(record=True) as w_messages:
            plugin(cubes, time_dt, time_extract)
            assert len(w_messages) == 1
            assert issubclass(w_messages[0].category, UserWarning)
            assert "Forecast time" in str(w_messages[0])


class TestExtractAdAtTime(TestCommonFunctions):
    """
    Test for the extraction of additional data, that is stored supplementary
    diagnostics, at a certain time.

    """

    def test_valid_extraction_time(self):
        """
        Case for a time that is available within the additional diagnostic.
        
        """
        plugin = extract_ad_at_time
        result = plugin(self.additional_data, self.time_dt, self.time_extract)
        self.assertIsInstance(result['air_temperature'], Cube)

    def test_invalid_extraction_time(self):
        """
        Case for a time that is available within the additional diagnostic.

        """
        plugin = extract_ad_at_time
        time_dt = dt(2017, 2, 20, 6, 0)
        time_extract = Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))
        with warnings.catch_warnings(record=True) as w_messages:
            plugin(self.additional_data, time_dt, time_extract)
            assert len(w_messages) == 1
            assert issubclass(w_messages[0].category, UserWarning)
            assert "Forecast time" in str(w_messages[0])


if __name__ == '__main__':
    unittest.main()

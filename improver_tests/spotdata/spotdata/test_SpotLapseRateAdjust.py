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
"""Unit tests for SpotLapseRateAdjust class"""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.constants import DALR
from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.apply_lapse_rate import SpotLapseRateAdjust
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.temporal import iris_time_to_datetime

from ...set_up_test_cubes import (
    construct_scalar_time_coords, construct_xy_coords, set_up_variable_cube)


class Test_SpotLapseRateAdjust(IrisTest):

    """Test class for the SpotLapseRateAdjust tests, setting up inputs."""

    def setUp(self):
        """
        Set up cubes for use in testing SpotLapseRateAdjust. Inputs are
        envisaged as follows:

        Gridded

         Lapse rate  Orography  Temperatures (not used directly)
          (x DALR)

            A B C      A B C        A   B   C

        a   2 1 1      1 1 1       270 270 270
        b   1 2 1      1 4 1       270 280 270
        c   1 1 2      1 1 1       270 270 270

        Spot
        (note the neighbours are identified with the A-C, a-c indices above)

         Site  Temperature Altitude  Nearest    DZ   MinDZ      DZ
                                     neighbour       neighbour

          0        280        3      Ac         2    Bb         -1
          1        270        4      Bb         0    Bb          0
          2        280        0      Ca        -1    Ca         -1


        """
        # Set up lapse rate cube
        lapse_rate_data = np.ones(9).reshape(3, 3).astype(np.float32) * DALR
        lapse_rate_data[0, 2] = 2 * DALR
        lapse_rate_data[1, 1] = 2 * DALR
        lapse_rate_data[2, 0] = 2 * DALR
        self.lapse_rate_cube = set_up_variable_cube(lapse_rate_data,
                                                    name="lapse_rate",
                                                    units="K m-1",
                                                    spatial_grid="equalarea")
        diagnostic_cube_hash = create_coordinate_hash(self.lapse_rate_cube)

        # Set up neighbour and spot diagnostic cubes
        y_coord, x_coord = construct_xy_coords(3, 3, "equalarea")
        y_coord = y_coord.points
        x_coord = x_coord.points

        # neighbours, each group is for a point under two methods, e.g.
        # [ 0.  0.  0.] is the nearest point to the first spot site, whilst
        # [ 1.  1. -1.] is the nearest point with minimum height difference.
        neighbours = np.array([[[0., 0., 2.],
                                [1., 1., -1.]],
                               [[1., 1., 0.],
                                [1., 1., 0.]],
                               [[2., 2., -1.],
                                [2., 2., -1.]]])
        altitudes = np.array([3, 4, 0])
        latitudes = np.array([y_coord[0], y_coord[1], y_coord[2]])
        longitudes = np.array([x_coord[0], x_coord[1], x_coord[2]])
        wmo_ids = np.arange(3)
        grid_attributes = ['x_index', 'y_index', 'vertical_displacement']
        neighbour_methods = ['nearest', 'nearest_minimum_dz']
        self.neighbour_cube = build_spotdata_cube(
            neighbours, 'grid_neighbours', 1, altitudes, latitudes,
            longitudes, wmo_ids, grid_attributes=grid_attributes,
            neighbour_methods=neighbour_methods)
        self.neighbour_cube.attributes['model_grid_hash'] = (
            diagnostic_cube_hash)

        time, = iris_time_to_datetime(self.lapse_rate_cube.coord("time"))
        frt, = iris_time_to_datetime(self.lapse_rate_cube.coord(
                "forecast_reference_time"))
        time_bounds = None

        time_coords = construct_scalar_time_coords(time, time_bounds, frt)
        time_coords = [item[0] for item in time_coords]

        # This temperature cube is set up with the spot sites having obtained
        # their temperature values from the nearest grid sites.
        temperatures_nearest = np.array([280, 270, 280])
        self.spot_temperature_nearest = build_spotdata_cube(
            temperatures_nearest, 'air_temperature', 'K', altitudes, latitudes,
            longitudes, wmo_ids, scalar_coords=time_coords)
        self.spot_temperature_nearest.attributes['model_grid_hash'] = (
            diagnostic_cube_hash)

        # This temperature cube is set up with the spot sites having obtained
        # their temperature values from the nearest minimum vertical
        # displacment grid sites. The only difference here is for site 0, which
        # now gets its temperature from Bb (see doc-string above).
        temperatures_mindz = np.array([270, 270, 280])
        self.spot_temperature_mindz = build_spotdata_cube(
            temperatures_mindz, 'air_temperature', 'K', altitudes, latitudes,
            longitudes, wmo_ids, scalar_coords=time_coords)
        self.spot_temperature_mindz.attributes['model_grid_hash'] = (
            diagnostic_cube_hash)


class Test_process(Test_SpotLapseRateAdjust):

    """Tests the class process method."""

    def test_basic(self):
        """Test that the plugin returns a cube which is unchanged except for
        data values."""

        plugin = SpotLapseRateAdjust()
        result = plugin(self.spot_temperature_nearest,
                        self.neighbour_cube,
                        self.lapse_rate_cube)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), self.spot_temperature_nearest.name())
        self.assertEqual(result.units, self.spot_temperature_nearest.units)
        self.assertEqual(result.coords(),
                         self.spot_temperature_nearest.coords())

    def test_nearest_neighbour_method(self):
        """Test that the plugin modifies temperatures as expected using the
        vertical displacements taken from the nearest neighbour method in the
        neighbour cube."""
        plugin = SpotLapseRateAdjust()
        expected = np.array(
            [280 + (2 * DALR), 270, 280 - DALR]).astype(np.float32)

        result = plugin(self.spot_temperature_nearest,
                        self.neighbour_cube,
                        self.lapse_rate_cube)
        self.assertArrayEqual(result.data, expected)

    def test_different_neighbour_method(self):
        """Test that the plugin uses the correct vertical displacements when
        a different neighbour method is set. This should result in these
        different values being chosen from the neighbour cube.

        In this case site 0 has a displacement of -1 from the chosen grid site,
        but the lapse rate at that site is 2*DALR, so the change below is by
        2*DALR, compared with site 2 which has the same displacement, but for
        which the lapse rate is just the DALR."""

        plugin = SpotLapseRateAdjust(
            neighbour_selection_method='nearest_minimum_dz')
        expected = np.array(
            [270 - (2 * DALR), 270, 280 - DALR]).astype(np.float32)

        result = plugin(self.spot_temperature_mindz,
                        self.neighbour_cube,
                        self.lapse_rate_cube)
        self.assertArrayEqual(result.data, expected)

    def test_xy_ordered_lapse_rate_cube(self):
        """Ensure a lapse rate cube that does not have the expected y-x
        ordering does not lead to different results. In this case the
        lapse rate cube looks like this:

         Lapse rate
          (x DALR)

            a b c

        A   1 1 2
        B   1 2 1
        C   2 1 2

        If the alternative ordering were not being handled (in this case by
        the SpotExtraction plugin) we would expect a different result for
        sites 0 and 2."""

        plugin = SpotLapseRateAdjust()
        expected = np.array(
            [280 + (2 * DALR), 270, 280 - DALR]).astype(np.float32)
        enforce_coordinate_ordering(
            self.lapse_rate_cube, ['projection_x_coordinate',
                                   'projection_y_coordinate'])

        result = plugin(self.spot_temperature_nearest,
                        self.neighbour_cube,
                        self.lapse_rate_cube)
        self.assertArrayEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()

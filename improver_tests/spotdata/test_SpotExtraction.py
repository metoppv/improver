# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for SpotExtraction class"""

import unittest
from datetime import datetime as dt
from datetime import timedelta

import iris
import numpy as np
from iris.tests import IrisTest

from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata import UNIQUE_ID_ATTRIBUTE
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.spot_extraction import SpotExtraction
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class Test_SpotExtraction(IrisTest):

    """Test class for the SpotExtraction tests, setting up inputs."""

    def setUp(self):
        """
        Set up cubes and sitelists for use in testing SpotExtraction.
        The envisaged scenario is an island (1) surrounded by water (0).

          Land-sea       Orography      Diagnostic

          0 0 0 0 0      0 0 0 0 0       0  1  2  3  4
          0 1 1 1 0      0 1 2 1 0       5  6  7  8  9
          0 1 1 1 0      0 2 3 2 0      10 11 12 13 14
          0 1 1 1 0      0 1 2 1 0      15 16 17 18 19
          0 0 0 0 0      0 0 0 0 0      20 21 22 23 24

        """
        # Set up diagnostic data cube and neighbour cubes.
        diagnostic_data = np.arange(25).reshape(5, 5)

        # Grid attributes must be included in diagnostic cubes so their removal
        # can be tested
        attributes = {"mosg__grid_domain": "global", "mosg__grid_type": "standard"}

        self.cell_methods = (
            iris.coords.CellMethod("maximum", coords="time", intervals="1 hour"),
        )

        time = dt(2020, 6, 15, 12)
        frt = time - timedelta(hours=6)

        diagnostic_cube_yx = set_up_variable_cube(
            diagnostic_data.T,
            name="air_temperature",
            units="K",
            attributes=attributes,
            domain_corner=(0, 0),
            grid_spacing=10,
            time=time,
            frt=frt,
        )
        diagnostic_cube_yx.cell_methods = self.cell_methods

        diagnostic_cube_xy = diagnostic_cube_yx.copy()
        enforce_coordinate_ordering(
            diagnostic_cube_xy,
            [
                diagnostic_cube_xy.coord(axis="x").name(),
                diagnostic_cube_xy.coord(axis="y").name(),
            ],
            anchor_start=False,
        )

        locations = np.array([0 + i for i in range(-3, 2)])

        # Create as int64 values
        location_points = np.array([location for location in locations])
        bounds = np.array([[location - 1, location] for location in locations])

        # Broadcast the times to a 2-dimensional grid that matches the diagnostic
        # data grid
        location_points = np.broadcast_to(location_points, (5, 5))
        bounds = np.broadcast_to(bounds, (5, 5, 2))
        # Create a 2-dimensional auxiliary coordinate
        self.location_aux_coord = iris.coords.AuxCoord(
            location_points, long_name="location", bounds=bounds, units="degrees"
        )

        diagnostic_cube_2d_time = diagnostic_cube_yx.copy()
        diagnostic_cube_2d_time.add_aux_coord(self.location_aux_coord, data_dims=(0, 1))

        expected_indices = [[0, 0], [0, 0], [2, 2], [2, 2]]
        points = [self.location_aux_coord.points[y, x] for y, x in expected_indices]
        bounds = [self.location_aux_coord.bounds[y, x] for y, x in expected_indices]
        self.expected_spot_time_coord = self.location_aux_coord.copy(
            points=points, bounds=bounds
        )

        diagnostic_cube_hash = create_coordinate_hash(diagnostic_cube_yx)

        # neighbours, each group is for a point under two methods, e.g.
        # [ 0.  0.  0.] is the nearest point to the first spot site, whilst
        # [ 1.  1. -1.] is the nearest land point to the same site.
        neighbours = np.array(
            [
                [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, -1.0, 0.0, 1.0]],
                [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [-1.0, 0.0, 0.0, 1.0]],
            ]
        )

        self.altitudes = np.array([0, 1, 3, 2])
        self.latitudes = np.array([10, 10, 20, 20])
        self.longitudes = np.array([10, 10, 20, 20])
        self.wmo_ids = np.arange(4)
        self.unique_site_id = np.arange(4)
        self.unique_site_id_key = "met_office_site_id"
        grid_attributes = ["x_index", "y_index", "vertical_displacement"]
        neighbour_methods = ["nearest", "nearest_land"]
        neighbour_cube = build_spotdata_cube(
            neighbours,
            "grid_neighbours",
            1,
            self.altitudes,
            self.latitudes,
            self.longitudes,
            self.wmo_ids,
            unique_site_id=self.unique_site_id,
            unique_site_id_key=self.unique_site_id_key,
            grid_attributes=grid_attributes,
            neighbour_methods=neighbour_methods,
        )
        neighbour_cube.attributes["model_grid_hash"] = diagnostic_cube_hash

        coordinate_cube = neighbour_cube.extract(
            iris.Constraint(neighbour_selection_method_name="nearest")
            & iris.Constraint(grid_attributes_key=["x_index", "y_index"])
        )
        coordinate_cube.data = np.rint(coordinate_cube.data).astype(int)

        self.diagnostic_cube_xy = diagnostic_cube_xy
        self.diagnostic_cube_yx = diagnostic_cube_yx
        self.diagnostic_cube_2d_time = diagnostic_cube_2d_time
        self.neighbours = neighbours
        self.neighbour_cube = neighbour_cube
        self.coordinate_cube = coordinate_cube

        self.expected_attributes = self.diagnostic_cube_xy.attributes
        for attr in MOSG_GRID_ATTRIBUTES:
            self.expected_attributes.pop(attr, None)
        self.expected_attributes["title"] = "unknown"
        self.expected_attributes["model_grid_hash"] = self.neighbour_cube.attributes[
            "model_grid_hash"
        ]


class Test__repr__(IrisTest):

    """Tests the class __repr__ function."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string with defaults."""
        plugin = SpotExtraction()
        result = str(plugin)
        msg = "<SpotExtraction: neighbour_selection_method: nearest>"
        self.assertEqual(result, msg)

    def test_non_default(self):
        """Test that the __repr__ returns the expected string with non-default
        options."""
        plugin = SpotExtraction(neighbour_selection_method="nearest_land")
        result = str(plugin)
        msg = "<SpotExtraction: neighbour_selection_method: nearest_land>"
        self.assertEqual(result, msg)


class Test_extract_coordinates(Test_SpotExtraction):

    """Test the extraction of x and y coordinate indices from a neighbour
    cube for a given neighbour_selection_method."""

    def test_nearest(self):
        """Test extraction of nearest neighbour x and y indices."""
        plugin = SpotExtraction(neighbour_selection_method="nearest")
        expected = self.neighbours[0, 0:2, :].astype(int)
        result = plugin.extract_coordinates(self.neighbour_cube)
        self.assertArrayEqual(result.data, expected)

    def test_nearest_land(self):
        """Test extraction of nearest land neighbour x and y indices."""
        plugin = SpotExtraction(neighbour_selection_method="nearest_land")
        expected = self.neighbours[1, 0:2, :].astype(int)
        result = plugin.extract_coordinates(self.neighbour_cube)
        self.assertArrayEqual(result.data, expected)

    def test_invalid_method(self):
        """Test attempt to extract neighbours found with a method that is not
        available within the neighbour cube. Raises an exception."""
        plugin = SpotExtraction(neighbour_selection_method="furthest")
        msg = 'The requested neighbour_selection_method "furthest" is not'
        with self.assertRaisesRegex(ValueError, msg):
            plugin.extract_coordinates(self.neighbour_cube)


class Test_check_for_unique_id(Test_SpotExtraction):

    """Test identification of unique site ID coordinates from coordinate
    attributes."""

    def test_unique_is_present(self):
        """Test that the IDs and coordinate name are returned if a unique site
        ID coordinate is present on the neighbour cube."""
        plugin = SpotExtraction()
        result = plugin.check_for_unique_id(self.neighbour_cube)
        self.assertArrayEqual(result[0], self.unique_site_id)
        self.assertEqual(result[1], self.unique_site_id_key)

    def test_unique_is_not_present(self):
        """Test that Nones are returned if no unique site ID coordinate is
        present on the neighbour cube."""
        self.neighbour_cube.remove_coord("met_office_site_id")
        plugin = SpotExtraction()
        result = plugin.check_for_unique_id(self.neighbour_cube)
        self.assertIsNone(result)


class Test_get_aux_coords(Test_SpotExtraction):

    """Test the extraction of scalar and non-scalar auxiliary coordinates
    from a cube."""

    def test_only_scalar_coords(self):
        """Test with an input cube containing only scalar auxiliary
        coordinates."""
        plugin = SpotExtraction()

        expected_scalar = self.diagnostic_cube_yx.aux_coords
        expected_nonscalar = []
        x_indices, y_indices = self.coordinate_cube.data
        scalar, nonscalar = plugin.get_aux_coords(
            self.diagnostic_cube_yx, x_indices, y_indices
        )
        self.assertArrayEqual(scalar, expected_scalar)
        self.assertArrayEqual(nonscalar, expected_nonscalar)

    def test_scalar_and_nonscalar_coords(self):
        """Test with an input cube containing scalar and nonscalar auxiliary
        coordinates. The returned non-scalar coordinate is a 1D representation
        of the 2D non-scalar input coordinate at spot sites."""
        plugin = SpotExtraction()

        expected_scalar = [
            coord
            for coord in self.diagnostic_cube_2d_time.aux_coords
            if coord.name() in ["time", "forecast_reference_time", "forecast_period"]
        ]
        expected_nonscalar = [self.expected_spot_time_coord]

        x_indices, y_indices = self.coordinate_cube.data

        scalar, nonscalar = plugin.get_aux_coords(
            self.diagnostic_cube_2d_time, x_indices, y_indices
        )
        print(nonscalar)
        self.assertArrayEqual(scalar, expected_scalar)
        self.assertArrayEqual(nonscalar, expected_nonscalar)

    def test_multiple_nonscalar_coords(self):
        """Test with an input cube containing multiple nonscalar auxiliary
        coordinates. The returned non-scalar coordinates are 1D representations
        of the 2D non-scalar input coordinates at spot sites."""
        plugin = SpotExtraction()

        additional_2d_crd = self.location_aux_coord.copy()
        additional_2d_crd.rename("kittens")
        self.diagnostic_cube_2d_time.add_aux_coord(additional_2d_crd, data_dims=(0, 1))
        additional_expected = self.expected_spot_time_coord.copy()
        additional_expected.rename("kittens")
        expected_nonscalar = [additional_expected, self.expected_spot_time_coord]
        x_indices, y_indices = self.coordinate_cube.data

        _, nonscalar = plugin.get_aux_coords(
            self.diagnostic_cube_2d_time, x_indices, y_indices
        )
        self.assertArrayEqual(nonscalar, expected_nonscalar)


class Test_get_coordinate_data(Test_SpotExtraction):

    """Test the extraction of data from the provided coordinates."""

    def test_coordinate_with_bounds_extraction(self):
        """Test extraction of coordinate data for a 2-dimensional auxiliary
        coordinate. In this case the coordinate has bounds."""
        plugin = SpotExtraction()

        expected_points = self.expected_spot_time_coord.points
        expected_bounds = self.expected_spot_time_coord.bounds
        x_indices, y_indices = self.coordinate_cube.data
        points, bounds = plugin.get_coordinate_data(
            self.diagnostic_cube_2d_time, x_indices, y_indices, coordinate="location"
        )
        self.assertArrayEqual(points, expected_points)
        self.assertArrayEqual(bounds, expected_bounds)

    def test_coordinate_without_bounds_extraction(self):
        """Test extraction of coordinate data for a 2-dimensional auxiliary
        coordinate. In this case the coordinate has no bounds."""
        plugin = SpotExtraction()

        expected_points = self.expected_spot_time_coord.points
        expected_bounds = None
        x_indices, y_indices = self.coordinate_cube.data

        self.diagnostic_cube_2d_time.coord("location").bounds = None
        points, bounds = plugin.get_coordinate_data(
            self.diagnostic_cube_2d_time, x_indices, y_indices, coordinate="location"
        )
        self.assertArrayEqual(points, expected_points)
        self.assertArrayEqual(bounds, expected_bounds)


class Test_build_diagnostic_cube(Test_SpotExtraction):

    """Test the building of a spot data cube with given inputs."""

    def test_building_cube(self):
        """Test that a cube is built as expected."""
        plugin = SpotExtraction()
        spot_values = np.array([0, 0, 12, 12])
        result = plugin.build_diagnostic_cube(
            self.neighbour_cube,
            self.diagnostic_cube_2d_time,
            spot_values,
            unique_site_id=self.unique_site_id,
            unique_site_id_key=self.unique_site_id_key,
            auxiliary_coords=[self.expected_spot_time_coord],
        )
        self.assertArrayEqual(result.coord("latitude").points, self.latitudes)
        self.assertArrayEqual(result.coord("longitude").points, self.longitudes)
        self.assertArrayEqual(result.coord("altitude").points, self.altitudes)
        self.assertArrayEqual(result.coord("wmo_id").points, self.wmo_ids)
        self.assertArrayEqual(
            result.coord(self.unique_site_id_key).points, self.unique_site_id
        )
        self.assertArrayEqual(
            result.coord("location").points, self.expected_spot_time_coord.points
        )
        self.assertArrayEqual(result.data, spot_values)


class Test_process(Test_SpotExtraction):

    """Test the process method which extracts data and builds cubes with
    metadata added."""

    def test_unmatched_cube_error(self):
        """Test that an error is raised if the neighbour cube and diagnostic
        cube do not have matching grids."""
        self.neighbour_cube.attributes["model_grid_hash"] = "123"
        plugin = SpotExtraction()
        msg = (
            "Cubes do not share or originate from the same grid, so cannot "
            "be used together."
        )
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)

    def test_returned_cube_nearest(self):
        """Test that data within the returned cube is as expected for the
        nearest neigbours."""
        plugin = SpotExtraction()
        expected = [0, 0, 12, 12]
        result = plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.name(), self.diagnostic_cube_xy.name())
        self.assertEqual(result.units, self.diagnostic_cube_xy.units)
        self.assertArrayEqual(result.coord("latitude").points, self.latitudes)
        self.assertArrayEqual(result.coord("longitude").points, self.longitudes)
        self.assertDictEqual(result.attributes, self.expected_attributes)

    def test_returned_cube_nearest_land(self):
        """Test that data within the returned cube is as expected for the
        nearest land neighbours."""
        plugin = SpotExtraction(neighbour_selection_method="nearest_land")
        expected = [6, 6, 12, 12]
        result = plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.name(), self.diagnostic_cube_xy.name())
        self.assertEqual(result.units, self.diagnostic_cube_xy.units)
        self.assertArrayEqual(result.coord("latitude").points, self.latitudes)
        self.assertArrayEqual(result.coord("longitude").points, self.longitudes)
        self.assertDictEqual(result.attributes, self.expected_attributes)

    def test_new_title(self):
        """Test title is updated as expected"""
        expected_attributes = self.expected_attributes
        expected_attributes["title"] = "IMPROVER Spot Forecast"
        plugin = SpotExtraction(neighbour_selection_method="nearest_land")
        result = plugin.process(
            self.neighbour_cube,
            self.diagnostic_cube_xy,
            new_title="IMPROVER Spot Forecast",
        )
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_cube_with_leading_dimensions(self):
        """Test that a cube with a leading dimension such as realization or
        probability results in a spotdata cube with the same leading
        dimension."""
        realization0 = iris.coords.DimCoord([0], standard_name="realization", units=1)
        realization1 = iris.coords.DimCoord([1], standard_name="realization", units=1)

        cube0 = self.diagnostic_cube_xy.copy()
        cube1 = self.diagnostic_cube_xy.copy()
        cube0.add_aux_coord(realization0)
        cube1.add_aux_coord(realization1)
        cubes = iris.cube.CubeList([cube0, cube1])
        cube = cubes.merge_cube()

        plugin = SpotExtraction()
        expected = [[0, 0, 12, 12], [0, 0, 12, 12]]
        expected_coord = iris.coords.DimCoord(
            [0, 1], standard_name="realization", units=1
        )
        result = plugin.process(self.neighbour_cube, cube)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.name(), cube.name())
        self.assertEqual(result.units, cube.units)
        self.assertArrayEqual(result.coord("latitude").points, self.latitudes)
        self.assertArrayEqual(result.coord("longitude").points, self.longitudes)
        self.assertEqual(result.coord("realization"), expected_coord)
        self.assertDictEqual(result.attributes, self.expected_attributes)

    def test_cell_methods(self):
        """Test cell methods from the gridded input cube are retained on the
        spotdata cube."""
        plugin = SpotExtraction(neighbour_selection_method="nearest_land")
        result = plugin.process(
            self.neighbour_cube,
            self.diagnostic_cube_xy,
            new_title="IMPROVER Spot Forecast",
        )
        self.assertEqual(result.cell_methods, self.cell_methods)

    def test_2d_aux_coords(self):
        """Test 2D auxiliary coordinates from the gridded input cube are
        retained as 1D coordinates associated with the spot-index on the
        spotdata cube."""
        plugin = SpotExtraction()
        print(self.diagnostic_cube_2d_time)
        result = plugin.process(
            self.neighbour_cube,
            self.diagnostic_cube_2d_time,
            new_title="IMPROVER Spot Forecast",
        )
        self.assertEqual(result.coord("location"), self.expected_spot_time_coord)

    def test_removal_of_internal_metadata(self):
        """Test that internal metadata used to identify the unique id coordinate
        is removed in the resulting spot diagnostic cube."""
        plugin = SpotExtraction()
        result = plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)
        self.assertNotIn(
            UNIQUE_ID_ATTRIBUTE,
            [att for att in result.coord(self.unique_site_id_key).attributes],
        )

    def test_yx_ordered_cube(self):
        """Test extraction of diagnostic data that is natively ordered yx."""
        plugin = SpotExtraction()
        expected = [0, 0, 12, 12]
        result = plugin.process(self.coordinate_cube, self.diagnostic_cube_yx)
        self.assertArrayEqual(result.data, expected)


if __name__ == "__main__":
    unittest.main()

# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for SpotLapseRateAdjust class"""

import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.constants import DALR
from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.apply_lapse_rate import SpotLapseRateAdjust
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    construct_scalar_time_coords,
    construct_yx_coords,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.temporal import iris_time_to_datetime


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
        self.lapse_rate_cube = set_up_variable_cube(
            lapse_rate_data,
            name="air_temperature_lapse_rate",
            units="K m-1",
            spatial_grid="equalarea",
        )
        self.lapse_rate_cube = add_coordinate(
            incube=self.lapse_rate_cube, coord_points=[1.5], coord_name="height"
        )
        diagnostic_cube_hash = create_coordinate_hash(self.lapse_rate_cube)

        # Set up neighbour and spot diagnostic cubes
        y_coord, x_coord = construct_yx_coords(3, 3, "equalarea")
        y_coord = y_coord.points
        x_coord = x_coord.points

        # neighbours, each group is for a point under two methods, e.g.
        # [ 0.  0.  0.] is the nearest point to the first spot site, whilst
        # [ 1.  1. -1.] is the nearest point with minimum height difference.
        neighbours = np.array(
            [
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [2.0, 0.0, -1.0]],
                [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 0.0, -1.0]],
            ]
        )
        altitudes = np.array([3, 4, 0])
        latitudes = np.array([y_coord[0], y_coord[1], y_coord[2]])
        longitudes = np.array([x_coord[0], x_coord[1], x_coord[2]])
        wmo_ids = np.arange(3)
        grid_attributes = ["x_index", "y_index", "vertical_displacement"]
        neighbour_methods = ["nearest", "nearest_minimum_dz"]
        self.neighbour_cube = build_spotdata_cube(
            neighbours,
            "grid_neighbours",
            1,
            altitudes,
            latitudes,
            longitudes,
            wmo_ids,
            grid_attributes=grid_attributes,
            neighbour_methods=neighbour_methods,
        )
        self.neighbour_cube.attributes["model_grid_hash"] = diagnostic_cube_hash

        (time,) = iris_time_to_datetime(self.lapse_rate_cube.coord("time"))

        (frt,) = iris_time_to_datetime(
            self.lapse_rate_cube.coord("forecast_reference_time")
        )
        time_bounds = None

        time_coords = construct_scalar_time_coords(time, time_bounds, frt)
        time_coords = [item[0] for item in time_coords]

        # This temperature cube is set up with the spot sites having obtained
        # their temperature values from the nearest grid sites.
        temperatures_nearest = np.array([280, 270, 280])
        self.spot_temperature_nearest = build_spotdata_cube(
            temperatures_nearest,
            "air_temperature",
            "K",
            altitudes,
            latitudes,
            longitudes,
            wmo_ids,
            scalar_coords=time_coords,
        )
        self.spot_temperature_nearest = add_coordinate(
            incube=self.spot_temperature_nearest,
            coord_points=[1.5],
            coord_name="height",
        )

        self.spot_temperature_nearest.attributes[
            "model_grid_hash"
        ] = diagnostic_cube_hash

        # This temperature cube is set up with the spot sites having obtained
        # their temperature values from the nearest minimum vertical
        # displacement grid sites. The only difference here is for site 0, which
        # now gets its temperature from Bb (see doc-string above).
        temperatures_mindz = np.array([270, 270, 280])
        self.spot_temperature_mindz = build_spotdata_cube(
            temperatures_mindz,
            "air_temperature",
            "K",
            altitudes,
            latitudes,
            longitudes,
            wmo_ids,
            scalar_coords=time_coords,
        )
        self.spot_temperature_mindz = add_coordinate(
            incube=self.spot_temperature_mindz, coord_points=[1.5], coord_name="height"
        )
        self.spot_temperature_mindz.attributes["model_grid_hash"] = diagnostic_cube_hash


class Test_process(Test_SpotLapseRateAdjust):

    """Tests the class process method."""

    def test_basic(self):
        """Test that the plugin returns a cube which is unchanged except for
        data values."""

        plugin = SpotLapseRateAdjust()
        result = plugin(
            self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
        )

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), self.spot_temperature_nearest.name())
        self.assertEqual(result.units, self.spot_temperature_nearest.units)
        self.assertEqual(result.coords(), self.spot_temperature_nearest.coords())

    def test_nearest_neighbour_method(self):
        """Test that the plugin modifies temperatures as expected for both air
        temperature and feels like temperature cubes, using the vertical
        displacements taken from the nearest neighbour method in the neighbour cube."""

        plugin = SpotLapseRateAdjust()
        expected = np.array([280 + (2 * DALR), 270, 280 - DALR]).astype(np.float32)

        # Air temperature cube
        result = plugin(
            self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
        )
        self.assertArrayEqual(result.data, expected)

        # Feels like temperature cube
        self.spot_temperature_nearest.rename("feels_like_temperature")
        result = plugin(
            self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
        )
        self.assertArrayEqual(result.data, expected)

    def test_different_neighbour_method(self):
        """Test that the plugin uses the correct vertical displacements when
        a different neighbour method is set. This should result in these
        different values being chosen from the neighbour cube.

        In this case site 0 has a displacement of -1 from the chosen grid site,
        but the lapse rate at that site is 2*DALR, so the change below is by
        2*DALR, compared with site 2 which has the same displacement, but for
        which the lapse rate is just the DALR."""

        plugin = SpotLapseRateAdjust(neighbour_selection_method="nearest_minimum_dz")
        expected = np.array([270 - (2 * DALR), 270, 280 - DALR]).astype(np.float32)

        result = plugin(
            self.spot_temperature_mindz, self.neighbour_cube, self.lapse_rate_cube
        )
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
        expected = np.array([280 + (2 * DALR), 270, 280 - DALR]).astype(np.float32)
        enforce_coordinate_ordering(
            self.lapse_rate_cube, ["projection_x_coordinate", "projection_y_coordinate"]
        )

        result = plugin(
            self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
        )
        self.assertArrayEqual(result.data, expected)

    def test_probability_cube(self):
        """Ensure that the plugin exits with value error if the spot data cube
        is in probability space. """

        diagnostic_cube_hash = create_coordinate_hash(self.lapse_rate_cube)
        data = np.ones((3, 3, 3), dtype=np.float32)
        threshold_points = np.array([276, 277, 278], dtype=np.float32)
        probability_cube = set_up_probability_cube(
            data, threshold_points, spp__relative_to_threshold="above"
        )

        probability_cube.attributes["model_grid_hash"] = diagnostic_cube_hash

        plugin = SpotLapseRateAdjust()
        msg = (
            "Input cube has a probability coordinate which cannot be lapse "
            "rate adjusted. Input data should be in percentile or "
            "deterministic space only."
        )

        with self.assertRaisesRegex(ValueError, msg):
            plugin(probability_cube, self.neighbour_cube, self.lapse_rate_cube)

    def test_different_dimensions(self):
        """Test that the lapse rate cube can be broadcast to the same dimensions
        as the spot data cube."""

        data = np.array([25, 50, 75], dtype=np.float32)
        spot_temperature_new_coord = add_coordinate(
            self.spot_temperature_nearest, data, "percentile", "%"
        )
        plugin = SpotLapseRateAdjust()
        result = plugin(
            spot_temperature_new_coord, self.neighbour_cube, self.lapse_rate_cube
        )
        expected = np.array([280 + (2 * DALR), 270, 280 - DALR]).astype(np.float32)
        for slice in result.data:
            self.assertArrayEqual(slice, expected)

    def test_using_fixed_lapse_rates(self):
        """Test that the data is as expected when using fixed lapse rates.
        This includes a lapse rate of 0, which leaves the data unchanged."""

        for lr in [0, 0.5 * DALR, DALR]:
            expected = np.array([280 + (2 * lr), 270, 280 - lr]).astype(np.float32)
            plugin = SpotLapseRateAdjust(fixed_lapse_rate=lr)
            result = plugin(self.spot_temperature_nearest, self.neighbour_cube)
            self.assertArrayEqual(result.data, expected)

    def test_diagnostic_name(self):
        """Test that appropriate error is raised when the input cube has a
        diagnostic name that is not air temperature or feels like temperature."""

        self.spot_temperature_nearest.rename("something")
        plugin = SpotLapseRateAdjust()
        msg = (
            "The diagnostic being processed is not air temperature or feels "
            "like temperature and therefore cannot be adjusted."
        )

        with self.assertRaisesRegex(ValueError, msg):
            plugin(
                self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
            )

    def test_lapse_rate_name(self):
        """Test that appropriate error is called when the input lapse rate cube
        has a diagnostic name that is not air temperature lapse rate."""

        self.lapse_rate_cube.rename("something")
        plugin = SpotLapseRateAdjust()
        msg = (
            "A cube has been provided as a lapse rate cube but does "
            "not have the expected name air_temperature_lapse_rate: "
            "{}".format(self.lapse_rate_cube.name())
        )

        with self.assertRaisesRegex(ValueError, msg):
            plugin(
                self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
            )

    def test_height_coord(self):
        """Test that the appropriate error is raised when the input cube has
        no single valued height coordinate."""

        self.lapse_rate_cube.remove_coord("height")
        plugin = SpotLapseRateAdjust()
        msg = (
            "Lapse rate cube does not contain a single valued height "
            "coordinate. This is required to ensure it is applied to "
            "equivalent temperature data."
        )

        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin(
                self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
            )

    def test_height_coords_match(self):
        """Test the the appropriate error is called when the input temperature
        cube and the lapse rate cube have differing height coordinates"""

        self.spot_temperature_nearest.coord("height").points = [4]

        plugin = SpotLapseRateAdjust()
        msg = (
            "A lapse rate cube was provided, but the height of the "
            "temperature data does not match that of the data used "
            "to calculate the lapse rates. As such the temperatures "
            "were not adjusted with the lapse rates."
        )
        with self.assertRaisesRegex(ValueError, msg):
            plugin(
                self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
            )

    def test_two_lapse_rate_sources(self):
        """Test that an appropriate error is raised when both a gridded and
        fixed lapse rate are provided."""

        plugin = SpotLapseRateAdjust(fixed_lapse_rate=-6e-3)
        msg = (
            "Both a lapse rate cube and a fixed lapse rate have been provided. "
            "Provide only one source of lapse rate information."
        )

        with self.assertRaisesRegex(ValueError, msg):
            plugin(
                self.spot_temperature_nearest, self.neighbour_cube, self.lapse_rate_cube
            )

    def test_no_lapse_rate_sources(self):
        """Test that an appropriate error is raised when no lapse rate source
        is provided."""

        plugin = SpotLapseRateAdjust()
        msg = (
            "No lapse rate cube has been provided, and no fixed lapse rate "
            "has been set. Provide one or other."
        )

        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.spot_temperature_nearest, self.neighbour_cube)


if __name__ == "__main__":
    unittest.main()

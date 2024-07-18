# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the build_spotdata_cube function"""

import unittest
from datetime import datetime

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.tests import IrisTest

from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import construct_scalar_time_coords
from improver.utilities.round import round_close


class Test_build_spotdata_cube(IrisTest):
    """Tests for the build_spotdata_cube function"""

    def setUp(self):
        """Set up some auxiliary coordinate points for re-use"""
        self.altitude = np.array([256.5, 359.1, 301.8, 406.2])
        self.latitude = np.linspace(58.0, 59.5, 4)
        self.longitude = np.linspace(-0.25, 0.5, 4)
        self.wmo_id = ["03854", "03962", "03142", "03331"]
        self.unique_site_id = [id.zfill(8) for id in self.wmo_id]

        self.neighbour_methods = ["nearest", "nearest_land"]
        self.grid_attributes = ["x_index", "y_index", "dz"]
        self.args = (
            "air_temperature",
            "degC",
            self.altitude,
            self.latitude,
            self.longitude,
            self.wmo_id,
        )

    def test_scalar(self):
        """Test output for a single site"""
        result = build_spotdata_cube(
            1.6, "air_temperature", "degC", 10.0, 59.5, 1.3, "03854"
        )

        # check result type
        self.assertIsInstance(result, iris.cube.Cube)

        # check data
        self.assertArrayAlmostEqual(result.data, np.array([1.6]))
        self.assertEqual(result.name(), "air_temperature")
        self.assertEqual(result.units, "degC")

        # check coordinate values and units
        self.assertEqual(result.coord("spot_index").points[0], 0)
        self.assertAlmostEqual(result.coord("altitude").points[0], 10.0)
        self.assertEqual(result.coord("altitude").units, "m")
        self.assertAlmostEqual(result.coord("latitude").points[0], 59.5)
        self.assertEqual(result.coord("latitude").units, "degrees")
        self.assertAlmostEqual(result.coord("longitude").points[0], 1.3)
        self.assertEqual(result.coord("longitude").units, "degrees")
        self.assertEqual(result.coord("wmo_id").points[0], "03854")

    def test_site_list(self):
        """Test output for a list of sites"""
        data = np.array([1.6, 1.3, 1.4, 1.1])
        result = build_spotdata_cube(data, *self.args)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayAlmostEqual(result.coord("altitude").points, self.altitude)
        self.assertArrayAlmostEqual(result.coord("latitude").points, self.latitude)
        self.assertArrayAlmostEqual(result.coord("longitude").points, self.longitude)
        self.assertArrayEqual(result.coord("wmo_id").points, self.wmo_id)

    def test_site_list_with_unique_id_coordinate(self):
        """Test output for a list of sites with a unique_id_coordinate."""
        data = np.array([1.6, 1.3, 1.4, 1.1])
        result = build_spotdata_cube(
            data,
            *self.args,
            unique_site_id=self.unique_site_id,
            unique_site_id_key="met_office_site_id",
        )

        self.assertArrayEqual(
            result.coord("met_office_site_id").points, self.unique_site_id
        )
        self.assertEqual(
            result.coord("met_office_site_id").attributes["unique_site_identifier"],
            "true",
        )

    def test_site_list_with_unique_id_coordinate_missing_name(self):
        """Test an error is raised if a unique_id_coordinate is provided but
        no name for the resulting coordinate."""
        data = np.array([1.6, 1.3, 1.4, 1.1])
        msg = "A unique_site_id_key must be provided"
        with self.assertRaisesRegex(ValueError, msg):
            build_spotdata_cube(
                data, *self.args, unique_site_id=self.unique_site_id,
            )

    def test_neighbour_method(self):
        """Test output where neighbour_methods is populated"""
        data = np.array([[1.6, 1.3, 1.4, 1.1], [1.7, 1.5, 1.4, 1.3]])

        result = build_spotdata_cube(
            data, *self.args, neighbour_methods=self.neighbour_methods
        )

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims("neighbour_selection_method")[0], 0)
        self.assertArrayEqual(
            result.coord("neighbour_selection_method").points, np.arange(2)
        )
        self.assertArrayEqual(
            result.coord("neighbour_selection_method_name").points,
            self.neighbour_methods,
        )

    def test_grid_attributes(self):
        """Test output where grid_attributes is populated"""
        data = np.array(
            [[1.6, 1.3, 1.4, 1.1], [1.7, 1.5, 1.4, 1.3], [1.8, 1.5, 1.5, 1.4]]
        )

        result = build_spotdata_cube(
            data, *self.args, grid_attributes=self.grid_attributes,
        )

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims("grid_attributes")[0], 0)
        self.assertArrayEqual(result.coord("grid_attributes").points, np.arange(3))
        self.assertArrayEqual(
            result.coord("grid_attributes_key").points, self.grid_attributes
        )

    def test_3d_spot_cube(self):
        """Test output with two extra dimensions"""
        data = np.ones((2, 3, 4), dtype=np.float32)
        result = build_spotdata_cube(
            data,
            *self.args,
            neighbour_methods=self.neighbour_methods,
            grid_attributes=self.grid_attributes,
        )

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims("neighbour_selection_method")[0], 0)
        self.assertEqual(result.coord_dims("grid_attributes")[0], 1)

    def test_3d_spot_cube_with_unequal_length_coordinates(self):
        """Test error is raised if coordinates lengths do not match data
        dimensions."""

        data = np.ones((4, 2, 2), dtype=np.float32)

        msg = "Unequal lengths"
        with self.assertRaisesRegex(ValueError, msg):
            build_spotdata_cube(
                data,
                *self.args,
                neighbour_methods=self.neighbour_methods,
                grid_attributes=self.grid_attributes,
            )

    def test_3d_spot_cube_for_time(self):
        """Test output with two extra dimensions, one of which is time with
        forecast_period as an auxiliary coordinate"""
        data = np.ones((3, 2, 4), dtype=np.float32)
        time_spec = TIME_COORDS["time"]
        time_units = Unit(time_spec.units)
        time_as_dt = [datetime(2021, 12, 25, 12, 0), datetime(2021, 12, 25, 12, 1)]
        time_points = round_close(
            np.array([time_units.date2num(t) for t in time_as_dt]),
            dtype=time_spec.dtype,
        )
        time_coord = DimCoord(time_points, units=time_units, standard_name="time")

        fp_spec = TIME_COORDS["forecast_period"]
        fp_units = Unit(fp_spec.units)
        fp_points = np.array([0, 3600], dtype=fp_spec.dtype)
        fp_coord = AuxCoord(fp_points, units=fp_units, standard_name="forecast_period")

        result = build_spotdata_cube(
            data,
            *self.args,
            grid_attributes=self.grid_attributes,
            additional_dims=[time_coord],
            additional_dims_aux=[[fp_coord]],
        )

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims("grid_attributes")[0], 0)
        self.assertEqual(result.coord_dims("time")[0], 1)
        self.assertEqual(result.coord_dims("forecast_period")[0], 1)

    def test_scalar_coords(self):
        """Test additional scalar coordinates"""
        [(time_coord, _), (frt_coord, _), (fp_coord, _)] = construct_scalar_time_coords(
            datetime(2015, 11, 23, 4, 30), None, datetime(2015, 11, 22, 22, 30)
        )

        data = np.ones((2, 4), dtype=np.float32)
        result = build_spotdata_cube(
            data,
            *self.args,
            scalar_coords=[time_coord, frt_coord, fp_coord],
            neighbour_methods=self.neighbour_methods,
        )

        self.assertEqual(result.coord("time").points[0], time_coord.points[0])
        self.assertEqual(
            result.coord("forecast_reference_time").points[0], frt_coord.points[0]
        )
        self.assertEqual(result.coord("forecast_period").points[0], fp_coord.points[0])

    def test_non_scalar_coords(self):
        """Test additional non-scalar coordinates, specifically multi-dimensional
        auxiliary coordinates that have been reshaped into 1-dimensional coordinates
        that should be associated with the spot-index coordinate."""
        times = np.array([datetime(2015, 11, 23, i, 0) for i in range(0, 4)])
        time_coord = iris.coords.AuxCoord(times, "time")

        data = np.ones((2, 4), dtype=np.float32)
        result = build_spotdata_cube(
            data,
            *self.args,
            auxiliary_coords=[time_coord],
            neighbour_methods=self.neighbour_methods,
        )
        self.assertArrayEqual(result.coord("time").points, times)
        self.assertEqual(result.coord_dims("time"), result.coord_dims("spot_index"))

    def test_renaming_to_set_standard_name(self):
        """Test that CF standard names are set as such in the returned cube,
        whilst non-standard names remain as the long_name."""
        standard_name_cube = build_spotdata_cube(
            1.6, "air_temperature", "degC", 10.0, 59.5, 1.3, "03854"
        )
        non_standard_name_cube = build_spotdata_cube(
            1.6, "toast_temperature", "degC", 10.0, 59.5, 1.3, "03854"
        )

        self.assertEqual(standard_name_cube.standard_name, "air_temperature")
        self.assertEqual(standard_name_cube.long_name, None)
        self.assertEqual(non_standard_name_cube.standard_name, None)
        self.assertEqual(non_standard_name_cube.long_name, "toast_temperature")


if __name__ == "__main__":
    unittest.main()

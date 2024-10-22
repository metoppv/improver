# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the threshold.LatitudeThreshold plugin."""

import unittest

import numpy as np
from iris.coords import AuxCoord, CellMethod
from iris.cube import Cube
from iris.tests import IrisTest

from improver.lightning import latitude_to_threshold
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.threshold import LatitudeDependentThreshold as Threshold


class Test__init(IrisTest):
    """Test exceptions from the __init__ method"""

    def test_not_callable(self):
        """Test a useful message is given if the specified threshold is not callable"""
        msg = "Threshold must be callable"
        with self.assertRaisesRegex(TypeError, msg):
            Threshold(None)


class Test__add_latitude_threshold_coord(IrisTest):
    """Test the _add_latitude_threshold_coord method"""

    def setUp(self):
        """Set up a cube and plugin for testing."""
        self.cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        self.plugin = Threshold(latitude_to_threshold)
        self.plugin.threshold_coord_name = self.cube.name()
        self.thresholds = np.array(
            latitude_to_threshold(
                self.cube.coord("latitude").points, midlatitude=1.0, tropics=3.0
            )
        )

    def test_basic(self):
        """Test a 1D threshold coordinate is created"""
        expected_points = self.thresholds.copy()
        self.plugin._add_latitude_threshold_coord(self.cube, self.thresholds)
        self.assertEqual(self.cube.ndim, 2)
        self.assertIn(
            "air_temperature",
            [coord.standard_name for coord in self.cube.coords(dim_coords=False)],
        )
        threshold_coord = self.cube.coord("air_temperature")
        self.assertEqual(threshold_coord.var_name, "threshold")
        self.assertArrayAlmostEqual(threshold_coord.points, expected_points)
        self.assertEqual(threshold_coord.units, self.cube.units)
        self.assertEqual(
            self.cube.coord_dims("latitude"), self.cube.coord_dims("air_temperature")
        )

    def test_long_name(self):
        """Test coordinate is created with non-standard diagnostic name"""
        self.cube.rename("sky_temperature")
        self.plugin.threshold_coord_name = self.cube.name()
        self.plugin._add_latitude_threshold_coord(self.cube, self.thresholds)
        self.assertIn(
            "sky_temperature",
            [coord.long_name for coord in self.cube.coords(dim_coords=False)],
        )

    def test_value_error(self):
        """Test method catches ValueErrors unrelated to name, by passing it a
        2D array of values where 1D is required"""
        with self.assertRaises(ValueError):
            self.plugin._add_latitude_threshold_coord(self.cube, np.ones((3, 3)))


class Test_process(IrisTest):
    """Test the thresholding plugin."""

    def setUp(self):
        """Create a cube with a constant non-zero value spanning latitudes from -60 to +60."""
        attributes = {"title": "UKV Model Forecast"}
        data = np.full((7, 3), fill_value=0.5, dtype=np.float32)
        self.cube = set_up_variable_cube(
            data,
            name="precipitation_amount",
            units="m",
            attributes=attributes,
            standard_grid_metadata="uk_det",
            domain_corner=(-60, 0),
            x_grid_spacing=20,
            y_grid_spacing=20,
        )

        self.masked_cube = self.cube.copy()
        data = np.copy(self.cube.data)
        mask = np.zeros_like(data)
        data[0][0] = -32768.0
        mask[0][0] = 1
        self.masked_cube.data = np.ma.MaskedArray(data, mask=mask)

        self.plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=1e-6, tropics=1.0)
        )

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        result = self.plugin(self.cube)
        self.assertIsInstance(result, Cube)

    def test_title_updated(self):
        """Test title is updated"""
        expected_title = "Post-Processed UKV Model Forecast"
        result = self.plugin(self.cube)
        self.assertEqual(result.attributes["title"], expected_title)

    def test_metadata_changes(self):
        """Test the metadata altering functionality"""
        plugin = self.plugin
        result = plugin(self.cube)
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "greater_than"
        expected_units = 1
        expected_points = plugin.threshold_function(self.cube.coord("latitude").points)
        expected_coord = AuxCoord(
            np.array(expected_points, dtype=np.float32),
            standard_name=self.cube.name(),
            var_name="threshold",
            units=self.cube.units,
            attributes={"spp__relative_to_threshold": "greater_than"},
        )
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertEqual(result.units, expected_units)
        self.assertEqual(result.coord(self.cube.name()), expected_coord)

    def test_threshold(self):
        """Test the latitude-dependent threshold functionality.
        We expect hits in mid-latitudes and not in the tropics."""
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[2:-2][:] = 0
        result = self.plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_masked_array(self):
        """Test masked array are handled correctly.
        Masked values are preserved following thresholding."""
        result = self.plugin(self.masked_cube)
        expected_result_array = np.ones_like(self.masked_cube.data)
        expected_result_array[2:-2][:] = 0
        self.assertArrayAlmostEqual(result.data.data, expected_result_array)
        self.assertArrayEqual(result.data.mask, self.masked_cube.data.mask)

    def test_threshold_negative(self):
        """Repeat the test with negative numbers when the threshold is negative."""
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2:-2][:] = 1
        self.cube.data = 0 - self.cube.data
        plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=-1e-6, tropics=-1.0)
        )
        result = plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_gt(self):
        """Test equal-to values when we are in > threshold mode."""
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[3][:] = 0
        plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=1e-6, tropics=0.5),
            comparison_operator=">",
        )
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "greater_than"

        result = plugin(self.cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_ge(self):
        """Test equal-to values when we are in >= threshold mode."""
        expected_result_array = np.ones_like(self.cube.data)
        plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=1e-6, tropics=0.5),
            comparison_operator=">=",
        )
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "greater_than_or_equal_to"

        result = plugin(self.cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_lt(self):
        """Test equal-to values when we are in < threshold mode."""
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[1:-1, :] = 1
        plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=0.5, tropics=1.0),
            comparison_operator="<",
        )
        name = "probability_of_{}_below_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "less_than"
        result = plugin(self.cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_le(self):
        """Test equal-to values when we are in le threshold mode."""
        expected_result_array = np.ones_like(self.cube.data)
        plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=0.5, tropics=1.0),
            comparison_operator="<=",
        )
        name = "probability_of_{}_below_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "less_than_or_equal_to"
        result = plugin(self.cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_unit_conversion(self):
        """Test data are correctly thresholded when the threshold (mm) is given in
        units different from that of the input cube (m)."""
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[3][:] = 0
        plugin = Threshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=1e-3, tropics=500.0),
            threshold_units="mm",
        )
        result = plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)
        expected_points = (
            plugin.threshold_function(self.cube.coord("latitude").points) / 1000
        )
        expected_coord = AuxCoord(
            np.array(expected_points, dtype=np.float32),
            standard_name=self.cube.name(),
            var_name="threshold",
            units=self.cube.units,
            attributes={"spp__relative_to_threshold": "greater_than"},
        )
        self.assertEqual(result.coord(self.cube.name()), expected_coord)

    def test_threshold_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        self.cube.data[2][2] = np.NAN
        msg = "NaN detected in input cube data"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(self.cube)

    def test_cell_method_updates(self):
        """Test plugin adds correct information to cell methods"""
        self.cube.add_cell_method(CellMethod("max", coords="time"))
        result = self.plugin(self.cube)
        (cell_method,) = result.cell_methods
        self.assertEqual(cell_method.method, "max")
        self.assertEqual(cell_method.coord_names, ("time",))
        self.assertEqual(cell_method.comments, ("of precipitation_amount",))


if __name__ == "__main__":
    unittest.main()

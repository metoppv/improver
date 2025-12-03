# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the standardise.StandardiseMetadata plugin."""

import unittest
from datetime import datetime
from unittest.mock import patch, sentinel

import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord

from improver.standardise import StandardiseMetadata
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver_tests import ImproverTest


class HaltExecution(Exception):
    pass


@patch("improver.standardise.as_cube")
def test_as_cubelist_called(mock_as_cube):
    mock_as_cube.side_effect = HaltExecution
    try:
        StandardiseMetadata(
            new_name=sentinel.new_name,
            new_units=sentinel.new_units,
            coords_to_remove=sentinel.coords_to_remove,
            coord_modification=sentinel.coord_modification,
            attributes_dict=sentinel.attributes_dict,
        )(sentinel.cube)
    except HaltExecution:
        pass
    mock_as_cube.assert_called_once_with(sentinel.cube)


class Test_process(ImproverTest):
    """Test the process method"""

    def setUp(self):
        """Set up input cube"""
        self.cube = set_up_variable_cube(
            282 * np.ones((5, 5), dtype=np.float32),
            spatial_grid="latlon",
            standard_grid_metadata="gl_det",
            time=datetime(2019, 10, 11),
            time_bounds=[datetime(2019, 10, 10, 23), datetime(2019, 10, 11)],
            frt=datetime(2019, 10, 10, 18),
        )

    def test_null(self):
        """Test process method with default arguments returns an unchanged
        cube"""
        result = StandardiseMetadata().process(self.cube.copy())
        self.assertIsInstance(result, iris.cube.Cube)
        np.testing.assert_array_almost_equal(result.data, self.cube.data)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_standardise_time_coords(self):
        """Test incorrect time-type coordinates are cast to the correct
        datatypes and units"""
        for coord in ["time", "forecast_period"]:
            self.cube.coord(coord).points = self.cube.coord(coord).points.astype(
                np.float64
            )
            self.cube.coord(coord).bounds = self.cube.coord(coord).bounds.astype(
                np.float64
            )
        self.cube.coord("forecast_period").convert_units("hours")
        result = StandardiseMetadata().process(self.cube)
        self.assertEqual(result.coord("forecast_period").units, "seconds")
        self.assertEqual(result.coord("forecast_period").points.dtype, np.int32)
        self.assertEqual(result.coord("forecast_period").bounds.dtype, np.int32)
        self.assertEqual(result.coord("time").points.dtype, np.int64)
        self.assertEqual(result.coord("time").bounds.dtype, np.int64)

    def test_standardise_time_coords_missing_fp(self):
        """Test a missing time-type coordinate does not cause an error when
        standardisation is required"""
        self.cube.coord("time").points = self.cube.coord("time").points.astype(
            np.float64
        )
        self.cube.remove_coord("forecast_period")
        result = StandardiseMetadata().process(self.cube)
        self.assertEqual(result.coord("time").points.dtype, np.int64)

    def test_collapse_scalar_dimensions(self):
        """Test scalar dimension is collapsed"""
        cube = iris.util.new_axis(self.cube, "time")
        result = StandardiseMetadata().process(cube)
        dim_coord_names = [coord.name() for coord in result.coords(dim_coords=True)]
        aux_coord_names = [coord.name() for coord in result.coords(dim_coords=False)]
        self.assertSequenceEqual(result.shape, (5, 5))
        self.assertNotIn("time", dim_coord_names)
        self.assertIn("time", aux_coord_names)

    def test_realization_not_collapsed(self):
        """Test scalar realization coordinate is preserved"""
        realization = AuxCoord([1], "realization")
        self.cube.add_aux_coord(realization)
        cube = iris.util.new_axis(self.cube, "realization")
        result = StandardiseMetadata().process(cube)
        dim_coord_names = [coord.name() for coord in result.coords(dim_coords=True)]
        self.assertSequenceEqual(result.shape, (1, 5, 5))
        self.assertIn("realization", dim_coord_names)

    def test_metadata_changes(self):
        """Test changes to cube name, coordinates and attributes without
        regridding"""
        new_name = "regridded_air_temperature"
        attribute_changes = {
            "institution": "Met Office",
            "mosg__grid_version": "remove",
        }
        expected_attributes = {
            "mosg__grid_domain": "global",
            "mosg__grid_type": "standard",
            "mosg__model_configuration": "gl_det",
            "institution": "Met Office",
        }
        expected_data = self.cube.data.copy() - 273.15
        # Add scalar height coordinate
        self.cube.add_aux_coord(DimCoord([1.5], standard_name="height", units="m"))
        # Modifier for scalar height coordinate
        coord_modification = {"height": 2.0}

        plugin = StandardiseMetadata(
            new_name=new_name,
            new_units="degC",
            coords_to_remove=["forecast_period"],
            coord_modification=coord_modification,
            attributes_dict=attribute_changes,
        )
        result = plugin.process(self.cube)
        self.assertEqual(result.name(), new_name)
        self.assertEqual(result.units, "degC")
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=5)
        self.assertEqual(result.coord("height").points, 2.0)
        self.assertDictEqual(result.attributes, expected_attributes)
        self.assertNotIn("forecast_period", [coord.name() for coord in result.coords()])

    def test_attempt_modify_dimension_coord(self):
        """Test that an exception is raised if the coord_modification targets
        a dimension coordinate."""

        coord_modification = {"latitude": [0.1, 1.2, 3.4, 5.6, 7]}
        msg = "Modifying dimension coordinate values is not allowed "
        plugin = StandardiseMetadata(coord_modification=coord_modification)

        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube)

    def test_attempt_modify_multi_valued_coord(self):
        """Test that an exception is raised if the coord_modification is used
        to modify a multi-valued coordinate which is not a dimension
        coordinate and is therefore missed by the previous test."""

        cube = self.cube.copy()
        kitten_coord = AuxCoord([1, 2, 3, 4, 5], long_name="kittens", units=1)
        cube.add_aux_coord(kitten_coord, 0)

        coord_modification = {"kittens": [2, 3, 4, 5, 6]}
        msg = "Modifying multi-valued coordinates is not allowed."

        plugin = StandardiseMetadata(coord_modification=coord_modification)
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cube)

    def test_attempt_modify_time_coord(self):
        """Test that an exception is raised if the coord_modification targets
        time coordinates."""

        msg = "Modifying time coordinates is not allowed."
        for coord in ["time", "forecast_period", "forecast_reference_time"]:
            coord_modification = {coord: 100}

            plugin = StandardiseMetadata(coord_modification=coord_modification)
            with self.assertRaisesRegex(ValueError, msg):
                plugin.process(self.cube)

    def test_discard_cellmethod(self):
        """Test changes to cell_methods"""
        cube = self.cube.copy()
        cube.cell_methods = [
            iris.coords.CellMethod(method="point", coords="time"),
            iris.coords.CellMethod(method="max", coords="realization"),
        ]
        result = StandardiseMetadata().process(cube)
        self.assertEqual(
            result.cell_methods,
            (iris.coords.CellMethod(method="max", coords="realization"),),
        )

    def test_float_deescalation(self):
        """Test precision de-escalation from float64 to float32"""
        cube = self.cube.copy()
        cube.data = cube.data.astype(np.float64)
        result = StandardiseMetadata().process(cube)
        self.assertEqual(result.data.dtype, np.float32)
        np.testing.assert_array_almost_equal(result.data, self.cube.data, decimal=4)

    def test_float_deescalation_with_unit_change(self):
        """Covers the bug where unit conversion from an integer input field causes
        float64 escalation"""
        cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.int16), name="rainrate", units="mm h-1"
        )
        result = StandardiseMetadata(new_units="m s-1").process(cube)
        self.assertEqual(cube.dtype, np.float32)
        self.assertEqual(result.data.dtype, np.float32)

    def test_air_temperature_status_flag_coord(self):
        """
        Ensure we handle cubes which now include an 'air_temperature_status flag'
        coord to signify points below surface altitude, where previously this
        was denoted by NaN values in the data.

        See https://github.com/metoppv/improver/pull/1839
        """
        cube = set_up_variable_cube(
            np.full((3, 3, 5, 5), fill_value=282, dtype=np.float32),
            spatial_grid="latlon",
            standard_grid_metadata="gl_det",
            pressure=True,
            vertical_levels=[100000.0, 97500.0, 95000.0],
            realizations=[0, 18, 19],
        )
        # The target cube has 'NaN' values in its data to denote points below
        # surface altitude.
        result_no_sf = cube.copy()
        result_no_sf.data[:, 0, ...] = np.nan
        target = StandardiseMetadata().process(result_no_sf)

        cube_with_flags = cube.copy()
        flag_status = np.zeros((3, 3, 5, 5), dtype=np.int8)
        flag_status[:, 0, ...] = 1
        status_flag_coord = AuxCoord(
            points=flag_status,
            standard_name="air_temperature status_flag",
            var_name="flag",
            attributes={
                "flag_meanings": "above_surface_pressure below_surface_pressure",
                "flag_values": np.array([0, 1], dtype="int8"),
            },
        )
        cube_with_flags.add_aux_coord(status_flag_coord, (0, 1, 2, 3))

        result = StandardiseMetadata().process(cube_with_flags)
        np.testing.assert_array_equal(result.data, target.data)
        self.assertEqual(result.coords(), target.coords())

    def test_air_temperature_status_flag_coord_without_realization(self):
        """
        Ensure we handle cubes which now include an 'air_temperature_status flag'
        coord to signify points below surface altitude, where previously this
        was denoted by NaN values in the data.

        Tests this standardisation works if input is a deterministic forecast without
        a realization coordinate.
        """
        cube = set_up_variable_cube(
            np.full((3, 5, 5), fill_value=282, dtype=np.float32),
            spatial_grid="latlon",
            standard_grid_metadata="gl_det",
            pressure=True,
            vertical_levels=[100000.0, 97500.0, 95000.0],
        )
        # The target cube has 'NaN' values in its data to denote points below
        # surface altitude.
        result_no_sf = cube.copy()
        result_no_sf.data[0, ...] = np.nan
        target = StandardiseMetadata().process(result_no_sf)

        cube_with_flags = cube.copy()
        flag_status = np.zeros((3, 5, 5), dtype=np.int8)
        flag_status[0, ...] = 1
        status_flag_coord = AuxCoord(
            points=flag_status,
            standard_name="air_temperature status_flag",
            var_name="flag",
            attributes={
                "flag_meanings": "above_surface_pressure below_surface_pressure",
                "flag_values": np.array([0, 1], dtype="int8"),
            },
        )
        cube_with_flags.add_aux_coord(status_flag_coord, (0, 1, 2))

        result = StandardiseMetadata().process(cube_with_flags)
        np.testing.assert_array_equal(result.data, target.data)
        self.assertEqual(result.coords(), target.coords())

    def test_long_name_removed(self):
        cube = self.cube.copy()
        cube.long_name = "kittens"
        result = StandardiseMetadata().process(cube)
        assert (
            result.long_name is None
        ), "long_name removal expected, but long_name is not None"


if __name__ == "__main__":
    unittest.main()

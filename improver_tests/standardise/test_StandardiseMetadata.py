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
"""Unit tests for the standardise.StandardiseMetadata plugin."""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.standardise import StandardiseMetadata
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test_process(IrisTest):
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
        self.plugin = StandardiseMetadata()

    def test_null(self):
        """Test process method with default arguments returns an unchanged
        cube"""
        result = self.plugin.process(self.cube.copy())
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, self.cube.data)
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
        result = self.plugin.process(self.cube)
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
        result = self.plugin.process(self.cube)
        self.assertEqual(result.coord("time").points.dtype, np.int64)

    def test_collapse_scalar_dimensions(self):
        """Test scalar dimension is collapsed"""
        cube = iris.util.new_axis(self.cube, "time")
        result = self.plugin.process(cube)
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
        result = self.plugin.process(cube)
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
        result = self.plugin.process(
            self.cube,
            new_name=new_name,
            new_units="degC",
            coords_to_remove=["forecast_period"],
            attributes_dict=attribute_changes,
        )
        self.assertEqual(result.name(), new_name)
        self.assertEqual(result.units, "degC")
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)
        self.assertDictEqual(result.attributes, expected_attributes)
        self.assertNotIn("forecast_period", [coord.name() for coord in result.coords()])

    def test_discard_cellmethod(self):
        """Test changes to cell_methods"""
        cube = self.cube.copy()
        cube.cell_methods = [
            iris.coords.CellMethod(method="point", coords="time"),
            iris.coords.CellMethod(method="max", coords="realization"),
        ]
        result = self.plugin.process(
            cube,
        )
        self.assertEqual(
            result.cell_methods,
            (iris.coords.CellMethod(method="max", coords="realization"),),
        )

    def test_float_deescalation(self):
        """Test precision de-escalation from float64 to float32"""
        cube = self.cube.copy()
        cube.data = cube.data.astype(np.float64)
        result = self.plugin.process(cube)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertArrayAlmostEqual(result.data, self.cube.data, decimal=4)

    def test_float_deescalation_with_unit_change(self):
        """Covers the bug where unit conversion from an integer input field causes
        float64 escalation"""
        cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.int16), name="rainrate", units="mm h-1"
        )
        result = self.plugin.process(cube, new_units="m s-1")
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
            height_levels=[100000.0, 97500.0, 95000.0],
            realizations=[0, 18, 19],
        )
        # The target cube has 'NaN' values in its data to denote points below
        # surface altitude.
        result_no_sf = cube.copy()
        result_no_sf.data[:, 0, ...] = np.nan
        target = self.plugin.process(result_no_sf)

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

        result = self.plugin.process(cube_with_flags)
        self.assertArrayEqual(result.data, target.data)
        self.assertEqual(result.coords(), target.coords())


if __name__ == "__main__":
    unittest.main()

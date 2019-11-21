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
"""Unit tests for the standardise.StandardiseGridAndMetadata plugin."""

import unittest
import numpy as np
from iris.tests import IrisTest

from improver.standardise import StandardiseGridAndMetadata
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test__init__(unittest.TestCase):
    """Test initialisation"""

    def test_default(self):
        """Test initialisation with default options"""
        plugin = StandardiseGridAndMetadata()
        self.assertEqual(plugin.regrid_mode, 'bilinear')
        self.assertEqual(plugin.extrapolation_mode, 'nanmask')
        self.assertIsNone(plugin.landmask_source_grid)
        self.assertIsNone(plugin.landmask_vicinity)
        self.assertEqual(plugin.landmask_name, 'land_binary_mask')
        self.assertSequenceEqual(
            plugin.grid_attributes,
            ['mosg__grid_version', 'mosg__grid_domain', 'mosg__grid_type',
             'mosg__model_configuration', 'institution'])

    def test_error_missing_landmask(self):
        """Test an error is thrown if no mask is provided for masked
        regridding"""
        msg = "requires an input landmask cube"
        with self.assertRaisesRegex(ValueError, msg):
            StandardiseGridAndMetadata(regrid_mode='nearest-with-mask')


class Test_process(IrisTest):
    """Test the process method. Complex regridded values are not tested here
    as this is covered by unit tests for the regridding routines
    (iris.cube.Cube.regrid and improver.standardise.RegridLandAndSea).
    """

    def setUp(self):
        """Set up input cubes and landmask"""
        self.cube = set_up_variable_cube(
            282*np.ones((15, 15), dtype=np.float32), spatial_grid='latlon',
            standard_grid_metadata='gl_det')

        # set up dummy landmask on source grid
        landmask = np.full((15, 15), False)
        self.landmask = self.cube.copy(data=landmask)
        self.landmask.rename("land_binary_mask")
        self.landmask.units = "no_unit"
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.landmask.remove_coord(coord)

        # set up dummy landmask on high resolution target grid
        self.target_grid = set_up_variable_cube(
            np.full((12, 12), False), name="land_binary_mask", units="no_unit",
            spatial_grid='equalarea', standard_grid_metadata='uk_det')
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.target_grid.remove_coord(coord)

    def test_null(self):
        """Test process method with default arguments returns an unchanged
        cube"""
        result = StandardiseGridAndMetadata().process(self.cube)
        self.assertArrayAlmostEqual(result.data, self.cube.data)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_basic_regrid(self):
        """Test default regridding arguments return expected dimensionality
        and updated grid-defining attributes"""
        expected_data = 282*np.ones((12, 12), dtype=np.float32)
        result = StandardiseGridAndMetadata().process(
            self.cube, self.target_grid)
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ['x', 'y']:
            self.assertEqual(
                result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(
            result.attributes, self.target_grid.attributes)

    def test_access_regrid_with_landmask(self):
        """Test the RegridLandAndSea module is correctly called when using
        landmask arguments. Diagnosed by identifiable error."""
        msg = "Distance of 10000m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            StandardiseGridAndMetadata(
                regrid_mode='nearest-with-mask', landmask=self.landmask,
                landmask_vicinity=10000).process(self.cube, self.target_grid)

    def test_run_regrid_with_landmask(self):
        """Test masked regridding (same expected values as basic, since input
        points are all equal)"""
        expected_data = 282*np.ones((12, 12), dtype=np.float32)
        result = StandardiseGridAndMetadata(
            regrid_mode='nearest-with-mask', landmask=self.landmask,
            landmask_vicinity=90000).process(self.cube, self.target_grid)
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ['x', 'y']:
            self.assertEqual(
                result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(
            result.attributes, self.target_grid.attributes)

    def test_error_regrid_with_incorrect_landmask(self):
        """Test an error is thrown if a landmask is provided that does not
        match the source grid"""
        landmask = self.target_grid.copy()
        plugin = StandardiseGridAndMetadata(
            regrid_mode='nearest-with-mask', landmask=landmask,
            landmask_vicinity=90000)
        msg = "Source landmask does not match input grid"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube, self.target_grid)

    def test_metadata_changes(self):
        """Test changes to cube name, coordinates and attributes without
        regridding"""
        new_name = "regridded_air_temperature"
        attribute_changes = {"institution": "Met Office",
                             "mosg__grid_version": "remove"}
        expected_attributes = {"mosg__grid_domain": "global",
                               "mosg__grid_type": "standard",
                               "mosg__model_configuration": "gl_det",
                               "institution": "Met Office"}
        expected_data = self.cube.data.copy() - 273.15
        result = StandardiseGridAndMetadata().process(
            self.cube, new_name=new_name, new_units="degC",
            coords_to_remove=["forecast_period"],
            attributes_dict=attribute_changes)
        self.assertEqual(result.name(), new_name)
        self.assertEqual(result.units, "degC")
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)
        self.assertDictEqual(result.attributes, expected_attributes)
        self.assertNotIn(
            "forecast_period", [coord.name() for coord in result.coords()])

    def test_attribute_changes_with_regridding(self):
        """Test attributes can be manually updated after regridding"""
        attribute_changes = {"institution": "Met Office",
                             "mosg__grid_version": "remove"}
        expected_attributes = {"mosg__grid_domain": "uk_extended",
                               "mosg__grid_type": "standard",
                               "mosg__model_configuration": "uk_det",
                               "institution": "Met Office"}
        result = StandardiseGridAndMetadata().process(
            self.cube, self.target_grid, attributes_dict=attribute_changes)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_fix_float64(self):
        """Test precision de-escalation"""
        self.cube.data = self.cube.data.astype(np.float64)
        result = StandardiseGridAndMetadata().process(
            self.cube, fix_float64=True)
        self.assertEqual(result.data.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()

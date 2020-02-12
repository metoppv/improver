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
from datetime import datetime

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.standardise import StandardiseGridAndMetadata
from improver.utilities.warnings_handler import ManageWarnings

from ..set_up_test_cubes import set_up_variable_cube

# The warning messages are internal to the iris.analysis module v2.2.0
IGNORED_MESSAGES = ["Using a non-tuple sequence for multidimensional indexing"]
WARNING_TYPES = [FutureWarning]


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

    def test_error_missing_landmask(self):
        """Test an error is thrown if no mask is provided for masked
        regridding"""
        msg = "requires an input landmask cube"
        with self.assertRaisesRegex(ValueError, msg):
            StandardiseGridAndMetadata(regrid_mode='nearest-with-mask')


class Test_process_no_regrid(IrisTest):
    """Test the process method without regridding options."""

    def setUp(self):
        """Set up input cube"""
        self.cube = set_up_variable_cube(
            282*np.ones((5, 5), dtype=np.float32), spatial_grid='latlon',
            standard_grid_metadata='gl_det', time=datetime(2019, 10, 11),
            time_bounds=[datetime(2019, 10, 10, 23), datetime(2019, 10, 11)],
            frt=datetime(2019, 10, 10, 18))
        self.plugin = StandardiseGridAndMetadata()

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
            self.cube.coord(coord).points = (
                self.cube.coord(coord).points.astype(np.float64))
            self.cube.coord(coord).bounds = (
                self.cube.coord(coord).bounds.astype(np.float64))
        self.cube.coord("forecast_period").convert_units("hours")
        result = self.plugin.process(self.cube)
        self.assertEqual(result.coord("forecast_period").units, "seconds")
        self.assertEqual(
            result.coord("forecast_period").points.dtype, np.int32)
        self.assertEqual(
            result.coord("forecast_period").bounds.dtype, np.int32)
        self.assertEqual(result.coord("time").points.dtype, np.int64)
        self.assertEqual(result.coord("time").bounds.dtype, np.int64)

    def test_standardise_time_coords_missing_fp(self):
        """Test a missing time-type coordinate does not cause an error when
        standardisation is required"""
        self.cube.coord("time").points = (
            self.cube.coord("time").points.astype(np.float64))
        self.cube.remove_coord("forecast_period")
        result = self.plugin.process(self.cube)
        self.assertEqual(result.coord("time").points.dtype, np.int64)

    def test_collapse_scalar_dimensions(self):
        """Test scalar dimension is collapsed"""
        cube = iris.util.new_axis(self.cube, "time")
        result = self.plugin.process(cube)
        dim_coord_names = [coord.name() for coord in
                           result.coords(dim_coords=True)]
        aux_coord_names = [coord.name() for coord in
                           result.coords(dim_coords=False)]
        self.assertSequenceEqual(result.shape, (5, 5))
        self.assertNotIn("time", dim_coord_names)
        self.assertIn("time", aux_coord_names)

    def test_realization_not_collapsed(self):
        """Test scalar realization coordinate is preserved"""
        realization = AuxCoord([1], "realization")
        self.cube.add_aux_coord(realization)
        cube = iris.util.new_axis(self.cube, "realization")
        result = self.plugin.process(cube)
        dim_coord_names = [coord.name() for coord in
                           result.coords(dim_coords=True)]
        self.assertSequenceEqual(result.shape, (1, 5, 5))
        self.assertIn("realization", dim_coord_names)

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
        result = self.plugin.process(
            self.cube, new_name=new_name, new_units="degC",
            coords_to_remove=["forecast_period"],
            attributes_dict=attribute_changes)
        self.assertEqual(result.name(), new_name)
        self.assertEqual(result.units, "degC")
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)
        self.assertDictEqual(result.attributes, expected_attributes)
        self.assertNotIn(
            "forecast_period", [coord.name() for coord in result.coords()])

    def test_float_deescalation(self):
        """Test precision de-escalation from float64 to float32"""
        cube = self.cube.copy()
        cube.data = cube.data.astype(np.float64)
        result = self.plugin.process(cube)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertArrayAlmostEqual(self.cube.data, result.data, decimal=4)


class Test_process_regrid_options(IrisTest):
    """Test the process method with regridding options. Regridded values
    are not tested here as this is covered by unit tests for the regridding
    routines (iris.cube.Cube.regrid and improver.standardise.RegridLandAndSea).
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

    def test_basic_regrid(self):
        """Test default regridding arguments return expected dimensionality
        and updated grid-defining attributes"""
        expected_data = 282*np.ones((12, 12), dtype=np.float32)
        expected_attributes = {"mosg__model_configuration": "gl_det",
                               "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"]}
        for attr in ["mosg__grid_domain", "mosg__grid_type",
                     "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = StandardiseGridAndMetadata().process(
            self.cube, self.target_grid.copy())
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ['x', 'y']:
            self.assertEqual(
                result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(result.attributes, expected_attributes)

    @ManageWarnings(ignored_messages=IGNORED_MESSAGES,
                    warning_types=WARNING_TYPES)
    def test_access_regrid_with_landmask(self):
        """Test the RegridLandAndSea module is correctly called when using
        landmask arguments. Diagnosed by identifiable error."""
        msg = "Distance of 10000m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            StandardiseGridAndMetadata(
                regrid_mode='nearest-with-mask', landmask=self.landmask,
                landmask_vicinity=10000).process(
                    self.cube, target_grid=self.target_grid)

    @ManageWarnings(ignored_messages=IGNORED_MESSAGES,
                    warning_types=WARNING_TYPES)
    def test_run_regrid_with_landmask(self):
        """Test masked regridding (same expected values as basic, since input
        points are all equal)"""
        expected_data = 282*np.ones((12, 12), dtype=np.float32)
        expected_attributes = {"mosg__model_configuration": "gl_det",
                               "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"]}
        for attr in ["mosg__grid_domain", "mosg__grid_type",
                     "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = StandardiseGridAndMetadata(
            regrid_mode='nearest-with-mask', landmask=self.landmask,
            landmask_vicinity=90000).process(
                self.cube, target_grid=self.target_grid.copy())
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ['x', 'y']:
            self.assertEqual(
                result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_error_regrid_with_incorrect_landmask(self):
        """Test an error is thrown if a landmask is provided that does not
        match the source grid"""
        landmask = self.target_grid.copy()
        plugin = StandardiseGridAndMetadata(
            regrid_mode='nearest-with-mask', landmask=landmask,
            landmask_vicinity=90000)
        msg = "Source landmask does not match input grid"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube, target_grid=self.target_grid)

    @ManageWarnings(record=True)
    def test_warning_source_not_landmask(self, warning_list=None):
        """Test warning is raised if landmask_source_grid is not a landmask"""
        expected_data = 282*np.ones((12, 12), dtype=np.float32)
        self.landmask.rename("not_a_landmask")
        result = StandardiseGridAndMetadata(
            regrid_mode='nearest-with-mask', landmask=self.landmask,
            landmask_vicinity=90000).process(
                self.cube, target_grid=self.target_grid)
        msg = "Expected land_binary_mask in input_landmask cube"
        self.assertTrue(any([msg in str(warning) for warning in warning_list]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertArrayAlmostEqual(result.data, expected_data)

    @ManageWarnings(record=True)
    def test_warning_target_not_landmask(self, warning_list=None):
        """Test warning is raised if target_grid is not a landmask"""
        expected_data = 282*np.ones((12, 12), dtype=np.float32)
        self.target_grid.rename("not_a_landmask")
        self.landmask.rename("not_a_landmask")
        result = StandardiseGridAndMetadata(
            regrid_mode='nearest-with-mask', landmask=self.landmask,
            landmask_vicinity=90000).process(
                self.cube, target_grid=self.target_grid)
        msg = "Expected land_binary_mask in target_grid cube"
        self.assertTrue(any([msg in str(warning) for warning in warning_list]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_attribute_changes_with_regridding(self):
        """Test attributes inherited on regridding"""
        expected_attributes = self.cube.attributes
        expected_attributes["title"] = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        for attr in ["mosg__grid_domain", "mosg__grid_type",
                     "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = StandardiseGridAndMetadata().process(
            self.cube, target_grid=self.target_grid)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_new_title(self):
        """Test new title can be set on regridding"""
        new_title = "Global Model Forecast on UK 2km Standard Grid"
        expected_attributes = self.cube.attributes
        expected_attributes["title"] = new_title
        for attr in ["mosg__grid_domain", "mosg__grid_type",
                     "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = StandardiseGridAndMetadata().process(
            self.cube, target_grid=self.target_grid, regridded_title=new_title)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_attribute_changes_after_regridding(self):
        """Test attributes can be manually updated after regridding"""
        attribute_changes = {"institution": "Met Office",
                             "mosg__grid_version": "remove"}
        expected_attributes = {"mosg__grid_domain": "uk_extended",
                               "mosg__grid_type": "standard",
                               "mosg__model_configuration": "gl_det",
                               "institution": "Met Office",
                               "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"]}
        result = StandardiseGridAndMetadata().process(
            self.cube, target_grid=self.target_grid,
            attributes_dict=attribute_changes)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_incorrect_grid_attributes_removed(self):
        """Test grid attributes not present on the target cube are removed
        after regridding"""
        self.target_grid.attributes.pop("mosg__grid_domain")
        result = StandardiseGridAndMetadata().process(
            self.cube, target_grid=self.target_grid)
        self.assertNotIn("mosg__grid_domain", result.attributes)


if __name__ == '__main__':
    unittest.main()

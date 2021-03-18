# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for the standardise.RegridLandSea plugin."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.standardise import RegridLandSea
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings

# The warning messages are internal to the iris.analysis module v2.2.0
IGNORED_MESSAGES = ["Using a non-tuple sequence for multidimensional indexing"]
WARNING_TYPES = [FutureWarning]


class Test__init__(IrisTest):
    """Test initialisation"""

    def test_error_unrecognised_regrid_mode(self):
        """Test error is thrown if regrid mode is not in expected values list"""
        msg = "Unrecognised regrid mode"
        with self.assertRaisesRegex(ValueError, msg):
            RegridLandSea(regrid_mode="kludge")

    def test_error_missing_landmask(self):
        """Test an error is thrown if no landmask is provided where required"""
        msg = "requires an input landmask cube"
        with self.assertRaisesRegex(ValueError, msg):
            RegridLandSea(regrid_mode="nearest-with-mask")


class Test_process(IrisTest):
    """Test the process method for the RegridLandSea plugin. Regridded values
    are not tested here as this is covered by unit tests for the regridding
    routines (iris.cube.Cube.regrid and improver.standardise.AdjustLandSeaPoints).
    """

    def setUp(self):
        """Set up input cubes and landmask"""
        # Set domain_corner to define a lat/lon grid which overlaps and completely covers the UK standard grid,
        # using 54.9,-2.5 as centre (as equalarea grid is configured)
        domain_corner = (54.9 - 15, -2.5 - 15)
        self.cube = set_up_variable_cube(
            282 * np.ones((15, 15), dtype=np.float32),
            spatial_grid="latlon",
            standard_grid_metadata="gl_det",
            domain_corner=domain_corner,
            grid_spacing=2,
        )

        # set up dummy landmask on source grid
        landmask = np.full((15, 15), False)
        self.landmask = self.cube.copy(data=landmask)
        self.landmask.rename("land_binary_mask")
        self.landmask.units = "no_unit"
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.landmask.remove_coord(coord)

        # set up dummy landmask on high resolution target grid
        self.target_grid = set_up_variable_cube(
            np.full((12, 12), False),
            name="land_binary_mask",
            units="no_unit",
            spatial_grid="equalarea",
            standard_grid_metadata="uk_det",
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.target_grid.remove_coord(coord)

    def test_basic_regrid(self):
        """Test default regridding arguments return expected dimensionality
        and updated grid-defining attributes"""
        expected_data = 282 * np.ones((12, 12), dtype=np.float32)
        expected_attributes = {
            "mosg__model_configuration": "gl_det",
            "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"],
        }
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = RegridLandSea()(self.cube, self.target_grid.copy())
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ["x", "y"]:
            self.assertEqual(result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(result.attributes, expected_attributes)

    @ManageWarnings(ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_access_regrid_with_landmask(self):
        """Test the AdjustLandSeaPoints module is correctly called when using
        landmask arguments. Diagnosed by identifiable error."""
        msg = "Distance of 1000m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            RegridLandSea(
                regrid_mode="nearest-with-mask",
                landmask=self.landmask,
                landmask_vicinity=1000,
            )(self.cube, self.target_grid)

    @ManageWarnings(ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_run_regrid_with_landmask(self):
        """Test masked regridding (same expected values as basic, since input
        points are all equal)"""
        expected_data = 282 * np.ones((12, 12), dtype=np.float32)
        expected_attributes = {
            "mosg__model_configuration": "gl_det",
            "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"],
        }
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = RegridLandSea(
            regrid_mode="nearest-with-mask",
            landmask=self.landmask,
            landmask_vicinity=90000,
        )(self.cube, self.target_grid.copy())
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ["x", "y"]:
            self.assertEqual(result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_error_regrid_with_incorrect_landmask(self):
        """Test an error is thrown if a landmask is provided that does not
        match the source grid"""
        landmask = self.target_grid.copy()
        plugin = RegridLandSea(
            regrid_mode="nearest-with-mask", landmask=landmask, landmask_vicinity=90000
        )
        msg = "Source landmask does not match input grid"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube, self.target_grid)

    @ManageWarnings(record=True)
    def test_warning_source_not_landmask(self, warning_list=None):
        """Test warning is raised if landmask_source_grid is not a landmask"""
        expected_data = 282 * np.ones((12, 12), dtype=np.float32)
        self.landmask.rename("not_a_landmask")
        result = RegridLandSea(
            regrid_mode="nearest-with-mask",
            landmask=self.landmask,
            landmask_vicinity=90000,
        )(self.cube, self.target_grid)
        msg = "Expected land_binary_mask in input_landmask cube"
        self.assertTrue(any(msg in str(warning) for warning in warning_list))
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertArrayAlmostEqual(result.data, expected_data)

    @ManageWarnings(record=True)
    def test_warning_target_not_landmask(self, warning_list=None):
        """Test warning is raised if target_grid is not a landmask"""
        expected_data = 282 * np.ones((12, 12), dtype=np.float32)
        self.target_grid.rename("not_a_landmask")
        self.landmask.rename("not_a_landmask")
        result = RegridLandSea(
            regrid_mode="nearest-with-mask",
            landmask=self.landmask,
            landmask_vicinity=90000,
        ).process(self.cube, self.target_grid)
        msg = "Expected land_binary_mask in target_grid cube"
        self.assertTrue(any(msg in str(warning) for warning in warning_list))
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_attribute_changes_with_regridding(self):
        """Test attributes inherited on regridding"""
        expected_attributes = self.cube.attributes
        expected_attributes["title"] = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = RegridLandSea()(self.cube, self.target_grid)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_new_title(self):
        """Test new title can be set on regridding"""
        new_title = "Global Model Forecast on UK 2km Standard Grid"
        expected_attributes = self.cube.attributes
        expected_attributes["title"] = new_title
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = RegridLandSea()(self.cube, self.target_grid, regridded_title=new_title)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_incorrect_grid_attributes_removed(self):
        """Test grid attributes not present on the target cube are removed
        after regridding"""
        self.target_grid.attributes.pop("mosg__grid_domain")
        result = RegridLandSea()(self.cube, self.target_grid)
        self.assertNotIn("mosg__grid_domain", result.attributes)


if __name__ == "__main__":
    unittest.main()

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the standardise.RegridLandSea plugin."""

import unittest

import numpy as np
import pytest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.regrid.landsea import RegridLandSea
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver_tests import ImproverTest


class Test__init__(unittest.TestCase):
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


class Test_process(ImproverTest):
    """Test the process method for the RegridLandSea plugin. Regridded values
    are not tested here as this is covered by unit tests for the regridding
    routines (iris.cube.Cube.regrid and improver.standardise.AdjustLandSeaPoints).
    """

    def setUp(self):
        """Set up input cubes and landmask"""
        # Set domain_corner to define a lat/lon grid which overlaps and completely
        # covers the UK standard grid,
        # using 54.9,-2.5 as centre (as equalarea grid is configured)
        domain_corner = (54.9 - 15, -2.5 - 15)
        self.cube = set_up_variable_cube(
            282 * np.ones((15, 15), dtype=np.float32),
            spatial_grid="latlon",
            standard_grid_metadata="gl_det",
            domain_corner=domain_corner,
            x_grid_spacing=2,
            y_grid_spacing=2,
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
        np.testing.assert_array_almost_equal(result.data, expected_data)
        for axis in ["x", "y"]:
            self.assertEqual(result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_access_regrid_with_landmask(self):
        """Test the AdjustLandSeaPoints module is correctly called when using
        landmask arguments. Diagnosed by identifiable error."""
        msg = "Distance of 1000.0m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            RegridLandSea(
                regrid_mode="nearest-with-mask",
                landmask=self.landmask,
                landmask_vicinity=1000,
            )(self.cube, self.target_grid)

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
        np.testing.assert_array_almost_equal(result.data, expected_data)
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

    def test_warning_source_not_landmask(self):
        """Test warning is raised if landmask_source_grid is not a landmask"""
        expected_data = 282 * np.ones((12, 12), dtype=np.float32)
        self.landmask.rename("not_a_landmask")
        msg = "Expected land_binary_mask in input_landmask cube"
        with pytest.warns(UserWarning, match=msg):
            result = RegridLandSea(
                regrid_mode="nearest-with-mask",
                landmask=self.landmask,
                landmask_vicinity=90000,
            )(self.cube, self.target_grid)

        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_warning_target_not_landmask(self):
        """Test warning is raised if target_grid is not a landmask"""
        expected_data = 282 * np.ones((12, 12), dtype=np.float32)
        self.target_grid.rename("not_a_landmask")
        self.landmask.rename("not_a_landmask")
        msg = "Expected land_binary_mask in target_grid cube"
        with pytest.warns(UserWarning, match=msg):
            result = RegridLandSea(
                regrid_mode="nearest-with-mask",
                landmask=self.landmask,
                landmask_vicinity=90000,
            ).process(self.cube, self.target_grid)

        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_attribute_changes_with_regridding(self):
        """Test attributes inherited on regridding"""
        expected_attributes = self.cube.attributes
        expected_attributes["title"] = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = RegridLandSea()(self.cube, self.target_grid)
        self.assertDictEqual(expected_attributes, result.attributes)

    def test_new_title(self):
        """Test new title can be set on regridding"""
        new_title = "Global Model Forecast on UK 2km Standard Grid"
        expected_attributes = self.cube.attributes
        expected_attributes["title"] = f"Post-Processed {new_title}"
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]
        result = RegridLandSea()(self.cube, self.target_grid, regridded_title=new_title)
        self.assertDictEqual(expected_attributes, result.attributes)

    def test_incorrect_grid_attributes_removed(self):
        """Test grid attributes not present on the target cube are removed
        after regridding"""
        self.target_grid.attributes.pop("mosg__grid_domain")
        result = RegridLandSea()(self.cube, self.target_grid)
        self.assertNotIn("mosg__grid_domain", result.attributes)

    def test_area_weighted_regrid(self):
        """Test esmf-area-weighted regridding returns expected dimensionality
        and updated grid-defining attributes, with characteristic area-weighted
        averaging behavior"""
        pytest.importorskip("esmf_regrid")

        # Create a checkerboard pattern to demonstrate area-weighted averaging
        # This will produce distinct results compared to nearest-neighbor
        checkerboard = np.zeros((15, 15), dtype=np.float32)
        checkerboard[::2, ::2] = 300.0  # High values
        checkerboard[1::2, 1::2] = 300.0
        checkerboard[::2, 1::2] = 260.0  # Low values
        checkerboard[1::2, ::2] = 260.0
        self.cube.data = checkerboard

        expected_attributes = {
            "mosg__model_configuration": "gl_det",
            "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"],
        }
        for attr in ["mosg__grid_domain", "mosg__grid_type", "mosg__grid_version"]:
            expected_attributes[attr] = self.target_grid.attributes[attr]

        result = RegridLandSea(regrid_mode="esmf-area-weighted")(
            self.cube, self.target_grid.copy()
        )

        # With area-weighted regridding, the checkerboard should average to ~280
        # (not exactly due to grid cell overlap), demonstrating conservative regridding
        self.assertEqual(result.shape, (12, 12))
        self.assertAlmostEqual(result.data.mean(), 280.0, delta=5.0)
        # Check that we get averaging, not just nearest values
        self.assertTrue(np.any((result.data > 260.0) & (result.data < 300.0)))

        for axis in ["x", "y"]:
            self.assertEqual(result.coord(axis=axis), self.target_grid.coord(axis=axis))
        self.assertDictEqual(result.attributes, expected_attributes)


if __name__ == "__main__":
    unittest.main()

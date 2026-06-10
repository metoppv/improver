# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the generate_ancillary.GenerateOrogBandAncils plugin."""

import unittest

import numpy as np
from cf_units import Unit
from iris.cube import Cube

from improver.generate_ancillaries.generate_ancillary import (
    GenerateOrographyBandAncils as GenOrogMasks,
)


def set_up_landmask_cube(landmask_data=None):
    """Set up a basic landmask cube."""
    if landmask_data is None:
        landmask_data = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
    return Cube(landmask_data, long_name="test land", units="1")


def set_up_orography_cube(orog_data=None):
    """Set up a basic orography cube."""
    if orog_data is None:
        orog_data = np.array(
            [[10.0, 0.0, 0.0], [20.0, 100.0, 15.0], [-10.0, 100.0, 40.0]]
        )
    return Cube(orog_data, long_name="test orog", units="m")


class Test_sea_mask(unittest.TestCase):
    """Test the masking out of sea points with the sea_mask method."""

    def setUp(self):
        """Set up for tests."""
        self.landmask = set_up_landmask_cube()
        self.orography = set_up_orography_cube()

    def test_basic(self):
        """Test that the expected data is returned when the landmask specifies
        a mix of land and sea points."""
        expected_data = np.array(
            [[10.0, 1e20, 1e20], [20.0, 1e20, 1e20], [-10, 100.0, 40]]
        )
        expected_mask = np.array(
            [[False, True, True], [False, True, True], [False, False, False]]
        )
        result = GenOrogMasks().sea_mask(self.landmask.data, self.orography.data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result.data, expected_data)
        np.testing.assert_array_almost_equal(result.mask, expected_mask)

    def test_basic_sea_fill_value(self):
        """Test that the expected data is returned when the landmask specifies
        a mix of land and sea points and a fill value is given."""
        expected_data = np.array([[10.0, 0, 0], [20.0, 0, 0], [-10, 100.0, 40]])
        result = GenOrogMasks().sea_mask(
            self.landmask.data, self.orography.data, sea_fill_value=0
        )
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected_data)
        self.assertEqual(np.ma.is_masked(result), False)

    def test_all_land_points(self):
        """Test that the expected data is returned when the landmask specifies
        only land points."""
        expected = np.array(
            [[10.0, 0.0, 0.0], [20.0, 100.0, 15.0], [-10.0, 100.0, 40.0]]
        )
        landmask_data = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        landmask = set_up_landmask_cube(landmask_data=landmask_data)
        result = GenOrogMasks().sea_mask(landmask.data, self.orography.data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected)


class Test_gen_orography_masks(unittest.TestCase):
    """
    Test the gen_orography_masks method orography band mask
    ancillary generation plugin.
    """

    def setUp(self):
        """setting up test input and output data sets"""
        self.landmask = set_up_landmask_cube()
        self.orography = set_up_orography_cube()
        self.valley_threshold = [-10, 10]
        self.exp_valleymask = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
        self.land_threshold = [0, 50]
        self.exp_landmask = np.array([[[1.0, 0, 0], [1.0, 0, 0], [0.0, 0.0, 1.0]]])

        self.nonzero_land_threshold = [30, 100]
        self.exp_nonzero_landmask = np.array([[[0, 0, 0], [0, 0, 0], [0, 1, 1]]])

        self.high_land_threshold = [500, 600]
        self.exp_high_landmask = np.array([[[0.0, 0, 0], [0.0, 0, 0], [0.0, 0.0, 0.0]]])

    def test_valleyband_data(self):
        """test correct mask is produced for land bands < 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.valley_threshold
        )
        np.testing.assert_array_almost_equal(result.data, self.exp_valleymask)

    def test_valleyband_cube(self):
        """test correct cube data is produced for land bands < 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.valley_threshold
        )
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "False"
        )
        self.assertEqual(
            result.coord("topographic_zone").points, np.mean(self.valley_threshold)
        )
        self.assertEqual(
            result.coord("topographic_zone").bounds[0][0], self.valley_threshold[0]
        )
        self.assertEqual(
            result.coord("topographic_zone").bounds[0][1], self.valley_threshold[1]
        )

    def test_landband_data(self):
        """test correct mask is produced for land bands > 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_threshold
        )
        np.testing.assert_array_almost_equal(result.data, self.exp_landmask)

    def test_landband_cube(self):
        """test correct cube data is produced for land bands > 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_threshold
        )
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "False"
        )
        self.assertEqual(
            result.coord("topographic_zone").points, np.mean(self.land_threshold)
        )
        self.assertEqual(
            result.coord("topographic_zone").bounds[0][0], self.land_threshold[0]
        )
        self.assertEqual(
            result.coord("topographic_zone").bounds[0][1], self.land_threshold[1]
        )

    def test_nonzero_landband_data(self):
        """test that correct data is produced when neither landband
        bound is zero."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.nonzero_land_threshold
        )
        np.testing.assert_array_almost_equal(result.data, self.exp_nonzero_landmask)

    def test_nonzero_landband_cube(self):
        """test that a correct cube is produced when neither landband
        bound is zero."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.nonzero_land_threshold
        )
        self.assertEqual(
            result.coord("topographic_zone").points,
            np.mean(self.nonzero_land_threshold),
        )

    def test_high_landband_cube(self):
        """test that a correct cube is produced when the land band is
        higher than any land in the test cube."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.high_land_threshold
        )
        self.assertEqual(
            result.coord("topographic_zone").points, np.mean(self.high_land_threshold)
        )

    def test_high_landband_data(self):
        """test that a correct mask is produced when the land band is
        higher than any land in the test cube."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.high_land_threshold
        )
        np.testing.assert_array_almost_equal(result.data, self.exp_high_landmask)

    def test_all_land_points(self):
        """Test that a correct mask is produced when the landsea mask only has
        land points in it."""
        land_mask_cube = self.landmask.copy()
        land_mask_cube.data = np.ones((3, 3))
        result = GenOrogMasks().gen_orography_masks(
            self.orography, land_mask_cube, self.valley_threshold
        )
        expected_data = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_any_surface_type_mask(self):
        """Test that the correct mask is produced when no landsea mask is
        provided. This is equivalent to the all_land_points test above."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, None, self.valley_threshold
        )
        expected_data = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        np.testing.assert_array_almost_equal(result.data, expected_data)
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "True"
        )

    def test_unit_conversion_for_landband_data(self):
        """test correct mask is produced for land bands > 0m"""
        land_threshold = [0, 0.05]
        threshold_units = "km"
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, land_threshold, units=threshold_units
        )
        np.testing.assert_array_almost_equal(result.data, self.exp_landmask)
        self.assertEqual(result.coord("topographic_zone").units, Unit("m"))


class Test_process(unittest.TestCase):
    """
    Test the process method orography zone mask ancillary generation plugin.
    """

    def setUp(self):
        """setting up test input and output data sets"""
        self.landmask = set_up_landmask_cube()
        self.orography = set_up_orography_cube()
        self.threshold_dict = {"bounds": [[-10, 0], [0, 50]], "units": "m"}

    def test_thresholdset(self):
        """test the plugin produces correct number of cubes"""
        result = GenOrogMasks().process(
            self.orography, self.threshold_dict, landmask=self.landmask
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for nbhood.ApplyNeighbourhoodProcessingWithAMask."""

import unittest

import iris
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube,
)


class Test_collapse_mask_coord(unittest.TestCase):
    """
    Test the collapse_mask_coord method.
    Although the normalisation of any weights isn't done explicitly in this
    function we want to test that the normalisation functionality in the
    underlying functions is suitable for the cases we need to handle in
    this plugin.
    """

    def setUp(self):
        """Set up a simple cube to collapse"""
        # A simplified example of neighbourhood processed data for a set of
        # topographic bands for a single threshold.
        data = np.array(
            [
                [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 1, 1], [1, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ],
            dtype=np.float32,
        )
        cube = set_up_probability_cube(
            np.ones((1, 3, 3), dtype=np.float32), [278.15], spatial_grid="equalarea"
        )
        self.cube = add_coordinate(cube, [50, 100, 150], "topographic_zone", "m")
        self.cube = iris.util.squeeze(self.cube)
        self.cube.data = data
        # Set up weights cubes
        weights_data = np.array(
            [
                [[1, 1, 0], [1, 0.5, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0.5, 0.75], [1, 0.75, 0.5]],
                [[0, 0, 0], [0, 0, 0.25], [0, 0.25, 0.5]],
            ],
            dtype=np.float32,
        )
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            name="topographic_zone_weights",
            units="1",
            spatial_grid="equalarea",
        )
        self.weights_cube = add_coordinate(
            cube, [50, 100, 150], "topographic_zone", "m"
        )
        self.weights_cube.data = weights_data

    def test_basic(self):
        """Test we get the expected result with a simple collapse"""
        expected_data = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0.5]], dtype=np.float32
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data, expected_data)
        self.assertNotIn("topographic_zone", result.coords())

    def test_renormalise_when_missing_data(self):
        """
        Test we get the expected result when the weights need renormalising
        to account for missing data that comes from neighbourhood processing
        where there are no points to process in a given band for a given point.
        The expected behaviour is that the weights are renormalised and the
        result for that point takes 100% of the band with a valid value in.
        """
        data = np.array(
            [
                [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 1, 1], [1, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, np.nan]],
            ],
            dtype=np.float32,
        )
        self.cube.data = data
        expected_data = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0]], dtype=np.float32
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data, expected_data)
        self.assertNotIn("topographic_zone", result.coords())

    def test_masked_weights_data(self):
        """Test points where weights are masked.
        Covers the case where sea points may be masked out so they aren't
        neighbourhood processed."""
        self.weights_cube.data[:, 0, 0] = np.nan
        self.weights_cube.data = np.ma.masked_invalid(self.weights_cube.data)
        expected_data = np.array(
            [[np.nan, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0.5]], dtype=np.float32
        )
        expected_mask = np.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data.data, expected_data)
        assert_allclose(result.data.mask, expected_mask)
        self.assertNotIn("topographic_zone", result.coords())

    def test_masked_weights_and_missing(self):
        """Test masked weights, and one point has nan in from
        neighbourhood processing.
        Covers the case where sea points may be masked out so they aren't
        neighbourhood processed. The nan point comes from neighbourhood
        processing where it has found no points to process for a given point
        in a given band."""
        self.weights_cube.data[:, 0, 0] = np.nan
        self.weights_cube.data = np.ma.masked_invalid(self.weights_cube.data)
        self.cube.data[2, 2, 2] = np.nan
        expected_data = np.array(
            [[np.nan, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0.0]], dtype=np.float32
        )
        expected_mask = np.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data.data, expected_data)
        assert_allclose(result.data.mask, expected_mask)
        self.assertNotIn("topographic_zone", result.coords())


class Test_process(unittest.TestCase):
    """Test the process method of ApplyNeighbourhoodProcessingWithAMask."""

    def setUp(self):
        """
        Set up a cube with a single threshold and a cube with two thresholds
        to be processed.
        Set up mask and weights cubes.
        Set up expected results.
        """
        # Set up cube to process.
        data = np.array([[[1, 1, 1], [1, 1, 0], [0, 0, 0]]], dtype=np.float32)
        self.cube = set_up_probability_cube(data, [278.15], spatial_grid="equalarea")
        # Set up mask cube. Currently mask cubes have sea points set to zero,
        # not masked out.
        mask_data = np.array(
            [
                [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ],
            dtype=np.float32,
        )
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            name="topographic_zone_weights",
            units="1",
            spatial_grid="equalarea",
        )
        self.mask_cube = add_coordinate(cube, [50, 100, 150], "topographic_zone", "m")
        self.mask_cube.data = mask_data

        # Set up weights cubes, with one masked sea point. Currently only
        # weights cubes can have masked out sea points.
        weights_data = np.array(
            [
                [[np.nan, 1, 0], [1, 0.5, 0], [0, 0, 0]],
                [[np.nan, 0, 1], [0, 0.5, 0.75], [0.75, 0.75, 0.5]],
                [[np.nan, 0, 0], [0, 0, 0.25], [0.25, 0.25, 0.5]],
            ],
            dtype=np.float32,
        )
        weights_data = np.ma.masked_invalid(weights_data)
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            name="topographic_zone_weights",
            units="1",
            spatial_grid="equalarea",
        )
        self.weights_cube = add_coordinate(
            cube, [50, 100, 150], "topographic_zone", "m"
        )
        self.weights_cube.data = weights_data

        # Set all cubes to have the same 2km spaced x and y coords.
        for cube in [self.cube, self.mask_cube, self.weights_cube]:
            cube.coord("projection_x_coordinate").points = np.array(
                [0, 2000, 4000], dtype=np.float32
            )
            cube.coord("projection_y_coordinate").points = np.array(
                [0, 2000, 4000], dtype=np.float32
            )
        # Set up a cube with multiple thresholds
        cube2 = self.cube.copy()
        cube2.coord("air_temperature").points = np.array([273.15], dtype=np.float32)
        self.multi_threshold_cube = iris.cube.CubeList([cube2, self.cube]).merge_cube()
        # Set up expected uncollapsed result
        self.expected_uncollapsed_result = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[np.nan, 0.5, 0.5], [0.0, 0.25, 0.33333334], [0.0, 0.0, 0.0]],
                [[np.nan, np.nan, np.nan], [np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        # Set up expected collapsed result
        expected_result = np.array(
            [[np.nan, 1.0, 0.5], [1.0, 0.625, 0.25], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        expected_mask = np.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        self.expected_collapsed_result = np.ma.MaskedArray(
            expected_result, expected_mask
        )

    def test_basic_no_collapse_square(self):
        """Test process for a cube with 1 threshold and no collapse.
        This test shows the result of neighbourhood processing the same input
        data three times with the three different masks for the different
        topographic zones."""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000
        )
        result = plugin(self.cube, self.mask_cube)
        assert_allclose(result.data, self.expected_uncollapsed_result, equal_nan=True)
        expected_coords = self.cube.coords()
        expected_coords.insert(0, self.mask_cube.coord("topographic_zone"))
        self.assertEqual(result.coords(), expected_coords)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_empty_no_collapse_square(self):
        """In this case, the data are all zero to trigger the "all zero" optimisation."""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000
        )
        self.cube.data = np.zeros_like(self.cube.data)
        expected = np.zeros_like(self.expected_uncollapsed_result)
        expected[np.where(np.isnan(self.expected_uncollapsed_result))] = np.nan
        result = plugin(self.cube, self.mask_cube)
        assert_allclose(result.data, expected, equal_nan=True)
        expected_coords = self.cube.coords()
        expected_coords.insert(0, self.mask_cube.coord("topographic_zone"))
        self.assertEqual(result.coords(), expected_coords)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_sparse_no_collapse_square(self):
        """Repeat test_basic_no_collapse_square but with a large buffer of zeroes
        so that we trigger the data-shrinking efficiency code.
        """
        data = np.zeros((1, 9, 9), dtype=np.float32)
        data[..., 3:6, 3:6] = np.array(
            [[[1, 1, 1], [1, 1, 0], [0, 0, 0]]], dtype=np.float32
        )
        cube = set_up_probability_cube(data, [278.15], spatial_grid="equalarea")
        # Set up mask cube. Currently mask cubes have sea points set to zero,
        # not masked out.
        mask_data = np.zeros((3, 9, 9), dtype=np.float32)
        mask_data[..., 3:6, 3:6] = np.array(
            [
                [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ],
            dtype=np.float32,
        )
        mask_cube = set_up_variable_cube(
            np.ones((9, 9), dtype=np.float32),
            name="topographic_zone_weights",
            units="1",
            spatial_grid="equalarea",
        )
        mask_cube = add_coordinate(mask_cube, [50, 100, 150], "topographic_zone", "m")
        mask_cube.data = mask_data

        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000
        )
        result = plugin(cube, mask_cube)
        assert_allclose(
            result.data[..., 3:6, 3:6], self.expected_uncollapsed_result, equal_nan=True
        )
        expected_coords = cube.coords()
        expected_coords.insert(0, mask_cube.coord("topographic_zone"))
        self.assertEqual(result.coords(), expected_coords)
        self.assertEqual(result.metadata, cube.metadata)

    def test_basic_no_collapse_circular(self):
        """Test process for a cube with 1 threshold and no collapse.
        This test shows the result of neighbourhood processing the same input
        data three times with the three different masks for the different
        topographic zones."""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "circular", 2000
        )
        expected_data = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, np.nan]],
                [[np.nan, 1.0, 0.75], [0.0, 0.0, 0.33333334], [0.0, 0.0, 0.0]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, 0.0], [np.nan, 0.0, 0.0]],
            ]
        )
        result = plugin(self.cube, self.mask_cube)
        assert_allclose(result.data, expected_data, equal_nan=True)
        expected_coords = self.cube.coords()
        expected_coords.insert(0, self.mask_cube.coord("topographic_zone"))
        self.assertEqual(result.coords(), expected_coords)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_basic_collapse_square(self):
        """Test process for a cube with 1 threshold and collapsing the topographic_zones.
        This test shows the result of neighbourhood processing the same input
        data three times with the three different masks for the different
        topographic zones, then doing a weighted collapse of the topopgraphic
        band taking into account any missing data."""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000, collapse_weights=self.weights_cube
        )
        result = plugin(self.cube, self.mask_cube)

        assert_allclose(
            result.data.data, self.expected_collapsed_result.data, equal_nan=True
        )
        assert_array_equal(result.data.mask, self.expected_collapsed_result.mask)
        self.assertEqual(result.coords(), self.cube.coords())
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_basic_collapse_circular(self):
        """Test process for a cube with 1 threshold and collapsing the topographic_zones.
        This test shows the result of neighbourhood processing the same input
        data three times with the three different masks for the different
        topographic zones, then doing a weighted collapse of the topopgraphic
        band taking into account any missing data."""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "circular", 2000, collapse_weights=self.weights_cube
        )
        expected_data = np.array(
            [[np.nan, 1.0, 0.75], [1.0, 0.5, 0.25], [0.0, 0.0, 0.0]]
        )
        result = plugin(self.cube, self.mask_cube)
        assert_allclose(result.data.data, expected_data, equal_nan=True)
        assert_array_equal(result.data.mask, self.expected_collapsed_result.mask)
        self.assertEqual(result.coords(), self.cube.coords())
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_no_collapse_multithreshold(self):
        """Test process for a cube with 2 thresholds and no collapse.
        Same data as test_basic_no_collapse with an extra point in the leading
        threshold dimension"""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000
        )
        result = plugin(self.multi_threshold_cube, self.mask_cube)
        expected_result = np.array(
            [self.expected_uncollapsed_result, self.expected_uncollapsed_result]
        )
        assert_allclose(result.data, expected_result, equal_nan=True)
        expected_coords = self.multi_threshold_cube.coords()
        expected_coords.insert(1, self.mask_cube.coord("topographic_zone"))
        self.assertEqual(result.coords(), expected_coords)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_collapse_multithreshold(self):
        """Test process for a cube with 2 thresholds and collapsing the topographic_zones.
        Same data as test_basic_collapse with an extra point in the leading
        threshold dimension"""
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", "square", 2000, collapse_weights=self.weights_cube
        )
        result = plugin(self.multi_threshold_cube, self.mask_cube)
        expected_result = np.ma.MaskedArray(
            [self.expected_collapsed_result, self.expected_collapsed_result]
        )
        assert_allclose(result.data.data, expected_result.data, equal_nan=True)
        assert_array_equal(result.data.mask, expected_result.mask)
        self.assertEqual(result.coords(), self.multi_threshold_cube.coords())
        self.assertEqual(result.metadata, self.multi_threshold_cube.metadata)


if __name__ == "__main__":
    unittest.main()

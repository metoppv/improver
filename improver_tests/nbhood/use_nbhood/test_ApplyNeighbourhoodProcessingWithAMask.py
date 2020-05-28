# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Unit tests for nbhood.ApplyNeighbourhoodProcessingWithAMask."""

import unittest
from collections import OrderedDict

import iris
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from iris.coords import DimCoord

from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask

from ..nbhood.test_BaseNeighbourhoodProcessing import set_up_cube
from ...set_up_test_cubes import (
    set_up_variable_cube,
    set_up_probability_cube,
    add_coordinate,
)


def add_dimensions_to_cube(cube, new_dims):
    """
    Add additional dimensions to a cube by adding new axes to the input cube
    and concatenating them.

    Args:
        cube (iris.cube.Cube):
            The cube we want to add dimensions to.
        new_dims (dict):
            A dictionary containing the names of the dimensions you want to
            add and the number of points you want in that dimension.
            e.g {"threshold": 3, "realization": 4}
            Points in the additional dimension will be integers
            counting up from 0.
            The data will all be copies of the input cube's data.
    Returns:
        iris.cube.Cube:
            The iris cube with the additional dimensions added.
    """
    for dim_name, dim_size in new_dims.items():
        cubes = iris.cube.CubeList()
        for i in range(dim_size):
            threshold_coord = DimCoord([i], long_name=dim_name)
            threshold_cube = iris.util.new_axis(cube.copy())
            threshold_cube.add_dim_coord(threshold_coord, 0)
            cubes.append(threshold_cube)
        cube = cubes.concatenate_cube()
    return cube


def set_up_topographic_zone_cube(
    mask_data,
    topographic_zone_point,
    topographic_zone_bounds,
    num_time_points=1,
    num_grid_points=16,
    num_realization_points=1,
):
    """Function to generate a cube with a topographic_zone coordinate. This
    uses the existing functionality from the set_up_cube function."""
    mask_cube = set_up_cube(
        zero_point_indices=((0, 0, 0, 0),),
        num_time_points=num_time_points,
        num_grid_points=num_grid_points,
        num_realization_points=num_realization_points,
    )
    mask_cube = iris.util.squeeze(mask_cube)
    mask_cube.data = mask_data
    mask_cube.long_name = "Topography mask"
    coord_name = "topographic_zone"
    threshold_coord = iris.coords.AuxCoord(
        topographic_zone_point, bounds=topographic_zone_bounds, long_name=coord_name
    )
    mask_cube.add_aux_coord(threshold_coord)
    mask_cube.attributes["Topographical Type"] = "Land"
    return mask_cube


class Test__init__(unittest.TestCase):

    """Test the __init__ method of ApplyNeighbourhoodProcessingWithAMask."""

    def test_basic(self):
        """Test that the __init__ method returns the expected string."""
        coord_for_masking = "topographic_zone"
        radii = 2000
        result = ApplyNeighbourhoodProcessingWithAMask(coord_for_masking, radii)
        msg = (
            "<ApplyNeighbourhoodProcessingWithAMask: coord_for_masking: "
            "topographic_zone, neighbourhood_method: square, radii: 2000, "
            "lead_times: None, collapse_weights: None, weighted_mode: True, "
            "sum_or_fraction: fraction, re_mask: False>"
        )
        self.assertEqual(str(result), msg)


class Test__repr__(unittest.TestCase):

    """Test the __repr__ method of ApplyNeighbourhoodProcessingWithAMask."""

    def test_basic(self):
        """Test that the __repr__ method returns the expected string."""
        coord_for_masking = "topographic_zone"
        radii = 2000
        result = str(ApplyNeighbourhoodProcessingWithAMask(coord_for_masking, radii))
        msg = (
            "<ApplyNeighbourhoodProcessingWithAMask: coord_for_masking: "
            "topographic_zone, neighbourhood_method: square, radii: 2000, "
            "lead_times: None, collapse_weights: None, weighted_mode: True, "
            "sum_or_fraction: fraction, re_mask: False>"
        )
        self.assertEqual(result, msg)


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
                [[1, 1, 0,], [1, 1, 0,], [0, 0, 0,],],
                [[0, 0, 1,], [0, 1, 1,], [1, 1, 0,],],
                [[0, 0, 0,], [0, 0, 0,], [0, 0, 1,],],
            ],
            dtype=np.float32,
        )
        cube = set_up_probability_cube(
            np.ones((1, 3, 3), dtype=np.float32), [278.15], spatial_grid="equalarea",
        )
        self.cube = add_coordinate(cube, [50, 100, 150], "topographic_zone", "m")
        self.cube = iris.util.squeeze(self.cube)
        self.cube.data = data
        # Set up weights cubes
        weights_data = np.array(
            [
                [[1, 1, 0,], [1, 0.5, 0,], [0, 0, 0,],],
                [[0, 0, 1,], [0, 0.5, 0.75,], [1, 0.75, 0.5,],],
                [[0, 0, 0,], [0, 0, 0.25,], [0, 0.25, 0.5,],],
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
        """Test we get a the expected reult with a simple collapse"""
        expected_data = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0.5]], dtype=np.float32
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data, expected_data)
        self.assertNotIn("topographic_zone", result.coords())

    def test_renormalise_when_missing_data(self):
        """
        Test we get a the expected reult when the weights need renormalising
        to account for missing data that comes from neighbourhood processing
        where there are no points to process in a given band for a given point.
        The expected behaviour is that the weights are renormalised and the
        result for that point takes 100% of the band with a valid value in.
        """
        data = np.array(
            [
                [[1, 1, 0,], [1, 1, 0,], [0, 0, 0,],],
                [[0, 0, 1,], [0, 1, 1,], [1, 1, 0,],],
                [[0, 0, 0,], [0, 0, 0,], [0, 0, np.nan,],],
            ],
            dtype=np.float32,
        )
        self.cube.data = data
        expected_data = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0]], dtype=np.float32
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data, expected_data)
        self.assertNotIn("topographic_zone", result.coords())

    def test_masked_weights_data(self):
        """Test points where weights are masked.
           Covers the case where sea points may be masked out so
           they aren't neighbourhood processed."""
        self.weights_cube.data[:, 0, 0] = np.nan
        self.weights_cube.data = np.ma.masked_invalid(self.weights_cube.data)
        expected_data = np.array(
            [[np.nan, 1.0, 1.0], [1.0, 1.0, 0.75], [1.0, 0.75, 0.5]], dtype=np.float32
        )
        expected_mask = np.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        plugin = ApplyNeighbourhoodProcessingWithAMask(
            "topographic_zone", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data.data, expected_data)
        assert_allclose(result.data.mask, expected_mask)
        self.assertNotIn("topographic_zone", result.coords())

    def test_masked_weights_and_missing(self):
        """Test masked weights, and one point has nan in from
           neighbourhood processing.
           Covers the case where sea points may be masked out so
           they aren't neighbourhood processed. The nan point comes
           from neighbourhood processing where it has found no points to
           process for a given point in a given band."""
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
            "topographic_zone", 2000, collapse_weights=self.weights_cube
        )
        result = plugin.collapse_mask_coord(self.cube)
        assert_allclose(result.data.data, expected_data)
        assert_allclose(result.data.mask, expected_mask)
        self.assertNotIn("topographic_zone", result.coords())


class Test_process(unittest.TestCase):

    """Test the process method of ApplyNeighbourhoodProcessingWithAMask."""

    def setUp(self):
        """
        Set up a cube with a single threshold to be processed and corresponding
        mask and weights cubes.
        """
        data = np.array([[[1, 1, 1,], [1, 1, 0,], [0, 0, 0,],],], dtype=np.float32,)
        self.cube = set_up_probability_cube(data, [278.15], spatial_grid="equalarea",)

        # Set up mask cube, with one masked sea point.
        mask_data = np.array(
            [
                [[np.nan, 1, 0,], [1, 1, 0,], [0, 0, 0,],],
                [[np.nan, 0, 1,], [0, 0, 1,], [1, 1, 0,],],
                [[np.nan, 0, 0,], [0, 0, 0,], [0, 0, 1,],],
            ],
            dtype=np.float32,
        )
        mask_data = np.ma.masked_invalid(mask_data)
        print(mask_data)
        print(mask_data.data)
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            name="topographic_zone_weights",
            units="1",
            spatial_grid="equalarea",
        )
        self.mask_cube = add_coordinate(cube, [50, 100, 150], "topographic_zone", "m")
        self.mask_cube.data = mask_data

        # Set up weights cubes, with one masked sea point.
        weights_data = np.array(
            [
                [[np.nan, 1, 0,], [1, 0.5, 0,], [0, 0, 0,],],
                [[np.nan, 0, 1,], [0, 0.5, 0.75,], [1, 0.75, 0.5,],],
                [[np.nan, 0, 0,], [0, 0, 0.25,], [0, 0.25, 0.5,],],
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

    def test_basic_no_collapse(self):
        """Test process for a cube with 1 threshold and no collapse.
        This test shows the result of neighbourhood processing the same input
        data three times with the three different masks for the different
        topographic zones."""
        plugin = ApplyNeighbourhoodProcessingWithAMask("topographic_zone", 2000)
        result = plugin(self.cube, self.mask_cube)
        # cube
        # [[[1, 1, 1,],
        #   [1, 1, 0,],
        #   [0, 0, 0,],],]
        #      mask       [
        #                 [0, 1, 0,],
        #                  [1, 1, 0,],
        #                  [0, 0, 0,],],
        #                 [[np.nan, 0, 1,],
        #                  [0, 0, 1,],
        #                  [1, 1, 0,],],
        #                 [[np.nan, 0, 0,],
        #                  [0, 0, 0,],
        #                  [0, 0, 1,],],
        #             ],
        # print(result.data)
        expected_result = np.array(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.5], [0.0, 0.0, 0.33333334], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, np.nan], [0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

    # def setUp(self):
    #     """Set up a cube."""
    #     self.cube = set_up_cube(zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
    #     # The neighbourhood code adds bounds to the coordinates if they are
    #     # not present so add them now to make it easier to compare input and
    #     # output from the plugin.
    #     self.cube.coord("projection_x_coordinate").guess_bounds()
    #     self.cube.coord("projection_y_coordinate").guess_bounds()
    #     self.cube = iris.util.squeeze(self.cube)
    #     mask_data = np.array(
    #         [
    #             [
    #                 [1, 0, 0, 0, 0],
    #                 [1, 1, 0, 0, 0],
    #                 [1, 1, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #             ],
    #             [
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 1, 1, 0],
    #                 [0, 0, 1, 1, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #             ],
    #             [
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 1, 1],
    #                 [0, 0, 0, 1, 1],
    #             ],
    #         ]
    #     )
    #     weights_data = np.array(
    #         [
    #             [
    #                 [0.8, 0.7, 0.0, 0.0, 0.0],
    #                 [0.7, 0.3, 0.0, 0.0, 0.0],
    #                 [0.3, 0.1, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0],
    #             ],
    #             [
    #                 [0.2, 0.3, 1.0, 1.0, 1.0],
    #                 [0.3, 0.7, 1.0, 1.0, 1.0],
    #                 [0.7, 0.9, 1.0, 1.0, 1.0],
    #                 [1.0, 1.0, 1.0, 0.9, 0.5],
    #                 [1.0, 1.0, 1.0, 0.6, 0.2],
    #             ],
    #             [
    #                 [0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.1, 0.5],
    #                 [0.0, 0.0, 0.0, 0.4, 0.8],
    #             ],
    #         ]
    #     )
    #     topographic_zone_points = [50, 150, 250]
    #     topographic_zone_bounds = [[0, 100], [100, 200], [200, 300]]
    #
    #     mask_cubes = iris.cube.CubeList([])
    #     weights_cubes = iris.cube.CubeList([])
    #     for mdata, wdata, pnt, bnd in zip(
    #         mask_data, weights_data, topographic_zone_points, topographic_zone_bounds
    #     ):
    #         for data, cubes in [(mdata, mask_cubes), (wdata, weights_cubes)]:
    #             cubes.append(
    #                 set_up_topographic_zone_cube(data, pnt, bnd, num_grid_points=5)
    #             )
    #     self.mask_cube = mask_cubes.merge_cube()
    #     self.weights_cube = weights_cubes.merge_cube()
    #
    # def test_basic(self):
    #     """Test that the expected result is returned, when the
    #     topographic_zone coordinate is iterated over."""
    #     expected = np.array(
    #         [
    #             [
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #             ],
    #             [
    #                 [np.nan, 1.00, 1.00, 1.00, 1.00],
    #                 [np.nan, 0.50, 0.75, 0.75, 1.00],
    #                 [np.nan, 0.50, 0.75, 0.75, 1.00],
    #                 [np.nan, 0.00, 0.50, 0.50, 1.00],
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #             ],
    #             [
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #                 [np.nan, np.nan, 1.00, 1.00, 1.00],
    #                 [np.nan, np.nan, 1.00, 1.00, 1.00],
    #                 [np.nan, np.nan, 1.00, 1.00, 1.00],
    #             ],
    #         ]
    #     )
    #     coord_for_masking = "topographic_zone"
    #     radii = 2000
    #     num_zones = len(self.mask_cube.coord(coord_for_masking).points)
    #     expected_shape = tuple([num_zones] + list(self.cube.data.shape))
    #     result = ApplyNeighbourhoodProcessingWithAMask(coord_for_masking, radii)(
    #         self.cube, self.mask_cube
    #     )
    #     self.assertEqual(result.data.shape, expected_shape)
    #     assert_allclose(result.data, expected)
    #
    # def test_collapse_preserve_dimensions_input(self):
    #     """Test that the dimensions on the output cube are the same as the
    #        input cube, apart from the additional topographic zone coordinate.
    #     """
    #     self.cube.remove_coord("realization")
    #     cube = add_dimensions_to_cube(
    #         self.cube, OrderedDict([("threshold", 3), ("realization", 4)])
    #     )
    #     coord_for_masking = "topographic_zone"
    #     radii = 2000
    #     uncollapsed_result = None
    #     mask_data = np.array(
    #         [
    #             [
    #                 [1, 1, 1, 1, 1],
    #                 [1, 1, 0, 0, 0],
    #                 [1, 1, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #             ],
    #             [
    #                 [0, 0, 0, 0, 1],
    #                 [0, 0, 1, 1, 1],
    #                 [0, 0, 1, 1, 1],
    #                 [1, 1, 0, 1, 0],
    #                 [1, 1, 0, 0, 0],
    #             ],
    #             [
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 1, 1],
    #                 [0, 0, 1, 1, 1],
    #                 [0, 0, 1, 1, 1],
    #             ],
    #         ]
    #     )
    #     mask_cube = self.mask_cube.copy(mask_data)
    #
    #     for collapse_weights in [None, self.weights_cube]:
    #         coords_order = [
    #             "realization",
    #             "threshold",
    #             "topographic_zone",
    #             "projection_y_coordinate",
    #             "projection_x_coordinate",
    #         ]
    #         plugin = ApplyNeighbourhoodProcessingWithAMask(
    #             coord_for_masking, radii, collapse_weights=collapse_weights,
    #         )
    #         result = plugin(cube, mask_cube)
    #         expected_dims = list(cube.dim_coords)
    #         if collapse_weights is None:
    #             uncollapsed_result = result
    #             expected_dims.insert(2, self.mask_cube.coord("topographic_zone"))
    #         else:
    #             coords_order.remove("topographic_zone")
    #             collapsed_result = plugin.collapse_mask_coord(uncollapsed_result)
    #             assert_allclose(result.data, collapsed_result.data)
    #         self.assertEqual(result.dim_coords, tuple(expected_dims))
    #         for dim, coord in enumerate(coords_order):
    #             self.assertEqual(result.coord_dims(coord), (dim,))
    #
    # def test_preserve_dimensions_with_single_point(self):
    #     """Test that the dimensions on the output cube are the same as the
    #        input cube, apart from the collapsed dimension.
    #        Check that a dimension coordinate with a single point is preserved
    #        and not demoted to a scalar coordinate."""
    #     self.cube.remove_coord("realization")
    #     cube = add_dimensions_to_cube(self.cube, {"threshold": 4, "realization": 1})
    #     coord_for_masking = "topographic_zone"
    #     radii = 2000
    #     result = ApplyNeighbourhoodProcessingWithAMask(coord_for_masking, radii)(
    #         cube, self.mask_cube
    #     )
    #     expected_dims = list(cube.dim_coords)
    #     expected_dims.insert(2, self.mask_cube.coord("topographic_zone"))
    #
    #     self.assertEqual(result.dim_coords, tuple(expected_dims))
    #     self.assertEqual(result.coord_dims("realization"), (0,))
    #     self.assertEqual(result.coord_dims("threshold"), (1,))
    #     self.assertEqual(result.coord_dims("topographic_zone"), (2,))
    #     self.assertEqual(result.coord_dims("projection_y_coordinate"), (3,))
    #     self.assertEqual(result.coord_dims("projection_x_coordinate"), (4,))
    #
    # def test_identical_slices(self):
    #     """Test that identical successive slices of the cube produce
    #        identical results."""
    #     expected = np.array(
    #         [
    #             [
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [1.00, 1.00, 1.00, np.nan, np.nan],
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #             ],
    #             [
    #                 [np.nan, 1.00, 1.00, 1.00, 1.00],
    #                 [np.nan, 0.50, 0.75, 0.75, 1.00],
    #                 [np.nan, 0.50, 0.75, 0.75, 1.00],
    #                 [np.nan, 0.00, 0.50, 0.50, 1.00],
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #             ],
    #             [
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #                 [np.nan, np.nan, np.nan, np.nan, np.nan],
    #                 [np.nan, np.nan, 1.00, 1.00, 1.00],
    #                 [np.nan, np.nan, 1.00, 1.00, 1.00],
    #                 [np.nan, np.nan, 1.00, 1.00, 1.00],
    #             ],
    #         ]
    #     )
    #     cube = set_up_cube(
    #         zero_point_indices=((0, 0, 2, 2), (1, 0, 2, 2)),
    #         num_grid_points=5,
    #         num_realization_points=2,
    #     )
    #     # The neighbourhood code adds bounds to the coordinates if they are
    #     # not present so add them now to make it easier to compare input and
    #     # output from the plugin.
    #     cube.coord("projection_x_coordinate").guess_bounds()
    #     cube.coord("projection_y_coordinate").guess_bounds()
    #     cube = iris.util.squeeze(cube)
    #     coord_for_masking = "topographic_zone"
    #     radii = 2000
    #     num_zones = len(self.mask_cube.coord(coord_for_masking).points)
    #     expected_shape = tuple(
    #         [cube.data.shape[0], num_zones] + list(cube.data.shape[1:])
    #     )
    #     result = ApplyNeighbourhoodProcessingWithAMask(coord_for_masking, radii)(
    #         cube, self.mask_cube
    #     )
    #     self.assertEqual(result.data.shape, expected_shape)
    #     for realization_slice in result.slices_over("realization"):
    #         assert_allclose(realization_slice.data, expected)


if __name__ == "__main__":
    unittest.main()

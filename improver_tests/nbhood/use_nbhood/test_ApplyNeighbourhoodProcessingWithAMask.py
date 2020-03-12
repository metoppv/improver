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
"""Unit tests for nbhood.ApplyNeighbourhoodProcessingWithAMask."""

import unittest
from collections import OrderedDict

import iris
import numpy as np
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask

from ..nbhood.test_BaseNeighbourhoodProcessing import set_up_cube


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
        mask_data, topographic_zone_point, topographic_zone_bounds,
        num_time_points=1, num_grid_points=16, num_realization_points=1):
    """Function to generate a cube with a topographic_zone coordinate. This
    uses the existing functionality from the set_up_cube function."""
    mask_cube = set_up_cube(
        zero_point_indices=((0, 0, 0, 0),),
        num_time_points=num_time_points, num_grid_points=num_grid_points,
        num_realization_points=num_realization_points)
    mask_cube = iris.util.squeeze(mask_cube)
    mask_cube.data = mask_data
    mask_cube.long_name = 'Topography mask'
    coord_name = 'topographic_zone'
    threshold_coord = iris.coords.AuxCoord(
        topographic_zone_point, bounds=topographic_zone_bounds,
        long_name=coord_name)
    mask_cube.add_aux_coord(threshold_coord)
    mask_cube.attributes['Topographical Type'] = "Land"
    return mask_cube


class Test__init__(IrisTest):

    """Test the __init__ method of ApplyNeighbourhoodProcessingWithAMask."""

    def test_basic(self):
        """Test that the __init__ method returns the expected string."""
        coord_for_masking = "topographic_zone"
        radii = 2000
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii)
        msg = ("<ApplyNeighbourhoodProcessingWithAMask: coord_for_masking: "
               "topographic_zone, neighbourhood_method: square, radii: 2000, "
               "lead_times: None, weighted_mode: True, "
               "sum_or_fraction: fraction, re_mask: False>")
        self.assertEqual(str(result), msg)


class Test__repr__(IrisTest):

    """Test the __repr__ method of ApplyNeighbourhoodProcessingWithAMask."""

    def test_basic(self):
        """Test that the __repr__ method returns the expected string."""
        coord_for_masking = "topographic_zone"
        radii = 2000
        result = str(ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii))
        msg = ("<ApplyNeighbourhoodProcessingWithAMask: coord_for_masking: "
               "topographic_zone, neighbourhood_method: square, radii: 2000, "
               "lead_times: None, weighted_mode: True, "
               "sum_or_fraction: fraction, re_mask: False>")
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the process method of ApplyNeighbourhoodProcessingWithAMask."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
        # The neighbourhood code adds bounds to the coordinates if they are
        # not present so add them now to make it easier to compare input and
        # output from the plugin.
        self.cube.coord("projection_x_coordinate").guess_bounds()
        self.cube.coord("projection_y_coordinate").guess_bounds()
        self.cube = iris.util.squeeze(self.cube)
        mask_data = np.array([[[1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]],
                              [[0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]],
                              [[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1]]])
        topographic_zone_points = [50, 150, 250]
        topographic_zone_bounds = [[0, 100], [100, 200], [200, 300]]

        mask_cubes = iris.cube.CubeList([])
        for data, point, bounds in zip(mask_data, topographic_zone_points,
                                       topographic_zone_bounds):
            mask_cubes.append(
                set_up_topographic_zone_cube(
                    data, point, bounds, num_grid_points=5))
        self.mask_cube = mask_cubes.merge_cube()

    def test_basic(self):
        """Test that the expected result is returned, when the
        topographic_zone coordinate is iterated over."""
        expected = np.array(
            [[[1.00, 1.00, 1.00, np.nan, np.nan],
              [1.00, 1.00, 1.00, np.nan, np.nan],
              [1.00, 1.00, 1.00, np.nan, np.nan],
              [1.00, 1.00, 1.00, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, 1.00, 1.00, 1.00, 1.00],
              [np.nan, 0.50, 0.75, 0.75, 1.00],
              [np.nan, 0.50, 0.75, 0.75, 1.00],
              [np.nan, 0.00, 0.50, 0.50, 1.00],
              [np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, 1.00, 1.00, 1.00],
              [np.nan, np.nan, 1.00, 1.00, 1.00],
              [np.nan, np.nan, 1.00, 1.00, 1.00]]])
        coord_for_masking = "topographic_zone"
        radii = 2000
        num_zones = len(self.mask_cube.coord(coord_for_masking).points)
        expected_shape = tuple(
            [num_zones] + list(self.cube.data.shape))
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii)(self.cube, self.mask_cube)
        self.assertEqual(result.data.shape, expected_shape)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_preserve_dimensions_input(self):
        """Test that the dimensions on the output cube are the same as the
           input cube, apart from the additional topographic zone coordinate.
        """
        self.cube.remove_coord("realization")
        cube = add_dimensions_to_cube(
            self.cube, OrderedDict([("threshold", 3), ("realization", 4)]))
        coord_for_masking = "topographic_zone"
        radii = 2000
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii)(cube, self.mask_cube)
        expected_dims = list(cube.dim_coords)
        expected_dims.insert(2, self.mask_cube.coord("topographic_zone"))
        self.assertEqual(result.dim_coords, tuple(expected_dims))
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertEqual(result.coord_dims("threshold"), (1,))
        self.assertEqual(result.coord_dims("topographic_zone"), (2,))
        self.assertEqual(result.coord_dims("projection_y_coordinate"), (3,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (4,))

    def test_preserve_dimensions_with_single_point(self):
        """Test that the dimensions on the output cube are the same as the
           input cube, apart from the collapsed dimension.
           Check that a dimension coordinate with a single point is preserved
           and not demoted to a scalar coordinate."""
        self.cube.remove_coord("realization")
        cube = add_dimensions_to_cube(self.cube,
                                      {"threshold": 4, "realization": 1})
        coord_for_masking = "topographic_zone"
        radii = 2000
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii)(cube, self.mask_cube)
        expected_dims = list(cube.dim_coords)
        expected_dims.insert(2, self.mask_cube.coord("topographic_zone"))

        self.assertEqual(result.dim_coords, tuple(expected_dims))
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertEqual(result.coord_dims("threshold"), (1,))
        self.assertEqual(result.coord_dims("topographic_zone"), (2,))
        self.assertEqual(result.coord_dims("projection_y_coordinate"), (3,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (4,))

    def test_identical_slices(self):
        """Test that identical successive slices of the cube produce
           identical results."""
        expected = np.array(
            [[[1.00, 1.00, 1.00, np.nan, np.nan],
              [1.00, 1.00, 1.00, np.nan, np.nan],
              [1.00, 1.00, 1.00, np.nan, np.nan],
              [1.00, 1.00, 1.00, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, 1.00, 1.00, 1.00, 1.00],
              [np.nan, 0.50, 0.75, 0.75, 1.00],
              [np.nan, 0.50, 0.75, 0.75, 1.00],
              [np.nan, 0.00, 0.50, 0.50, 1.00],
              [np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, 1.00, 1.00, 1.00],
              [np.nan, np.nan, 1.00, 1.00, 1.00],
              [np.nan, np.nan, 1.00, 1.00, 1.00]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (1, 0, 2, 2)), num_grid_points=5,
            num_realization_points=2)
        # The neighbourhood code adds bounds to the coordinates if they are
        # not present so add them now to make it easier to compare input and
        # output from the plugin.
        cube.coord("projection_x_coordinate").guess_bounds()
        cube.coord("projection_y_coordinate").guess_bounds()
        cube = iris.util.squeeze(cube)
        coord_for_masking = "topographic_zone"
        radii = 2000
        num_zones = len(self.mask_cube.coord(coord_for_masking).points)
        expected_shape = tuple(
            [cube.data.shape[0], num_zones] + list(cube.data.shape[1:])
        )
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii)(cube, self.mask_cube)
        self.assertEqual(result.data.shape, expected_shape)
        for realization_slice in result.slices_over("realization"):
            self.assertArrayAlmostEqual(realization_slice.data, expected)


if __name__ == '__main__':
    unittest.main()

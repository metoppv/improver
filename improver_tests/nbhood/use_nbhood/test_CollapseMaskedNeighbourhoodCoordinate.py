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
"""Unit tests for nbhood.CollapseMaskedNeighbourhoodCoordinate."""


import unittest

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.nbhood.use_nbhood import CollapseMaskedNeighbourhoodCoordinate

from ..nbhood.test_BaseNeighbourhoodProcessing import set_up_cube
from .test_ApplyNeighbourhoodProcessingWithAMask import (
    add_dimensions_to_cube, set_up_topographic_zone_cube)


class Test__repr__(IrisTest):

    """Test the __repr__ method of CollapseMaskedNeighbourhoodCoordinate."""

    def test_basic(self):
        """Test that the __repr__ method returns the expected string."""
        coord_masked = "topographic_zone"
        weights = iris.cube.Cube(np.array([1.0]), long_name="weights")
        result = str(CollapseMaskedNeighbourhoodCoordinate(
            coord_masked, weights))
        msg = ("<ApplyNeighbourhoodProcessingWithAMask:"
               " coord_masked: topographic_zone,"
               " weights: weights / (unknown)"
               "                 (-- : 1)>")
        self.assertEqual(result, msg)


class Test_renormalize_weights(IrisTest):

    """Test the renormalize_weights function."""

    def setUp(self):
        """Set up a weights cube and default plugin instance."""
        self.mask = np.array([[[1, 1, 1, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1]],
                              [[1, 1, 1, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1]],
                              [[1, 1, 1, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1]]])
        weights_data = np.array([[[0.8, 0.7, 0.0, 0.0, 0.0],
                                  [0.7, 0.3, 0.0, 0.0, 0.0],
                                  [0.3, 0.1, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0]],
                                 [[0.2, 0.3, 1.0, 1.0, 1.0],
                                  [0.3, 0.7, 1.0, 1.0, 1.0],
                                  [0.7, 0.9, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 0.9, 0.5],
                                  [1.0, 1.0, 1.0, 0.6, 0.2]],
                                 [[0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.1, 0.5],
                                  [0.0, 0.0, 0.0, 0.4, 0.8]]])
        topographic_zone_points = [50, 150, 250]
        topographic_zone_bounds = [[0, 100], [100, 200], [200, 300]]

        weights_cubes = iris.cube.CubeList([])
        for data, point, bounds in zip(weights_data, topographic_zone_points,
                                       topographic_zone_bounds):
            weights_cubes.append(
                set_up_topographic_zone_cube(
                    data, point, bounds, num_grid_points=5))
        self.weights_cube = weights_cubes.merge_cube()
        self.plugin = CollapseMaskedNeighbourhoodCoordinate("topographic_zone",
                                                            self.weights_cube)

    def test_basic(self):
        """Test weights_cube is still a cube after the function call"""
        nbhooded_cube = self.weights_cube.copy()
        self.plugin.renormalize_weights(nbhooded_cube)
        self.assertIsInstance(self.weights_cube, iris.cube.Cube)

    def test_no_NaNs_in_nbhooded_cube(self):
        """No NaNs in the neighbourhood cube, so no renormalization is
           needed"""
        nbhooded_cube = self.weights_cube.copy()
        expected_weights = self.weights_cube.data.copy()
        self.plugin.renormalize_weights(nbhooded_cube)
        self.assertArrayAlmostEqual(expected_weights, self.weights_cube.data)

    def test_all_NaNs_in_nbhooded_cube(self):
        """Test an error is raised when all NaNs in the nbhood cube so cannot
           have any sensible weights."""
        nbhood_data = np.empty(self.weights_cube.data.shape)
        nbhood_data[:] = np.nan
        nbhooded_cube = self.weights_cube.copy(nbhood_data)
        message = "Sum of weights must be > 0.0"
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.renormalize_weights(nbhooded_cube)

    def test_no_NaNs_in_nbhooded_cube_and_masked_weights(self):
        """No NaNs in the neighbourhood cube, but masked weights."""
        nbhooded_cube = self.weights_cube.copy()

        self.weights_cube.data = np.ma.masked_array(self.weights_cube.data,
                                                    mask=self.mask)
        expected_weights = self.weights_cube.data.copy()
        self.plugin.renormalize_weights(nbhooded_cube)
        self.assertArrayAlmostEqual(expected_weights.data,
                                    self.weights_cube.data.data)
        self.assertArrayAlmostEqual(expected_weights.mask,
                                    self.weights_cube.data.mask)

    def test_some_NaNs_in_nbhooded_cube(self):
        """Some NaNs in the neighbourhood cube, so renormalization is needed"""
        nbhood_data = np.ones((3, 5, 5))
        nbhood_data[0, 0:2, 0] = np.nan
        nbhood_data[2, 2:4, 4] = np.nan

        expected_weights = np.array([[[0.0, 0.7, 0.0, 0.0, 0.0],
                                      [0.0, 0.3, 0.0, 0.0, 0.0],
                                      [0.3, 0.1, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]],
                                     [[1.0, 0.3, 1.0, 1.0, 1.0],
                                      [1.0, 0.7, 1.0, 1.0, 1.0],
                                      [0.7, 0.9, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 0.9, 1.0],
                                      [1.0, 1.0, 1.0, 0.6, 0.2]],
                                     [[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.1, 0.0],
                                      [0.0, 0.0, 0.0, 0.4, 0.8]]])
        nbhooded_cube = self.weights_cube.copy(nbhood_data)
        self.plugin.renormalize_weights(nbhooded_cube)
        self.assertArrayAlmostEqual(expected_weights, self.weights_cube.data)

    def test_some_NaNs_in_nbhooded_cube_and_masked_weights(self):
        """Some NaNs in the neighbourhood cube, so renormalization is needed.
           As the points with NaNs in the neighbourhood cube are masked in the
           weights cube then they are not renormalizeed."""
        nbhood_data = np.ones((3, 5, 5))
        nbhood_data[0, 0:2, 0] = np.nan
        nbhood_data[2, 2:4, 4] = np.nan
        nbhooded_cube = self.weights_cube.copy()
        self.weights_cube.data = np.ma.masked_array(self.weights_cube.data,
                                                    mask=self.mask)
        expected_weights = self.weights_cube.data.copy()
        self.plugin.renormalize_weights(nbhooded_cube)
        self.assertArrayAlmostEqual(expected_weights.data,
                                    self.weights_cube.data.data)
        self.assertArrayAlmostEqual(expected_weights.mask,
                                    self.weights_cube.data.mask)

    def test_coordinate_not_found(self):
        """Coordinate not found on the cube."""
        nbhooded_cube = self.weights_cube.copy()
        plugin = CollapseMaskedNeighbourhoodCoordinate("kitten",
                                                       self.weights_cube)
        message = "Expected to find exactly 1 .* coordinate, but found none."
        with self.assertRaisesRegex(CoordinateNotFoundError, message):
            plugin.renormalize_weights(nbhooded_cube)

    def test_normalizing_along_another_axis_with_error(self):
        """Normalizing along another axis, when raises error.
           In this case we are normalizing along an axis where in some places
           the sum along that axis is zero."""
        nbhooded_cube = self.weights_cube.copy()
        plugin = CollapseMaskedNeighbourhoodCoordinate(
            "projection_x_coordinate", self.weights_cube)
        message = "Sum of weights must be > 0.0"
        with self.assertRaisesRegex(ValueError, message):
            plugin.renormalize_weights(nbhooded_cube)

    def test_normalizing_along_another_axis(self):
        """Normalizing along another axis, when this is a valid thing to do.
           This is normalizing along the rows of the input weights."""
        input_weights = np.array([[[0.0, 0.7, 0.0, 0.0, 0.0],
                                   [0.0, 0.3, 0.0, 0.0, 0.0],
                                   [0.3, 0.1, 0.0, 0.0, 0.0],
                                   [0.2, 0.0, 0.0, 0.0, 0.0],
                                   [0.1, 0.0, 0.0, 0.0, 0.0]],
                                  [[1.0, 0.0, 1.0, 1.0, 1.0],
                                   [1.0, 0.0, 1.0, 1.0, 1.0],
                                   [0.0, 0.0, 1.0, 1.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.9, 0.1],
                                   [1.0, 1.0, 1.0, 0.0, 0.0]],
                                  [[0.2, 0.0, 0.0, 0.0, 0.0],
                                   [0.3, 0.0, 0.0, 0.0, 0.0],
                                   [0.1, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.1, 0.0],
                                   [0.0, 0.0, 0.0, 0.4, 0.8]]])
        expected_weights = np.array([[[0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.75, 0.25, 0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0, 0.0]],
                                     [[0.25, 0.0, 0.25, 0.25, 0.25],
                                      [0.25, 0.0, 0.25, 0.25, 0.25],
                                      [0.0, 0.0, 0.333333, 0.333333, 0.333333],
                                      [0.0, 0.0, 0.0, 0.9, 0.1],
                                      [0.333333, 0.333333, 0.333333, 0.0, 0.0]
                                      ],
                                     [[1.0, 0.0, 0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.333333, 0.666667]]])
        weights_cube = self.weights_cube.copy(input_weights)
        nbhooded_cube = self.weights_cube.copy()
        plugin = CollapseMaskedNeighbourhoodCoordinate(
            "projection_x_coordinate", weights_cube)
        plugin.renormalize_weights(nbhooded_cube)
        self.assertArrayAlmostEqual(expected_weights, weights_cube.data)


class Test_process(IrisTest):

    """Test the process method of CollapseMaskedNeighbourhoodCoordinate"""

    def setUp(self):
        """Set up a weights cube and default plugin to use in unittests."""
        weights_data = np.array([[[0.8, 0.7, 0.0, 0.0, 0.0],
                                  [0.7, 0.3, 0.0, 0.0, 0.0],
                                  [0.3, 0.1, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0]],
                                 [[0.2, 0.3, 1.0, 1.0, 1.0],
                                  [0.3, 0.7, 1.0, 1.0, 1.0],
                                  [0.7, 0.9, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 0.9, 0.5],
                                  [1.0, 1.0, 1.0, 0.6, 0.2]],
                                 [[0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.1, 0.5],
                                  [0.0, 0.0, 0.0, 0.4, 0.8]]])
        topographic_zone_points = [50, 150, 250]
        topographic_zone_bounds = [[0, 100], [100, 200], [200, 300]]

        weights_cubes = iris.cube.CubeList([])
        for data, point, bounds in zip(weights_data, topographic_zone_points,
                                       topographic_zone_bounds):
            weights_cubes.append(
                set_up_topographic_zone_cube(
                    data, point, bounds, num_grid_points=5))
        self.weights_cube = weights_cubes.merge_cube()
        nbhood_data = np.ones((3, 5, 5))
        nbhood_data[0] = nbhood_data[0]*0.1
        nbhood_data[1] = nbhood_data[1]*0.2
        nbhood_data[2] = nbhood_data[2]*0.1
        self.nbhooded_cube = self.weights_cube.copy(nbhood_data)
        self.plugin = CollapseMaskedNeighbourhoodCoordinate("topographic_zone",
                                                            self.weights_cube)

    def test_basic(self):
        """Test that a cube is returned from the process method"""
        result = self.plugin.process(self.nbhooded_cube)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_no_NaNs_in_nbhooded_cube(self):
        """No NaNs in the neighbourhood cube, so no renormalization is needed.
           The neighbourhood data has all values of 0.1 for the top and bottom
           bands and 0.2 for points in the middle band. The weights are used
           to calculate the weighted average amongst the bands."""

        expected_result = np.array([[0.12, 0.13, 0.2, 0.2, 0.2],
                                    [0.13, 0.17, 0.2, 0.2, 0.2],
                                    [0.17, 0.19, 0.2, 0.2, 0.2],
                                    [0.2, 0.2, 0.2, 0.19, 0.15],
                                    [0.2, 0.2, 0.2, 0.16, 0.12]])

        result = self.plugin.process(self.nbhooded_cube)
        self.assertArrayAlmostEqual(expected_result, result.data)

    def test_some_NaNs_in_nbhooded_cube(self):
        """Some NaNs in the neighbourhood cube, so renormalizing is needed.
           The neighbourhood data has all values of 0.1 for the top and bottom
           bands and 0.2 for points in the middle band. The weights are used
           to calculate the weighted average amongst the bands. This is the
           same test as above, but a few of the weights are modified where
           there are NaNs in the neighbourhood data, giving slightly different
           results."""
        self.nbhooded_cube.data[0, 0:2, 0] = np.nan
        self.nbhooded_cube.data[2, 3:, 4] = np.nan
        expected_result = np.array([[0.2, 0.13, 0.2, 0.2, 0.2],
                                    [0.2, 0.17, 0.2, 0.2, 0.2],
                                    [0.17, 0.19, 0.2, 0.2, 0.2],
                                    [0.2, 0.2, 0.2, 0.19, 0.2],
                                    [0.2, 0.2, 0.2, 0.16, 0.2]])
        result = self.plugin.process(self.nbhooded_cube)
        self.assertArrayAlmostEqual(expected_result, result.data)

    def test_landsea_mask_in_weights(self):
        """Test that the final result from collapsing the nbhood retains the
           mask from weights input."""
        self.nbhooded_cube.data[0, 0:2, 0] = np.nan
        self.nbhooded_cube.data[2, 3:, 4] = np.nan
        mask = np.array([[[1, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1]],
                         [[1, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1]],
                         [[1, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1]]])
        self.weights_cube.data = np.ma.masked_array(self.weights_cube.data,
                                                    mask=mask)
        plugin = CollapseMaskedNeighbourhoodCoordinate(
            "topographic_zone", weights=self.weights_cube)
        result = plugin.process(self.nbhooded_cube)
        expected_result = np.array([[0.0, 0.0, 0.0, 0.2, 0.2],
                                    [0.0, 0.0, 0.2, 0.2, 0.2],
                                    [0.0, 0.19, 0.2, 0.2, 0.2],
                                    [0.0, 0.2, 0.2, 0.19, 0.2],
                                    [0.0, 0.0, 0.0, 0.0, 0.0]])
        expected_mask = np.array([[True, True, True, False, False],
                                  [True, True, False, False, False],
                                  [True, False, False, False, False],
                                  [True, False, False, False, False],
                                  [True, True, True, True, True]])

        self.assertArrayAlmostEqual(expected_result, result.data.data)
        self.assertArrayAlmostEqual(expected_mask, result.data.mask)
        self.assertEqual(expected_mask.dtype, bool)
        self.assertIsInstance(expected_result, np.ndarray)

    def test_multidimensional_neighbourhood_input(self):
        """Test that we can collapse the right dimension when there
           are additional leading dimensions like threshold."""
        nbhooded_cube = add_dimensions_to_cube(self.nbhooded_cube,
                                               {"threshold": 3})
        expected_result = np.array([[[0.12, 0.13, 0.2, 0.2, 0.2],
                                     [0.13, 0.17, 0.2, 0.2, 0.2],
                                     [0.17, 0.19, 0.2, 0.2, 0.2],
                                     [0.2, 0.2, 0.2, 0.19, 0.15],
                                     [0.2, 0.2, 0.2, 0.16, 0.12]],
                                    [[0.12, 0.13, 0.2, 0.2, 0.2],
                                     [0.13, 0.17, 0.2, 0.2, 0.2],
                                     [0.17, 0.19, 0.2, 0.2, 0.2],
                                     [0.2, 0.2, 0.2, 0.19, 0.15],
                                     [0.2, 0.2, 0.2, 0.16, 0.12]],
                                    [[0.12, 0.13, 0.2, 0.2, 0.2],
                                     [0.13, 0.17, 0.2, 0.2, 0.2],
                                     [0.17, 0.19, 0.2, 0.2, 0.2],
                                     [0.2, 0.2, 0.2, 0.19, 0.15],
                                     [0.2, 0.2, 0.2, 0.16, 0.12]]])
        result = self.plugin.process(nbhooded_cube)
        self.assertArrayAlmostEqual(expected_result, result.data)

    def test_preserve_dimensions_input(self):
        """Test that the dimensions on the output cube are the same as the
           input cube, apart from the collapsed dimension.
           Add threshold and realization coordinates and check they are in the
           right place after collapsing the topographic_zone coordinate."""
        nbhood_cube = self.weights_cube.copy()
        nbhood_cube.remove_coord("realization")
        nbhood_cube = add_dimensions_to_cube(
            nbhood_cube, {"threshold": 3, "realization": 4})
        result = self.plugin.process(nbhood_cube)
        expected_dims = [coord for coord in nbhood_cube.dim_coords
                         if coord.long_name is not "topographic_zone"]
        self.assertEqual(result.dim_coords, tuple(expected_dims))
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertEqual(result.coord_dims("threshold"), (1,))
        self.assertEqual(result.coord_dims("projection_y_coordinate"), (2,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (3,))

    def test_preserve_dimensions_with_single_point(self):
        """Test that the dimensions on the output cube are the same as the
           input cube, appart from the collapsed dimension.
           Add threshold and realization coordinates and check they are in the
           right place after collapsing the topographic_zone coordinate.
           Check that a dimension coordinate with a single point is preserved
           and not demoted to a scalar coordinate."""
        nbhood_cube = self.weights_cube.copy()
        nbhood_cube.remove_coord("realization")
        nbhood_cube = add_dimensions_to_cube(
            nbhood_cube, {"threshold": 3, "realization": 1})
        result = self.plugin.process(nbhood_cube)
        expected_dims = [coord for coord in nbhood_cube.dim_coords
                         if coord.long_name is not "topographic_zone"]
        self.assertEqual(result.dim_coords, tuple(expected_dims))
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertEqual(result.coord_dims("threshold"), (1,))
        self.assertEqual(result.coord_dims("projection_y_coordinate"), (2,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (3,))


if __name__ == '__main__':
    unittest.main()

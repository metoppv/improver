# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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

import iris
from iris.tests import IrisTest
import numpy as np

from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube)


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
               "lead_times: None, ens_factor: 1.0, weighted_mode: True, "
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
               "lead_times: None, ens_factor: 1.0, weighted_mode: True, "
               "sum_or_fraction: fraction, re_mask: False>")
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the process method of ApplyNeighbourhoodProcessingWithAMask."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
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
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, radii).process(self.cube, self.mask_cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()

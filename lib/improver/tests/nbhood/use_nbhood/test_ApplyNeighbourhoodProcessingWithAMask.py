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

import


def set_up_topographic_zone_cube(
        mask_data, topoographic_zone_point, topographic_zone_bounds):
    mask_cube = iris.cube.Cube(mask_data, long_name='Topography mask')
    coord_name = 'topographic_zone'
    threshold_coord = iris.coords.AuxCoord(
        topographic_zone_point, bounds=topographic_bounds,
        long_name=coord_name)
    mask_cube.add_aux_coord(threshold_coord)
    mask_cube.attributes['Topographical Type'] = "Land"
    return mask_cube


class Test__init__(IrisTest):

    """Test the __init__ method of ApplyNeighbourhoodProcessingWithAMask."""

    def test_basic(self):
        coord_for_masking = "topographic_zone"
        neighbourhood_method = "square"
        radii = 2000
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, neighbourhood_method, radii)
        msg = ""
        self.assertEqual(str(result), msg)


class Test__repr__(IrisTest):

    """Test the __repr__ mtehod of ApplyNeighbourhoodProcessingWithAMask."""

    def test_basic(self):
        coord_for_masking = "topographic_zone"
        neighbourhood_method = "square"
        radii = 2000
        result = str(ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, neighbourhood_method, radii))
        msg = ""
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the process method of ApplyNeighbourhoodProcessingWithAMask."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
        mask_data = np.array([[[1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 1, 0, 0, 0]
                               [0, 0, 0, 0, 0]
                               [0, 0, 0, 0, 0]],
                              [[0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 1, 1, 0]
                               [0, 0, 0, 0, 0]
                               [0, 0, 0, 0, 0]],
                              [[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]
                               [0, 0, 0, 1, 1]
                               [0, 0, 0, 1, 1]]])
        topographic_zone_points = [50, 150, 250]
        topographic_zone_bounds = [[0, 100], [100, 200], [200, 300]]

        mask_cubes = iris.cube.CubeList([])
        for data, point, bounds in zip(mask_data, topographic_zone_points,
                                       topographic_zone_bounds):
            mask_cubes.append(
                set_up_topographic_zone_cube(data, point, bounds))
        self.mask_cube = mask_cubes.concatenate_cube()

    def test_basic(self):
        coord_for_masking = "topographic_zone"
        neighbourhood_method = "square"
        radii = 2000
        result = ApplyNeighbourhoodProcessingWithAMask(
            coord_for_masking, neighbourhood_method, radii
            ).process(self.cube, self.mask_cube)
        





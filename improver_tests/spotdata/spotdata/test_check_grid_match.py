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
"""Unit tests for check_grid_match function."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.spot_extraction import check_grid_match

from ...set_up_test_cubes import set_up_variable_cube


class Test_check_grid_match(IrisTest):

    """Test the check_grid_match function."""

    def setUp(self):
        """Set up cubes for use in testing."""

        data = np.ones(9).reshape(3, 3).astype(np.float32)
        self.reference_cube = set_up_variable_cube(data,
                                                   spatial_grid="equalarea")
        self.cube1 = self.reference_cube.copy()
        self.cube2 = self.reference_cube.copy()
        self.unmatched_cube = set_up_variable_cube(data,
                                                   spatial_grid="latlon")

        self.diagnostic_cube_hash = create_coordinate_hash(self.reference_cube)

        neighbours = np.array([[[0., 0., 0.]]])
        altitudes = np.array([0])
        latitudes = np.array([0])
        longitudes = np.array([0])
        wmo_ids = np.array([0])
        grid_attributes = ['x_index', 'y_index', 'vertical_displacement']
        neighbour_methods = ['nearest']
        self.neighbour_cube = build_spotdata_cube(
            neighbours, 'grid_neighbours', 1, altitudes, latitudes,
            longitudes, wmo_ids, grid_attributes=grid_attributes,
            neighbour_methods=neighbour_methods)
        self.neighbour_cube.attributes['model_grid_hash'] = (
            self.diagnostic_cube_hash)

    def test_matching_grids(self):
        """Test a case in which the grids match. There is no assert
        statement as this test is successful if no exception is raised."""
        cubes = [self.reference_cube, self.cube1, self.cube2]
        check_grid_match(cubes)

    def test_non_matching_grids(self):
        """Test a case in which a cube with an unmatching grid is included in
        the comparison, raising a ValueError."""
        cubes = [self.reference_cube, self.cube1, self.unmatched_cube]
        msg = ("Cubes do not share or originate from the same grid, so cannot "
               "be used together.")
        with self.assertRaisesRegex(ValueError, msg):
            check_grid_match(cubes)

    def test_using_model_grid_hash(self):
        """Test a case in which one of the cubes is a spotdata cube without a
        spatial grid. This cube includes a model_grid_hash to indicate on which
        grid the neighbours were found."""
        cubes = [self.reference_cube, self.neighbour_cube, self.cube2]
        check_grid_match(cubes)

    def test_using_model_grid_hash_reordered_cubes(self):
        """Test as above but using the neighbour_cube as the first in the list
        so that it acts as the reference for all the other cubes."""
        cubes = [self.neighbour_cube, self.reference_cube, self.cube2]
        check_grid_match(cubes)

    def test_multiple_model_grid_hash_cubes(self):
        """Test that a check works when all the cubes passed to the function
        have model_grid_hashes."""
        self.cube1.attributes["model_grid_hash"] = self.diagnostic_cube_hash
        cubes = [self.neighbour_cube, self.cube1]
        check_grid_match(cubes)

    def test_mismatched_model_grid_hash_cubes(self):
        """Test that a check works when all the cubes passed to the function
        have model_grid_hashes and these do not match."""
        self.cube1.attributes["model_grid_hash"] = "123"
        cubes = [self.neighbour_cube, self.cube1]
        msg = ("Cubes do not share or originate from the same grid, so cannot "
               "be used together.")
        with self.assertRaisesRegex(ValueError, msg):
            check_grid_match(cubes)


if __name__ == '__main__':
    unittest.main()

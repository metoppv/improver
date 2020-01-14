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
""" Unit tests for optical flow utilities """

import unittest

import iris
from iris.coords import DimCoord
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import check_input_coords

from ..forecasting.test_AdvectField import set_up_xy_velocity_cube


class Test_check_input_coords(IrisTest):
    """Tests for the check_input_coords function"""

    def setUp(self):
        """Set up dummy cube and plugin instance"""
        self.valid = set_up_xy_velocity_cube("advection_velocity_x")

    def test_missing_spatial_dimension(self):
        """Test rejects cube missing y axis"""
        invalid_1d = self.valid[0]
        invalid_1d.remove_coord("projection_y_coordinate")
        with self.assertRaises(InvalidCubeError):
            check_input_coords(invalid_1d)

    def test_additional_scalar_dimension(self):
        """Test accepts cube with single realization coordinate"""
        vel = self.valid.copy()
        vel.add_aux_coord(DimCoord(1, standard_name="realization"))
        check_input_coords(vel)

    def test_additional_nonscalar_dimension(self):
        """Test rejects cube with multiple realizations"""
        vel1 = self.valid.copy()
        vel1.add_aux_coord(DimCoord(1, standard_name="realization"))
        vel2 = self.valid.copy()
        vel2.add_aux_coord(DimCoord(2, standard_name="realization"))
        invalid_3d, = (iris.cube.CubeList([vel1, vel2])).merge()
        msg = "Cube has 3"
        with self.assertRaisesRegexp(InvalidCubeError, msg):
            check_input_coords(invalid_3d)

    def test_time(self):
        """Test rejects cube without time coord"""
        cube = self.valid.copy()
        cube.remove_coord("time")
        msg = "Input cube has no time coordinate"
        with self.assertRaisesRegexp(InvalidCubeError, msg):
            check_input_coords(cube, require_time=True)


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
import numpy as np
from iris.coords import DimCoord
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import (
    _perturb_background_flow,
    check_input_coords,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

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
        (invalid_3d,) = (iris.cube.CubeList([vel1, vel2])).merge()
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


class Test__perturb_background_flow(IrisTest):
    """Test for the _perturb_background_flow private utility"""

    def setUp(self):
        """Set up input cubes"""
        wind_u = set_up_variable_cube(
            np.full((10, 10), 2.1, dtype=np.float32),
            name="grid_eastward_wind",
            units="m s-1",
            spatial_grid="equalarea",
        )
        wind_v = wind_u.copy(data=np.full((10, 10), 2.4, dtype=np.float32))
        wind_v.rename("grid_northward_wind")
        self.background_flow = iris.cube.CubeList([wind_u, wind_v])

        flow_data = 0.1 * np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 2, 3, 2, 1],
                [1, 3, 4, 4, 2],
                [1, 2, 3, 3, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=np.float32,
        )

        padded_flow_data = np.zeros((10, 10), dtype=np.float32)
        padded_flow_data[3:8, 3:8] = flow_data

        flow_x = set_up_variable_cube(
            padded_flow_data,
            name="precipitation_advection_x_velocity",
            units="m s-1",
            spatial_grid="equalarea",
        )
        flow_y = flow_x.copy()
        flow_y.rename("precipitation_advection_y_velocity")
        self.perturbations = iris.cube.CubeList([flow_x, flow_y])

        data_with_nans = np.full((10, 10), np.nan, dtype=np.float32)
        data_with_nans[3:8, 3:8] = flow_data
        flow_xm = flow_x.copy(data=data_with_nans)
        flow_ym = flow_y.copy(data=data_with_nans)
        self.perturbations_with_nans = iris.cube.CubeList([flow_xm, flow_ym])

        self.expected_u = wind_u.data + flow_x.data
        self.expected_v = wind_v.data + flow_y.data

    def test_basic(self):
        """Test function returns cubes with expected names"""
        expected_names = [
            "precipitation_advection_x_velocity",
            "precipitation_advection_y_velocity",
        ]
        result = _perturb_background_flow(self.background_flow, self.perturbations)
        for i, cube in enumerate(result):
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertEqual(cube.name(), expected_names[i])

    def test_values(self):
        """Test function returns expected values"""
        result = _perturb_background_flow(self.background_flow, self.perturbations)
        self.assertArrayAlmostEqual(result[0].data, self.expected_u)
        self.assertArrayAlmostEqual(result[1].data, self.expected_v)

    def test_units(self):
        """Test values are returned in units of perturbations"""
        for cube in self.background_flow:
            cube.convert_units("knots")
        result = _perturb_background_flow(self.background_flow, self.perturbations)
        for cube in result:
            self.assertEqual(cube.units, "m s-1")
        self.assertArrayAlmostEqual(result[0].data, self.expected_u)
        self.assertArrayAlmostEqual(result[1].data, self.expected_v)

    def test_nans_values(self):
        """Test correct values are returned when an input contains nan values"""
        result = _perturb_background_flow(
            self.background_flow, self.perturbations_with_nans
        )
        self.assertArrayAlmostEqual(result[0].data, self.expected_u)
        self.assertArrayAlmostEqual(result[1].data, self.expected_v)


if __name__ == "__main__":
    unittest.main()

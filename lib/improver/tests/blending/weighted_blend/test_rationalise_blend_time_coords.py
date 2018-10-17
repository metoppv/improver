# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Module to test the rationalise_blend_time_coords function."""

import unittest
import numpy as np
from cf_units import date2num, Unit
from datetime import datetime

import numpy as np
import iris
from iris.tests import IrisTest
from iris.coords import AuxCoord, DimCoord

from improver.utilities.cube_manipulation import merge_cubes
from improver.blending.weighted_blend import rationalise_blend_time_coords


class Test_rationalise_blend_time_coords(IrisTest):
    """Tests for the rationalise_cycle_blend_time_coords function"""

    def setUp(self):
        """Set up a list of cubes from different models with some probability
        data in them."""
        data = np.full((3, 3), 0.6, dtype=np.float32)

        y_coord = DimCoord([40., 45., 50.], 'latitude', 'degrees')
        x_coord = DimCoord([-5., 0., 5.], 'longitude', 'degrees')

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        dt = datetime(2017, 1, 10, 3, 0)
        dt_num = date2num(dt, time_origin, calendar)
        time_coord = AuxCoord(dt_num, "time", units=tunit)

        base_cube = iris.cube.Cube(
            data, long_name="probability_of_air_temperature", units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
            aux_coords_and_dims=[(time_coord, None)])

        ukv_frt_coord = AuxCoord(
            dt_num-4, "forecast_reference_time", units=tunit)
        ukv_fp_coord = AuxCoord(4, "forecast_period", units="hours")

        self.ukv_cube = base_cube.copy()
        self.ukv_cube.add_aux_coord(ukv_frt_coord)
        self.ukv_cube.add_aux_coord(ukv_fp_coord)
        self.ukv_cube.attributes['mosg__model_configuration'] = 'uk_det'

        enuk_frt_coord = AuxCoord(
            dt_num-3, "forecast_reference_time", units=tunit)
        enuk_fp_coord = AuxCoord(3, "forecast_period", units="hours")

        self.enuk_cube = base_cube.copy()
        self.enuk_cube.add_aux_coord(enuk_frt_coord)
        self.enuk_cube.add_aux_coord(enuk_fp_coord)
        self.enuk_cube.attributes['mosg__model_configuration'] = 'uk_ens'

        self.cubelist = iris.cube.CubeList([self.ukv_cube, self.enuk_cube])
        self.cube = merge_cubes(self.cubelist)

    def test_null(self):
        """Test function does nothing if not given a relevant coord"""
        reference_cubelist = self.cubelist.copy()
        rationalise_blend_time_coords(self.cubelist, "realization")
        self.assertEqual(self.cubelist, reference_cubelist)

    def test_create_fp(self):
        """Test function creates forecast_period coord if blending over
        forecast_reference_time"""
        reference_coord = self.cube.coord("forecast_period")
        reference_coord.convert_units("seconds")
        for cube in self.cubelist:
            cube.remove_coord("forecast_period")
        rationalise_blend_time_coords(self.cubelist, "forecast_reference_time")
        merged_cube = merge_cubes(self.cubelist)
        self.assertEqual(merged_cube.coord("forecast_period"), reference_coord)

    def test_unify_frt(self):
        """Test function equalises forecast reference times if weighting a
        model blend by forecast_period"""
        expected_frt, = self.enuk_cube.coord("forecast_reference_time").points
        expected_fp = 3.
        rationalise_blend_time_coords(
            self.cubelist, "model", weighting_coord="forecast_period")
        merged_cube = merge_cubes(self.cubelist)
        for coord in ["forecast_reference_time", "forecast_period"]:
            self.assertEqual(len(merged_cube.coord(coord).points), 1)
        self.assertAlmostEqual(
            merged_cube.coord("forecast_reference_time").points[0],
            expected_frt)
        self.assertAlmostEqual(
            merged_cube.coord("forecast_period").points[0], expected_fp)


if __name__ == '__main__':
    unittest.main()

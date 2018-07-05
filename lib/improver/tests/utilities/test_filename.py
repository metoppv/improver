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
"""Unit tests for file name generation."""

import unittest
import numpy as np
from cf_units import Unit

import iris
from iris.coords import DimCoord, AuxCoord

from improver.utilities.filename import generate_file_name


class Test_generate_file_name(unittest.TestCase):
    """Test the generate_file_name function"""

    def setUp(self):
        """Set up dummy cube"""
        x_coord = DimCoord(np.arange(3), "projection_x_coordinate", units="km")
        y_coord = DimCoord(np.arange(3), "projection_y_coordinate", units="km")
        data = np.zeros((3, 3))

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        t_coord = AuxCoord(np.linspace(402192.5, 402292.5, 1),
                           "time", units=tunit)

        fp_coord = AuxCoord(900, "forecast_period", units="s")

        self.cube = iris.cube.Cube(data, "air_temperature", units='degreesC',
                                   dim_coords_and_dims=[(y_coord, 0),
                                                        (x_coord, 1)])
        self.cube.add_aux_coord(t_coord)
        self.cube.add_aux_coord(fp_coord)

    def test_basic(self):
        """Test basic file name generation"""
        name = generate_file_name(self.cube)
        self.assertIsInstance(name, str)
        self.assertEqual(name, "20151119T0030Z-PT0000H15M-air_temperature.nc")


if __name__ == '__main__':
    unittest.main()


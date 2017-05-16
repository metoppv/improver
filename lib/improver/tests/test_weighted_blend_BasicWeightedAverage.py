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
"""Unit tests for the weighted_blend.BasicWeightedAverage plugin."""


import unittest

from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.weighted_blend import BasicWeightedAverage


class TestBasicWeightedAverage(IrisTest):

    """Test the Basic Weighted Average plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        data = np.zeros((2, 5, 5))
        data[0][:][:] = 0.0
        data[1][:][:] = 1.0
        cube = Cube(data, standard_name="precipitation_amount",
                    units="kg m^-2 s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 5), 'latitude',
                                    units='degrees'), 1)
        cube.add_dim_coord(DimCoord(np.linspace(120, 180, 5), 'longitude',
                                    units='degrees'), 2)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(AuxCoord([402192.5,402193.5],
                                    "time", units=tunit), 0)
        self.cube = cube

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        coord = "time"
        plugin = BasicWeightedAverage(coord)
        result = plugin.process(self.cube)
        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()

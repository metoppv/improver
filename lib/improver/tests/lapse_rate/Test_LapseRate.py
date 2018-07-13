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
"""Unit tests for the LapseRate plugin."""

import numpy as np
import unittest

import cf_units
import iris
from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import (DimCoord,
                         AuxCoord)

from improver.grids import STANDARD_GRID_CCRS

from improver.lapse_rate import LapseRate


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(LapseRate())
        msg = ('<LapseRate>')
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the LapseRate processing works"""

    def setUp(self):
        """Create a cube containing a regular grid."""

        grid_size = 16
        data = np.zeros((1, grid_size, grid_size))

        realization = DimCoord([0], 'realization', units=1)
        time = DimCoord([402192.5], standard_name='time',
                        units=cf_units.Unit('hours since 1970-01-01 00:00:00',
                                            calendar='gregorian'))
        projection_y = DimCoord(np.arange(0, grid_size, 1),
                                'projection_y_coordinate',
                                units='m',
                                coord_system=STANDARD_GRID_CCRS)
        projection_x = DimCoord(np.arange(0, grid_size, 1),
                                'projection_x_coordinate',
                                units='m',
                                coord_system=STANDARD_GRID_CCRS)

        # Set up temperature cube.
        self.temperature = Cube(data, standard_name="air_temperature",
                                dim_coords_and_dims=[(realization, 0),
                                                     (projection_y, 1),
                                                     (projection_x, 2)],
                                units="K")

        height = AuxCoord([1.5], standard_name='height', units='m')
        self.temperature.add_aux_coord(height)
        self.temperature.attributes['institution'] = 'Met Office'

        # Creates bands of temperature running East/West.
        self.temperature.data[:, :, :] = 0
        self.temperature.data[:, 0:4, 4:6] = -5
        self.temperature.data[:, 0:4, 6:8] = -10
        self.temperature.data[:, 0:4, 8:10] = 10
        self.temperature.data[:, 0:4, 10:12] = 5
        self.temperature.data[:, 4:8, 4:6] = -10
        self.temperature.data[:, 4:8, 6:8] = -20
        self.temperature.data[:, 4:8, 8:10] = 20
        self.temperature.data[:, 4:8, 10:12] = 10

        # Copies temperature cube to create orography cube.
        self.orography = self.temperature.copy()[0]
        self.orography.remove_coord("realization")
        self.orography.rename("surface_altitude")
        self.orography.units = cf_units.Unit('m')

        # Creates a valley in front of a ridge running North/South.
        self.orography.data[:] = 0
        self.orography.data[:, 4:6] = -10
        self.orography.data[:, 6:8] = -20
        self.orography.data[:, 8:10] = 20
        self.orography.data[:, 10:12] = 10
        self.orography.data[12:] = 0  # Sets shore line to sea level

        # Copies orography cube to create land/sea mask cube.
        self.land_sea_mask = self.orography.copy()
        self.land_sea_mask.rename("land_binary_mask")
        self.land_sea_mask.units = cf_units.Unit('1')

        # Creates 'shoreline' running West/East at bottom of domain.
        self.land_sea_mask.data[:] = 1
        self.land_sea_mask.data[12:] = 0

    def test_basic(self):
        """Test that the plugin returns expected data types. """

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertIsInstance(result_cube, Cube)


if __name__ == '__main__':
    unittest.main()

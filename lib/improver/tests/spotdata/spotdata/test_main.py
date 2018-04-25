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
"""Unit tests for the spotdata.main"""

from collections import OrderedDict
import numpy as np
import unittest

import cf_units
import iris
from iris.coords import DimCoord
from iris.tests import IrisTest
from iris.cube import Cube

from improver.spotdata.main import run_spotdata as Function
from improver.spotdata.main import process_diagnostic


class Test_main(IrisTest):

    """Test the SpotData framework."""

    def setUp(self):
        """Create components required for testing the spotdata framework.
        Here we are testing various error captures, so we need enough data
        to get to the various error checks within run_spotdata and
        process_diagnostic.

        """
        # Create air temperature data cube.
        data = np.arange(0, 800, 1)
        data.resize(2, 20, 20)
        latitudes = np.linspace(-90, 90, 20)
        longitudes = np.linspace(-180, 180, 20)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees')
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees')

        # Valid at times 2017-02-17 06:00:00, 07:00:00
        time = DimCoord(
            [1487311200, 1487314800], standard_name='time',
            units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                calendar='gregorian'))
        forecast_ref_time = time[0].copy()
        forecast_ref_time.rename('forecast_reference_time')

        cube = Cube(data,
                    long_name="air_temperature",
                    dim_coords_and_dims=[(time, 0),
                                         (latitude, 1),
                                         (longitude, 2)],
                    units="K")
        cube.add_aux_coord(forecast_ref_time)

        # Create the orography ancillary cube.
        orography = Cube(np.ones((20, 20)),
                         long_name="surface_altitude",
                         dim_coords_and_dims=[(latitude, 0),
                                              (longitude, 1)],
                         units="m")

        self.ancillary_data = {}
        self.ancillary_data.update({'orography': orography})

        # A sample spotdata diagnostic recipe specifying how to select the
        # neighbouring grid point and then extract the data.
        diagnostic_recipe = {
            "temperature": {
                "diagnostic_name": "air_temperature",
                "extrema": False,
                "filepath": "temperature_at_screen_level",
                "interpolation_method": "use_nearest",
                "neighbour_finding": {
                    "land_constraint": False,
                    "method": "fast_nearest_neighbour",
                    "vertical_bias": None
                    }
                }
            }

        diagnostic_recipe["temperature"]["data"] = iris.cube.CubeList([cube])
        diagnostic_recipe["temperature"]["additional_data"] = None
        self.diagnostic_recipe = diagnostic_recipe

        self.sites = OrderedDict()
        self.sites['100'] = {'latitude': 50,
                             'longitude': 0,
                             'altitude': 10,
                             'utc_offset': 0,
                             'wmo_site': 0}

        self.config_constants = {}

        self.args = (self.diagnostic_recipe, self.ancillary_data,
                     self.sites, self.config_constants)

        self.kwargs = {
            'use_multiprocessing': False
            }


class Test_run_spotdata(Test_main):
    """Test the run_framework interface with various options."""

    def test_nominal_run_with_kwargs(self):
        """Test a typical run of the routine completes successfully."""
        result = Function(*self.args, **self.kwargs)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0][0], Cube)
        self.assertEqual(result[0][0].name(), 'air_temperature')
        self.assertEqual(result[1][0], None)

    def test_nominal_run_with_kwargs_for_multiprocessing(self):
        """Test a typical run of the routine completes successfully
        when multiprocessing is enabled."""
        kwargs = {
            'use_multiprocessing': False
            }
        result = Function(*self.args, **kwargs)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0][0], Cube)
        self.assertEqual(result[0][0].name(), 'air_temperature')
        self.assertEqual(result[1][0], None)

    def test_nominal_run_without_kwargs(self):
        """Test a typical run of the routine completes as intended.
        If there are no keyword arguments then the current time will be used
        and this will not match any of the times within the input cubes."""
        result = Function(*self.args)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0][0], Cube)
        self.assertEqual(result[0][0].name(), 'air_temperature')
        self.assertEqual(result[1][0], None)


class Test_process_diagnostic(Test_main):
    """Test the process_diagnostic function."""

    def test_nominal_run(self):
        """Test a typical run of process_diagnostics."""
        neighbours = {
            'fast_nearest_neighbour-None-False':
                np.array([(15, 10, 9.0, False)],
                         dtype=[('i', '<i8'), ('j', '<i8'),
                                ('dz', '<f8'), ('edgepoint', '?')])}
        result = process_diagnostic(
            self.diagnostic_recipe, neighbours, self.sites,
            self.ancillary_data, "temperature")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Cube)
        self.assertEqual(result[1], None)


if __name__ == '__main__':
    unittest.main()

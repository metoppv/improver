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
"""
Unit tests for the
`ensemble_copula_coupling.RebadgePercentilesAsMembers` class.

"""
import unittest

from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsMembers as Plugin)
from improver.tests.helper_functions_ensemble_calibration import (
    set_up_temperature_cube, _add_forecast_reference_time_and_forecast_period)


class Test_process(IrisTest):

    """
    Test the process method of the RebadgePercentilesAsMembers plugin.
    """

    def setUp(self):
        cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        percentile_points = np.arange(len(cube.coord("realization").points))
        cube.coord("realization").points = percentile_points
        cube.coord("realization").rename("percentile")
        self.current_temperature_cube = cube

    def test_basic(self):
        """"""
        cube = self.current_temperature_cube
        plugin = Plugin()
        result = plugin.process(cube)
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.coord("realization"), DimCoord)

    def test_number_of_members(self):
        """"""
        cube = self.current_temperature_cube
        plen = len(cube.coord("percentile").points)
        plugin = Plugin()
        result = plugin.process(cube)
        self.assertEqual(len(result.coord("realization").points), plen)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, np.array([0, 1, 2]))

    def test_no_percentile_coord(self):
        """"""
        cube = self.current_temperature_cube
        cube.coord("percentile").rename("realization")
        plugin = Plugin()
        msg = "The percentile coordinate could not be found"
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            plugin.process(cube)


if __name__ == '__main__':
    unittest.main()

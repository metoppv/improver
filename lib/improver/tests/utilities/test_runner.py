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
"""Unit tests for running plugins."""

import unittest
import numpy as np
from iris.cube import Cube, CubeList
from iris.coords import DimCoord
from improver.utilities.runner import apply_plugin


def mock_plugin():
    """ Set up a mock plugin. """
    class Plugin(object):
        """Mock plugin class"""
        @staticmethod
        def process(cube):
            """Dummy process method which does nothing"""
            return cube

    return Plugin()


def sample_cubelist():
    """Set up dummy cube for test"""
    data = np.array([[1., 5., 10.],
                     [3., 4., 7.],
                     [0., 2., 1.]])
    cube = Cube(data, "precipitation_amount", units="kg m^-2 s^-1")
    cube.add_dim_coord(DimCoord(np.linspace(0.0, 4.0, 3),
                                'projection_y_coordinate',
                                units='m'), 0)
    cube.add_dim_coord(DimCoord(np.linspace(0.0, 4.0, 3),
                                'projection_x_coordinate',
                                units='m'), 1)
    return CubeList([cube, cube])


class Test_runner_runs(unittest.TestCase):
    """ Test function to execute the plugin. """

    def setUp(self):
        self.cubelist = sample_cubelist()
        self.plugin = mock_plugin()

    def test_runner(self):
        """ Test handing the runner a mock Plugin.method and a CubeList. """
        result = apply_plugin(self.plugin.process, self.cubelist)
        self.assertTrue(result == self.cubelist)

    def test_runner_omitting_method(self):
        """ Test handing the runner a mock Plugin instance and a CubeList. """
        result = apply_plugin(self.plugin, self.cubelist)
        self.assertTrue(result == self.cubelist)


if __name__ == '__main__':
    unittest.main()

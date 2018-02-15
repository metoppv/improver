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
"""Unit tests for saving functionality."""

import os
import unittest
import numpy as np
from subprocess import call
from tempfile import mkdtemp

import iris
from iris.tests import IrisTest
from improver.utilities.save import save_netcdf

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


class Test_save_netcdf(IrisTest):

    """Test function to save iris cubes as netcdf.

    NOTE this is a dummy class as "save_netcdf" is currently just wrapping
    iris.fileformats.netcdf.save.  More tests will be added when local_keys
    functionality is incorporated.
    """

    def setUp(self):
        """ Set up cube to write, read and check """
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_temperature_cube()

    def tearDown(self):
        """ Remove temporary directories created for testing. """
        call(['rm', '-f', self.filepath])
        call(['rmdir', self.directory])

    def test_basic(self):
        """ Test saves file in required location """
        self.assertFalse(os.path.exists(self.filepath))
        save_netcdf(self.cube, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_saved_cube(self):
        """ Test valid cube can be read from saved file """
        save_netcdf(self.cube, self.filepath)
        cube = iris.load_cube(self.filepath)
        self.assertTrue(isinstance(cube, iris.cube.Cube))
        self.assertTrue(np.array_equal(cube.data, self.cube.data))


if __name__ == '__main__':
    unittest.main()

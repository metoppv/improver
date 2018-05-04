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
""" Unit tests for the optical_flow.OpticalFlow plugin """

import datetime
import unittest
import numpy as np

import iris
from iris.tests import IrisTest

from improver.optical_flow.optical_flow import OpticalFlow
#import sys
#sys.path.append('/home/h04/csand/IMPROVER/IMPRO623/from_Martina_copy')
#from Decomposition_2017_modified import OFC as OpticalFlow



class Test__init__(IrisTest):
    """Test class initialisation"""

    """
    DUMMY CLASS - provides baseline for Martina's code functionality
    that I can modify as I refactor.  (TODO get this working...)
    """

    def test_basic(self):
        """Test advection velocity in the x-direction"""

        first_input = np.broadcast_to(
            np.array([2., 3., 4., 5., 6., 5., 4., 3., 2., 1.]), (10, 10))
        second_input = np.broadcast_to(
            np.array([1., 2., 3., 4., 5., 6., 5., 4., 3., 2.]), (10, 10))

        expected_ucomp = np.ones((10, 10))
        expected_vcomp = np.zeros((10, 10))
    
        plugin = OpticalFlow(data1=first_input, data2=second_input,
                             kernel=3, myboxsize=3, iterations=10)

        """
        NOTE by experimentation:
            - Minimum stable value for "myboxsize" is 3 - will need to use large arrays for unit testing
            - TODO read the documentation!

        """

        print plugin.ucomp[0:2, :]
        print plugin.vcomp[0:2, :]


if __name__ == '__main__':
    unittest.main()

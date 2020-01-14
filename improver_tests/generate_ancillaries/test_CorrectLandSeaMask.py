# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Unit tests for the generate_ancillary.CorrectLandSeaMask plugin."""


import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_ancillary import \
    CorrectLandSeaMask as CorrectLand


class Test_process(IrisTest):
    """Test the land-sea mask correction plugin."""
    def setUp(self):
        """setting up paths to test ancillary files"""
        landmask_data = np.array([[0.2, 0., 0.],
                                  [0.7, 0.5, 0.05],
                                  [1, 0.95, 0.7]])
        self.landmask = Cube(landmask_data, long_name='test land')
        self.expected_mask = np.array([[False, False, False],
                                       [True, True, False],
                                       [True, True, True]])

    def test_landmaskcorrection(self):
        """Test landmask correction. Note that the name land_binary_mask is
        enforced to reflect the change that has been made."""
        result = CorrectLand().process(self.landmask)
        self.assertEqual(result.name(), 'land_binary_mask')
        self.assertArrayEqual(result.data, self.expected_mask)


if __name__ == "__main__":
    unittest.main()

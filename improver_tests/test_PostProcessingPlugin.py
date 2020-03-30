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
"""Unit tests for the improver.PostProcessingPlugin abstract base class"""

import unittest
import numpy as np

from improver import PostProcessingPlugin
from .set_up_test_cubes import set_up_variable_cube


class DummyPlugin(PostProcessingPlugin):
    """Dummy class inheriting from the abstract base class"""

    def process(self, cube):
        """Local process method has no effect"""
        return cube


class Test_process(unittest.TestCase):
    """Tests for functionality implemented when "process" is called"""

    def setUp(self):
        """Set up a plugin and cube"""
        self.plugin = DummyPlugin()
        self.cube = set_up_variable_cube(
            np.ones((3, 3, 3), dtype=np.float32),
            standard_grid_metadata='uk_det',
            attributes={'title': 'UKV Model Forecast'})

    def test_title_updated(self):
        """Test title is updated as expected"""
        expected_title = "Post-Processed UKV Model Forecast"
        result = self.plugin(self.cube)
        self.assertEqual(result.attributes["title"], expected_title)

    def test_title_preserved(self):
        """Test title is preserved if it contains 'Post-Processed'"""
        expected_title = "IMPROVER Post-Processed Multi-Model Blend"
        self.cube.attributes["title"] = expected_title
        result = self.plugin(self.cube)
        self.assertEqual(result.attributes["title"], expected_title)


if __name__ == '__main__':
    unittest.main()

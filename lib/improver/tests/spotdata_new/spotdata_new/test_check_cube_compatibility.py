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
"""Unit tests for check_cube_compatibility function."""

import unittest

import iris
from iris.tests import IrisTest

from improver.spotdata_new.spot_extraction import check_cube_compatibility


class Test_check_cube_compatibility(IrisTest):

    """Test class for the check_cube_compatibility function."""

    def setUp(self):
        """Set up cubes for use in testing."""

        reference_cube = iris.cube.Cube(
            [0], long_name="reference_cube", units=1,
            attributes={
                'mosg__grid_domain': 'region',
                'mosg__grid_type': 'standard',
                'mosg__grid_version': '1.2.0',
                'mosg__model_configuration': 'region_det'})

        self.reference_cube = reference_cube

    def test_none_identifier(self):
        """Test case in which the identifier is set to None. This should
        always result in success as no test is performed. We change one
        attribute which should otherwise result in a False result."""
        matching_cube = self.reference_cube.copy()
        matching_cube.attributes['mosg__model_configuration'] = 'region_ens'
        identifier = None
        result = check_cube_compatibility(identifier, self.reference_cube,
                                          matching_cube)
        self.assertTrue(result)

    def test_no_identifier_success(self):
        """Test case in which no identifier is provided, which matches all
        attributes."""
        matching_cube = self.reference_cube.copy()
        identifier = ''
        result = check_cube_compatibility(identifier, self.reference_cube,
                                          matching_cube)
        self.assertTrue(result)

    def test_no_identifier_failure(self):
        """Test case in which no identifier is provided, which matches all
        attributes. Here we add an extra attribute to one cube so the test
        should fail."""
        matching_cube = self.reference_cube.copy()
        matching_cube.attributes['extra_attribute'] = 1
        identifier = ''
        result = check_cube_compatibility(identifier, self.reference_cube,
                                          matching_cube)
        self.assertFalse(result)

    def test_matching_cube(self):
        """Test case in which cubes match exactly."""

        matching_cube = self.reference_cube.copy()
        identifier = 'mosg'
        result = check_cube_compatibility(identifier, self.reference_cube,
                                          matching_cube)
        self.assertTrue(result)

    def test_matching_grid_disallowed(self):
        """Test case in which cubes share the same grid but not the same
        model. In this case we are looking for an exact match on all mosg
        attributes."""

        matching_grid = self.reference_cube.copy()
        matching_grid.attributes['mosg__model_configuration'] = 'region_ens'
        identifier = 'mosg'
        result = check_cube_compatibility(identifier, self.reference_cube,
                                          matching_grid)
        self.assertFalse(result)

    def test_matching_grid_allowed(self):
        """Test case in which cubes share the same grid but not the same
        model. In this case we are looking for a match only on mosg_grid
        attributes, so this is allowed."""

        matching_grid = self.reference_cube.copy()
        matching_grid.attributes['mosg__model_configuration'] = 'region_ens'
        identifier = 'mosg__grid'
        result = check_cube_compatibility(identifier, self.reference_cube,
                                          matching_grid)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()

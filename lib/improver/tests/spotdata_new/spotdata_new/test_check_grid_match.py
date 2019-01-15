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
"""Unit tests for check_grid_match function."""

import unittest
import numpy as np

from iris.tests import IrisTest

from improver.spotdata_new.spot_extraction import check_grid_match
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_check_grid_match(IrisTest):

    """Test the check_grid_match function."""

    def setUp(self):
        """Set up cubes for use in testing."""

        attributes = {
            'mosg__grid_domain': 'uk',
            'mosg__grid_type': 'standard',
            'mosg__grid_version': '1.2.0',
            'mosg__model_configuration': 'uk_det'}

        data = np.ones(9).reshape(3, 3).astype(np.float32)
        self.reference_cube = set_up_variable_cube(data, attributes=attributes,
                                                   spatial_grid="equalarea")
        self.cube1 = self.reference_cube.copy()
        self.cube2 = self.reference_cube.copy()

    def test_matching_metadata(self):
        """Test a case in which the grid metadata matches. There is no assert
        statement as this test is successful if no exception is raised."""
        cubes = [self.reference_cube, self.cube1, self.cube2]
        check_grid_match('mosg', cubes)

    def test_non_matching_metadata(self):
        """Test a case in which the grid metadata does not match. This will
        raise an ValueError."""
        self.reference_cube.attributes["mosg__grid_domain"] = "eire"
        cubes = [self.reference_cube, self.cube1, self.cube2]
        msg = "Cubes do not share the metadata identified"
        with self.assertRaisesRegex(ValueError, msg):
            check_grid_match('mosg', cubes)

    def test_ignore_non_matching_metadata(self):
        """Test a case in which the grid metadata does not match but this is
        forceably ignored by the user by setting self.grid_metadata_identifier
        to None."""
        self.reference_cube.attributes["mosg__grid_domain"] = "eire"
        cubes = [self.reference_cube, self.cube1, self.cube2]
        check_grid_match(None, cubes)

    def test_no_identifier_success(self):
        """Test case in which an empty string is provided as the identifier,
        which matches all keys, assuming no numeric keys."""
        cubes = [self.reference_cube, self.cube1, self.cube2]
        check_grid_match('', cubes)

    def test_no_identifier_failure(self):
        """Test case in which an empty string is provided as the identifier,
        which matches all keys, assuming no numeric keys. In this case we
        expect a failure as we add an extra attribute."""
        self.cube1.attributes['extra_attribute'] = 'extra'
        cubes = [self.reference_cube, self.cube1, self.cube2]
        msg = "Cubes do not share the metadata identified"
        with self.assertRaisesRegex(ValueError, msg):
            check_grid_match('', cubes)


if __name__ == '__main__':
    unittest.main()

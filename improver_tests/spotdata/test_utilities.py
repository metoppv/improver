# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for spot data related utilities"""


from improver.spotdata.utilities import check_for_unique_id

from .test_SpotExtraction import Test_SpotExtraction


class Test_check_for_unique_id(Test_SpotExtraction):

    """Test identification of unique site ID coordinates from coordinate
    attributes."""

    def test_unique_is_present(self):
        """Test that the IDs and coordinate name are returned if a unique site
        ID coordinate is present on the neighbour cube."""
        result = check_for_unique_id(self.neighbour_cube)
        self.assertArrayEqual(result[0], self.unique_site_id)
        self.assertEqual(result[1], self.unique_site_id_key)

    def test_unique_is_not_present(self):
        """Test that Nones are returned if no unique site ID coordinate is
        present on the neighbour cube."""
        self.neighbour_cube.remove_coord("met_office_site_id")
        result = check_for_unique_id(self.neighbour_cube)
        self.assertIsNone(result)

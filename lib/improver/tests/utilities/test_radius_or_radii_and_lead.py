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
"""Unit tests for utilities.cli_utilities."""

import unittest

from improver.utilities.cli_utilities import radius_or_radii_and_lead


class TestRadiusOrRadiiAndLead(unittest.TestCase):
    """Tests radius_or_radii_and_lead to split the data correctly."""

    def test_radius(self):
        """Tests a correct radius and a radii/lead of None.
        radius_or_radii is the first input, lead is None."""
        radius, lead = radius_or_radii_and_lead(3.3, None)

        self.assertEqual(radius, 3.3)
        self.assertIsNone(lead)

    def test_both_None(self):
        """ Tests if both are None then both are returned None."""
        radius, lead = radius_or_radii_and_lead(None, None)

        self.assertIsNone(radius)
        self.assertIsNone(lead)

    def test_radii_and_lead(self):
        """Tests if radius is None and radii/lead is a list of csv.
        radius_or_radii is the [0] of the second index separated on the commas
        lead is the [1] of the second index separated on the commas."""
        radii, lead = radius_or_radii_and_lead(
            None, ["0,36,72,144", "18000,54000,90000,162000"])

        self.assertEqual(radii, ['0', '36', '72', '144'])
        self.assertEqual(lead, ['18000', '54000', '90000', '162000'])

    def test_both_used(self):
        """Tests if both arguments are used.
        The output is the same as if the second argument is None."""
        radius, lead = radius_or_radii_and_lead(
            3.3, ["0,36,72,144", "18000,54000,90000,162000"])

        self.assertEqual(radius, 3.3)
        self.assertIsNone(lead)


if __name__ == '__main__':
    unittest.main()

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
"""Unit tests for nbhood.__init__"""

import unittest

from improver.nbhood import radius_by_lead_time


class Test_radius_by_lead_time(unittest.TestCase):

    """Test the radius_by_lead_time method."""

    def test_single_radius(self):
        """Test that when a single radius is provided with no lead times the
        returned objects are a float equal to the input radius and a NoneType
        representing the lead times."""

        radii = ['10000']
        lead_times = None
        radii_out, lead_times_out = radius_by_lead_time(radii, lead_times)

        self.assertEqual(radii_out, float(radii[0]))
        self.assertEqual(lead_times_out, None)
        self.assertIsInstance(radii_out, float)

    def test_multiple_radii(self):
        """Test that when multiple radii are provided with lead times the
        returned objects are two lists, one of radii as floats and one of lead
        times as ints."""

        radii = ['10000', '20000']
        lead_times = ['0', '10']
        radii_out, lead_times_out = radius_by_lead_time(radii, lead_times)

        self.assertEqual(radii_out, list(map(float, radii)))
        self.assertEqual(lead_times_out, list(map(int, lead_times)))
        self.assertIsInstance(radii_out[0], float)
        self.assertIsInstance(lead_times_out[0], int)

    def test_unmatched_input_length(self):
        """Test that when multiple radii are provided with an unmatched number
        of lead times an exception is raised."""

        radii = ['10000', '20000']
        lead_times = ['0', '10', '20']

        msg = "If leadtimes are supplied, it must be a list of equal length"
        with self.assertRaisesRegex(ValueError, msg):
            radius_by_lead_time(radii, lead_times)

    def test_multiple_radii_no_lead_times(self):
        """Test that when multiple radii are provided with no lead times an
        exception is raised."""

        radii = ['10000', '20000']
        lead_times = None

        msg = "Multiple radii have been supplied but no associated lead times."
        with self.assertRaisesRegex(ValueError, msg):
            radius_by_lead_time(radii, lead_times)


if __name__ == '__main__':
    unittest.main()

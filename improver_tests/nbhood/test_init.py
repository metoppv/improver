# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for nbhood.__init__"""

import unittest

from improver.nbhood import radius_by_lead_time


class Test_radius_by_lead_time(unittest.TestCase):

    """Test the radius_by_lead_time method."""

    def test_single_radius(self):
        """Test that when a single radius is provided with no lead times the
        returned objects are a float equal to the input radius and a NoneType
        representing the lead times."""

        radii = ["10000"]
        lead_times = None
        radii_out, lead_times_out = radius_by_lead_time(radii, lead_times)

        self.assertEqual(radii_out, float(radii[0]))
        self.assertEqual(lead_times_out, None)
        self.assertIsInstance(radii_out, float)

    def test_multiple_radii(self):
        """Test that when multiple radii are provided with lead times the
        returned objects are two lists, one of radii as floats and one of lead
        times as ints."""

        radii = ["10000", "20000"]
        lead_times = ["0", "10"]
        radii_out, lead_times_out = radius_by_lead_time(radii, lead_times)

        self.assertEqual(radii_out, list(map(float, radii)))
        self.assertEqual(lead_times_out, list(map(int, lead_times)))
        self.assertIsInstance(radii_out[0], float)
        self.assertIsInstance(lead_times_out[0], int)

    def test_unmatched_input_length(self):
        """Test that when multiple radii are provided with an unmatched number
        of lead times an exception is raised."""

        radii = ["10000", "20000"]
        lead_times = ["0", "10", "20"]

        msg = "If leadtimes are supplied, it must be a list of equal length"
        with self.assertRaisesRegex(ValueError, msg):
            radius_by_lead_time(radii, lead_times)

    def test_multiple_radii_no_lead_times(self):
        """Test that when multiple radii are provided with no lead times an
        exception is raised."""

        radii = ["10000", "20000"]
        lead_times = None

        msg = "Multiple radii have been supplied but no associated lead times."
        with self.assertRaisesRegex(ValueError, msg):
            radius_by_lead_time(radii, lead_times)


if __name__ == "__main__":
    unittest.main()

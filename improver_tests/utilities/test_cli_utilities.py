# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for utilities.cli_utilities."""

import unittest
from unittest.mock import mock_open, patch

from improver.utilities.cli_utilities import load_json_or_none


class Test_load_json_or_none(unittest.TestCase):
    """Tests load_json_or_none to call loading json or return None."""

    @patch("builtins.open", new_callable=mock_open, read_data='{"k": "v"}')
    def test_loading_file(self, m):
        """Tests if called with a filepath, loads a dict."""
        file_path = "filename"
        dict_read = load_json_or_none(file_path)
        self.assertEqual(dict_read, {"k": "v"})
        m.assert_called_with("filename", "r")

    @patch("builtins.open", new_callable=mock_open, read_data='{"k": "v"}')
    def test_none(self, m):
        """Tests if called with None returns None."""
        file_path = None
        dict_read = load_json_or_none(file_path)
        self.assertIsNone(dict_read)
        m.assert_not_called()


if __name__ == "__main__":
    unittest.main()

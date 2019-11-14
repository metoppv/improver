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
"""Unit tests for cli.__init__"""

import unittest
from unittest.mock import patch

import improver
from improver.cli import (
    docutilize, unbracket,
    maybe_coerce_with, inputcube, inputjson, with_output,
    with_intermediate_output)


def dummy_function(first, second=0, third=2):
    """A dummy function for testing clize usage.

    Args:
        first (str):
            The first argument.
        second (int):
            The second argument.
        third (iris.cube.Cube):
            The third argument

    Returns:
        (iris.cube.Cube)

    """
    first = int(first)
    return first + first


@with_output
def wrapped_with_output(first):
    """dummy function for testing with_output wrapper"""
    return dummy_function(first)


@with_intermediate_output
def wrapped_with_intermediate_output(first):
    """dummy function for testing with_intermediate_output wrapper"""
    return dummy_function(first), True


class Test_docutilize(unittest.TestCase):

    """Test the docutilize function."""

    def setUp(self):
        self.expected = """A dummy function for testing clize usage.

:param first: The first argument.
:type first: str
:param second: The second argument.
:type second: int
:param third: The third argument
:type third: iris.cube.Cube

:returns: (iris.cube.Cube)
"""

    def test_obj(self):
        """Tests the docutilize function on an object"""
        doc = docutilize(dummy_function)

        self.assertFalse(isinstance(doc, str))
        self.assertEqual(self.expected.strip(), doc.__doc__.strip())

    def test_str(self):
        """Tests the docutilize function on a string"""
        doc = docutilize(dummy_function.__doc__)
        self.assertEqual(self.expected.strip(), doc.strip())


class Test_maybe_coerce_with(unittest.TestCase):
    """Tests the maybe_coerce_with function."""

    def test_none_string(self):
        """Tests that if a none string is passed in, it will return it."""
        result = maybe_coerce_with(dummy_function, 2)
        expected = 2
        self.assertEqual(result, expected)

    def test_string(self):
        """Tests that if a string is passed in, it will called the function."""
        result = maybe_coerce_with(dummy_function, "2")
        # Dummy function will be 2 + 2 therefore 4.
        self.assertEqual(result, 4)


class Test_inputcube(unittest.TestCase):
    """Tests the input cube function"""

    @patch('improver.cli.maybe_coerce_with', return_value='return')
    def test_basic(self, m):
        """Tests that input cube calls load_cube with the string"""
        result = inputcube("foo")
        m.assert_called_with(improver.utilities.load.load_cube, "foo")
        self.assertEqual(result, 'return')


class Test_inputjson(unittest.TestCase):
    """Tests the input cube function"""

    @patch('improver.cli.maybe_coerce_with', return_value={"mocked": 1})
    def test_basic(self, m):
        """Tests that input json calls load_json_or_none with the string"""
        result = inputjson("foo")
        m.assert_called_with(
            improver.utilities.cli_utilities.load_json_or_none, "foo")
        self.assertEqual(result, {"mocked": 1})


class Test_with_output(unittest.TestCase):
    """Tests the with_output wrapper"""

    @patch('improver.utilities.save.save_netcdf')
    def test_without_output(self, m):
        """Tests that the result of the wrapped function is returned"""
        result = wrapped_with_output(2)
        m.assert_not_called()
        self.assertEqual(result, 4)

    @patch('improver.utilities.save.save_netcdf')
    def test_with_output(self, m):
        """Tests that save_netcdf it called with object and string"""
        # pylint disable is needed as it can't see the wrappers output kwarg.
        # pylint: disable=E1123
        result = wrapped_with_output(2, output="foo")
        m.assert_called_with(4, 'foo')
        self.assertEqual(result, None)


class Test_with_intermediate_output(unittest.TestCase):
    """Tests the intermediate output wrapper"""

    @patch('improver.utilities.save.save_netcdf')
    def test_without_output(self, m):
        """Tests that the wrapped function is called and result is returned"""
        result = wrapped_with_intermediate_output(2)
        m.assert_not_called()
        self.assertEqual(result, 4)

    @patch('improver.utilities.save.save_netcdf')
    def test_with_output(self, m):
        """Tests with an intermediate_output

        Tests that save_netcdf is called with object and string, and
        wrapped function returns the result.

        """
        # pylint disable is needed as it can't see the wrappers output kwarg.
        # pylint: disable=E1123
        result = wrapped_with_intermediate_output(2, intermediate_output="foo")
        m.assert_called_with(True, 'foo')
        self.assertEqual(result, 4)


class Test_unbracket(unittest.TestCase):
    """Test the unbracket function"""

    def test_basic(self):
        """Tests that a list of strings changes '[' into nested lists"""
        to_test = ['foo', '[', 'bar', 'a', 'b', ']',
                   '[', 'baz', 'c', ']', '-o', 'z']
        expected = ['foo', ['bar', 'a', 'b'], ['baz', 'c'], '-o', 'z']
        result = unbracket(to_test)
        self.assertEqual(result, expected)

    def test_mismatched_open_brackets(self):
        """Tests if there isn't a corresponding ']' it raises an error"""
        msg = 'Mismatched bracket at position'
        with self.assertRaisesRegex(ValueError, msg):
            unbracket(['foo', '[', 'bar'])

    def test_mismatched_close_brackets(self):
        """Tests if there isn't a corresponding '[' it raises an error"""
        msg = 'Mismatched bracket at position'
        with self.assertRaisesRegex(ValueError, msg):
            unbracket(['foo', ']', 'bar'])


if __name__ == '__main__':
    unittest.main()

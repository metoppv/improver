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
from unittest.mock import patch, mock_open

import clize

from improver.cli import docutilize, HelpForNapoleonDocstring, ObjectAsStr


def dummy_function(first, second, third):
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
    return first + second * third


def empty_dummy(first, second, third):
    return first + second * third


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


class test_ObjAsStr(unittest.TestCase):
    """Unit tests for ObjAsStr"""
    def test_ObjectAsStr(self):
        jsn = {'file': 1}
        x = ObjectAsStr(jsn)





if __name__ == '__main__':
    unittest.main()

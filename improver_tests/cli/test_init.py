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

import numpy as np
import improver
from improver.cli import (
    clizefy,
    create_constrained_inputcubelist_converter,
    docutilize,
    inputcube,
    inputjson,
    maybe_coerce_with,
    unbracket,
    with_intermediate_output,
    with_output,
)
from improver.utilities.load import load_cube
from iris.cube import CubeList

from ..set_up_test_cubes import set_up_variable_cube


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
        # pylint: disable=unexpected-keyword-arg
        result = wrapped_with_intermediate_output(2, intermediate_output="foo")
        m.assert_called_with(True, 'foo')
        self.assertEqual(result, 4)


def replace_load_with_extract(func, cube, constraints=None):
    """Function to replicate the call to maybe_coerce_with within
    create_constrained_inputcubelist_converter.

    Args:
        func:
            Unused argument for the purpose of replicating the
            maybe_coerce_with interface.
        cube (iris.cube.Cube or iris.cube.CubeList):
            Cube or CubeList to be extracted from.
        constraints (str or None):
            Expected name of cube for extraction.

    Returns:
        iris.cube.Cube:
            The extracted cube.

    Raises:
        ValueError:
            Error to replicate the behaviour of the call of maybe_coerce_with,
            where loading a cube with an invalid constraint results in a
            ValueError.
    """
    del func
    if constraints:
        cube = cube.extract(constraints)
    if isinstance(cube, CubeList):
        cube, = cube.copy()
    if cube is None:
        raise ValueError
    return cube


class Test_create_constrained_inputcubelist_converter(unittest.TestCase):
    """Tests the creature constraint_inputcubelist_converter"""

    def setUp(self):
        """Sets up some example names to use"""
        self.speed_name = 'wind_speed'
        self.direction_name = 'wind_from_direction'

    @patch('improver.cli.maybe_coerce_with', side_effect=['cube1', 'cube2'])
    def test_basic(self, m):
        """Tests that maybe_coerce_with is called twice. both times with
        load_cube as the first argument, a list of 2 items as second argument
        and the constraint list of the two strings given to
        create_constrained_inputcubelist_converter.

        the returned result is the first two arguments used by the Mocked
        side_effect.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name, self.direction_name])
        foobe_list = ['foo', 'bar']
        result = constrained_list(foobe_list)
        m.assert_any_call(load_cube, foobe_list, constraints=self.speed_name)
        m.assert_any_call(load_cube, foobe_list,
                          constraints=self.direction_name)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'cube1')
        self.assertEqual(result[1], 'cube2')

    @patch('improver.cli.maybe_coerce_with', side_effect=['cube1'])
    def test_lists(self, m):
        """Tests that maybe_coerce_with is called once with
        load_cube as the first argument, a list of 1 item as second argument
        and the first constraint list given to
        create_constrained_inputcubelist_converter.

        the returned result is the first argument used by the Mocked
        side_effect.

        Because the first constraint list of speed_name returns a full match,
        the second one gets skipped.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name], [self.direction_name])
        result = constrained_list(['foo'])
        m.assert_any_call(load_cube, ['foo'], constraints=self.speed_name)
        self.assertEqual(m.call_count, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 'cube1')

    @patch('improver.cli.maybe_coerce_with', side_effect=[ValueError, 'cube1'])
    def test_when_first_list_does_not_match(self, m):
        """Tests that maybe_coerce_with is called.
        With load_cube as the first argument,
        a list of 2 items as second argument
        and the constraint of the second list given to
        create_constrained_inputcubelist_converter.

        the returned result is the first two arguments used by the Mocked
        side_effect.

        Because the first call to maybe_coerce_with returns a ValueError,
        the second list of constraints will be used.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name], ['cats'])
        foobe_list = ['foo']
        result = constrained_list(foobe_list)
        for constraint in [self.speed_name, 'cats']:
            m.assert_any_call(load_cube, foobe_list, constraints=constraint)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 'cube1')

    @patch('improver.cli.maybe_coerce_with', side_effect=['cube1', 'cube2'])
    def test_different_length_constraints_big_first_select_first(self, m):
        """Tests when the two constraint lists are different sizes.
        The bigger of the two constraints is first.
        The selected list is first.
        Tests that maybe_coerce_with is called with
        load_cube as the first argument, the list given to
        create_constrained_inputcubelist_converter as second argument
        and the selected constraints.

        the returned result is the first two arguments used by the Mocked
        side_effect.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name, self.direction_name],
            ['cats']
        )
        foobe_list = ['foo', 'bar']
        result = constrained_list(foobe_list)
        for constraint in [self.speed_name, self.direction_name]:
            m.assert_any_call(load_cube, foobe_list, constraints=constraint)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'cube1')
        self.assertEqual(result[1], 'cube2')

    @patch('improver.cli.maybe_coerce_with', side_effect=[ValueError,
                                                          'cube1', 'cube2'])
    def test_different_length_constraints_big_first_select_second(self, m):
        """Tests when the two constraint lists are different sizes.
        The bigger of the two constraints is first.
        The selected list is second.
        Tests that maybe_coerce_with is called with
        load_cube as the first argument, the list given to
        create_constrained_inputcubelist_converter as second argument
        and the selected constraints.

        the returned result is the first two arguments used by the Mocked
        side_effect.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name, self.direction_name],
            ['cats']
        )
        foobe_list = ['foo']
        result = constrained_list(foobe_list)
        for constraint in [self.speed_name, 'cats']:
            m.assert_any_call(load_cube, foobe_list, constraints=constraint)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 'cube1')

    @patch('improver.cli.maybe_coerce_with', side_effect=['cube1'])
    def test_different_length_constraints_big_last_select_first(self, m):
        """Tests when the two constraint lists are different sizes.
        The bigger of the two constraints is last.
        The selected list is first.
        Tests that maybe_coerce_with is called with
        load_cube as the first argument, the list given to
        create_constrained_inputcubelist_converter as second argument
        and the selected constraints.

        the returned result is the first two arguments used by the Mocked
        side_effect.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            ['cats'],
            [self.speed_name, self.direction_name]
        )
        foobe_list = ['foo']
        result = constrained_list(foobe_list)
        m.assert_any_call(load_cube, foobe_list, constraints='cats')
        self.assertEqual(m.call_count, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 'cube1')

    @patch('improver.cli.maybe_coerce_with', side_effect=[ValueError,
                                                          'cube1', 'cube2'])
    def test_different_length_constraints_big_last_select_second(self, m):
        """Tests when the two constraint lists are different sizes.
        The bigger of the two constraints is last.
        The selected list is second.
        Tests that maybe_coerce_with is called with
        load_cube as the first argument, the list given to
        create_constrained_inputcubelist_converter as second argument
        and the selected constraints.

        the returned result is the first two arguments used by the Mocked
        side_effect.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            ['cats'],
            [self.speed_name, self.direction_name]
        )
        foobe_list = ['foo', 'bar']
        result = constrained_list(foobe_list)
        for constraint in ['cats', self.speed_name, self.direction_name]:
            m.assert_any_call(load_cube, foobe_list, constraints=constraint)
        self.assertEqual(m.call_count, 3)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'cube1')
        self.assertEqual(result[1], 'cube2')

    @patch('improver.cli.maybe_coerce_with', side_effect=['cube1',
                                                          'cube1', 'cube2'])
    def test_when_first_match_but_wrong_size(self, m):
        """Tests when there is a match but the length is wrong, continues to
        find a full match.
        Calls maybe_coerce_with three times due to the first match not being
        the same length as the input_list.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            ['cats'],
            ['cats', self.direction_name]
        )
        foobe_list = ['foo', 'bar']
        result = constrained_list(foobe_list)
        for constraint in ['cats', 'cats', self.direction_name]:
            m.assert_any_call(load_cube, foobe_list, constraints=constraint)
        self.assertEqual(m.call_count, 3)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'cube1')
        self.assertEqual(result[1], 'cube2')

    @patch('improver.cli.maybe_coerce_with', side_effect=ValueError)
    def test_err_when_no_match(self, m):
        """Tests that raises an error when no cubes match any constraints.
        Tests that assertEqual is called for the number of constraint lists.
        """
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name], [self.direction_name])
        msg = 'No cubes matching'
        with self.assertRaisesRegex(ValueError, msg):
            constrained_list(['foo'])
        for constraint in [self.speed_name, self.direction_name]:
            m.assert_any_call(load_cube, ['foo'], constraints=constraint)
        self.assertEqual(m.call_count, 2)

    @patch('improver.cli.maybe_coerce_with', side_effect=['Cube1'])
    def test_err_when_match_wrong_size(self, m):
        """Tests that raises an error when no cubes match any constraints"""
        constrained_list = create_constrained_inputcubelist_converter(
            [self.speed_name])
        msg = 'Partial match found'
        with self.assertRaisesRegex(ValueError, msg):
            constrained_list(['foo', 'bar'])
        m.assert_any_call(load_cube, ['foo', 'bar'],
                          constraints=self.speed_name)
        self.assertEqual(m.call_count, 1)


class Test_clizefy(unittest.TestCase):
    """Test the clizefy decorator function"""

    @patch('improver.cli.docutilize', return_value=None)
    def test_basic(self, m):
        """Tests basic behaviour"""

        def func():
            """Dummy"""

        clizefied = clizefy(func)
        self.assertIs(func, clizefied)
        self.assertTrue(hasattr(clizefied, 'cli'))
        clizefied_cli = clizefied.cli
        clizefied_again = clizefy()(clizefied)
        self.assertIs(clizefied_cli, clizefied_again.cli)
        clizefied_cli('argv[0]', '--help')
        m.assert_called_with(func.__doc__)


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

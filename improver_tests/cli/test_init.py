# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
from iris.cube import CubeList
from iris.exceptions import ConstraintMismatchError

import improver
from improver.cli import (
    clizefy,
    create_constrained_inputcubelist_converter,
    docutilize,
    inputcube,
    inputcubelist,
    inputjson,
    maybe_coerce_with,
    run_main,
    unbracket,
    with_output,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


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


@clizefy
@with_output
def wrapped_with_output(first):
    """dummy function for testing with_output wrapper"""
    return dummy_function(first)


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

    @patch("improver.cli.maybe_coerce_with", return_value="return")
    def test_basic(self, m):
        """Tests that input cube calls load_cube with the string"""
        result = inputcube("foo")
        m.assert_called_with(improver.utilities.load.load_cube, "foo")
        self.assertEqual(result, "return")


class Test_inputcubelist(unittest.TestCase):
    """Tests the input cubelist function"""

    @patch("improver.cli.maybe_coerce_with", return_value="return")
    def test_basic(self, m):
        """Tests that input cubelist calls load_cubelist with the string"""
        result = inputcubelist("foo")
        m.assert_called_with(improver.utilities.load.load_cubelist, "foo")
        self.assertEqual(result, "return")


class Test_inputjson(unittest.TestCase):
    """Tests the input cube function"""

    @patch("improver.cli.maybe_coerce_with", return_value={"mocked": 1})
    def test_basic(self, m):
        """Tests that input json calls load_json_or_none with the string"""
        result = inputjson("foo")
        m.assert_called_with(improver.utilities.cli_utilities.load_json_or_none, "foo")
        self.assertEqual(result, {"mocked": 1})


class Test_with_output(unittest.TestCase):
    """Tests the with_output wrapper"""

    @patch("improver.utilities.save.save_netcdf")
    def test_without_output(self, m):
        """Tests that the result of the wrapped function is returned"""
        result = wrapped_with_output.cli("argv[0]", "2")
        m.assert_not_called()
        self.assertEqual(result, 4)

    @patch("improver.utilities.save.save_netcdf")
    def test_with_output(self, m):
        """Tests that save_netcdf is called with object and string, default
        compression_level=1 and default least_significant_digit=None"""
        # pylint disable is needed as it can't see the wrappers output kwarg.
        # pylint: disable=E1123
        result = wrapped_with_output.cli("argv[0]", "2", "--output=foo")
        m.assert_called_with(4, "foo", 1, None)
        self.assertEqual(result, None)

    @patch("improver.utilities.save.save_netcdf")
    def test_with_output_compression_level(self, m):
        """Tests that save_netcdf is called with object and string, compression-level=9 and default least-significant-digit=None"""
        # pylint disable is needed as it can't see the wrappers output kwarg.
        # pylint: disable=E1123
        result = wrapped_with_output.cli(
            "argv[0]", "2", "--output=foo", "--compression-level=9"
        )
        m.assert_called_with(4, "foo", 9, None)
        self.assertEqual(result, None)

    @patch("improver.utilities.save.save_netcdf")
    def test_with_output_no_compression(self, m):
        """Tests that save_netcdf is called with object and string, compression-level=0 and default least-significant-digit=None"""
        # pylint disable is needed as it can't see the wrappers output kwarg.
        # pylint: disable=E1123
        result = wrapped_with_output.cli(
            "argv[0]", "2", "--output=foo", "--compression-level=0"
        )
        m.assert_called_with(4, "foo", 0, None)
        self.assertEqual(result, None)

    @patch("improver.utilities.save.save_netcdf")
    def test_with_output_with_least_significant_figure(self, m):
        """Tests that save_netcdf is called with object and string, compression-level=0 and least-significant-digit=2 """
        # pylint disable is needed as it can't see the wrappers output kwarg.
        # pylint: disable=E1123
        result = wrapped_with_output.cli(
            "argv[0]",
            "2",
            "--output=foo",
            "--compression-level=0",
            "--least-significant-digit=2",
        )
        m.assert_called_with(4, "foo", 0, 2)
        self.assertEqual(result, None)


def setup_for_mock():
    """Function that returns a CubeList of wind_speed and wind_from_direction

    These cubes should be the same as the setup cubes.

    Returns:
        iris.cube.CubeList:
            The CubeList.
    """
    return CubeList(
        [
            set_up_variable_cube(
                data=np.zeros((2, 2), dtype=np.float32), name="wind_speed"
            ),
            set_up_variable_cube(
                data=np.zeros((2, 2), dtype=np.float32), name="wind_from_direction"
            ),
        ]
    )


class Test_create_constrained_inputcubelist_converter(unittest.TestCase):
    """Tests the create_constrained_inputcubelist_converter"""

    def setUp(self):
        data = np.zeros((2, 2), dtype=np.float32)
        self.wind_speed_cube = set_up_variable_cube(data, name="wind_speed")
        self.wind_dir_cube = set_up_variable_cube(data, name="wind_from_direction")
        self.wind_cubes = CubeList([self.wind_speed_cube, self.wind_dir_cube])

    def test_basic(self):
        """Tests a basic creation of create_constrained_inputcubelist_converter"""
        func = create_constrained_inputcubelist_converter(
            lambda cube: cube.name()
            in ["wind_speed", "airspeed_velocity_of_unladen_swallow"]
        )
        result = func(self.wind_cubes)
        self.assertEqual(self.wind_speed_cube, result[0])
        self.assertEqual(1, len(result))

    def test_extracting_two_cubes(self):
        """Tests a creation of with two cube names"""
        func = create_constrained_inputcubelist_converter(
            "wind_speed", "wind_from_direction"
        )
        result = func(self.wind_cubes)
        self.assertEqual(self.wind_speed_cube, result[0])
        self.assertEqual(self.wind_dir_cube, result[1])
        self.assertEqual(2, len(result))

    @patch("improver.cli.maybe_coerce_with", return_value=setup_for_mock())
    def test_basic_given_str(self, mocked_maybe_coerce):
        """Tests that a str is given to maybe_coerce_with which would return a CubeList."""
        func = create_constrained_inputcubelist_converter(
            "wind_speed", "wind_from_direction"
        )
        result = func(self.wind_cubes)
        self.assertEqual(self.wind_speed_cube, result[0])
        self.assertEqual(self.wind_dir_cube, result[1])
        self.assertEqual(2, len(result))
        mocked_maybe_coerce.assert_called_once()

    def test_list_two_valid(self):
        """Tests that one cube is loaded from each list."""
        func = create_constrained_inputcubelist_converter(
            lambda cube: cube.name()
            in ["airspeed_velocity_of_unladen_swallow", "wind_speed"],
            lambda cube: cube.name() in ["direction_of_swallow", "wind_from_direction"],
        )
        result = func(self.wind_cubes)
        self.assertEqual(self.wind_speed_cube, result[0])
        self.assertEqual(self.wind_dir_cube, result[1])
        self.assertEqual(2, len(result))

    def test_list_two_diff_shapes(self):
        """Tests that one cube is loaded from each list
        when the two lists are different sizes.
        """
        func = create_constrained_inputcubelist_converter(
            "wind_speed",
            lambda cube: cube.name() in ["direction_of_swallow", "wind_from_direction"],
        )
        result = func(self.wind_cubes)
        self.assertEqual(self.wind_speed_cube, result[0])
        self.assertEqual(self.wind_dir_cube, result[1])
        self.assertEqual(2, len(result))

    def test_list_no_match(self):
        """Tests that providing no valid constraints raises a ConstraintMismatchError."""
        func = create_constrained_inputcubelist_converter(
            "airspeed_velocity_of_unladen_swallow",
        )
        with self.assertRaisesRegex(ConstraintMismatchError, "^Got 0 cubes"):
            func(self.wind_cubes)

    def test_two_valid_cubes(self):
        """Tests that providing 2 valid constraints raises a ConstraintMismatchError."""
        func = create_constrained_inputcubelist_converter(
            lambda cube: cube.name() in ["wind_speed", "wind_from_direction"],
        )
        with self.assertRaisesRegex(ConstraintMismatchError, "^Got 2 cubes"):
            func(self.wind_cubes)


class Test_clizefy(unittest.TestCase):
    """Test the clizefy decorator function"""

    @patch("improver.cli.docutilize", return_value=None)
    def test_basic(self, m):
        """Tests basic behaviour"""

        def func():
            """Dummy"""

        clizefied = clizefy(func)
        self.assertIs(func, clizefied)
        self.assertTrue(hasattr(clizefied, "cli"))
        clizefied_cli = clizefied.cli
        clizefied_again = clizefy()(clizefied)
        self.assertIs(clizefied_cli, clizefied_again.cli)
        clizefied_cli("argv[0]", "--help")
        m.assert_called_with(func.__doc__)


class Test_unbracket(unittest.TestCase):
    """Test the unbracket function"""

    def test_basic(self):
        """Tests that a list of strings changes '[' into nested lists"""
        to_test = ["foo", "[", "bar", "a", "b", "]", "[", "baz", "c", "]", "-o", "z"]
        expected = ["foo", ["bar", "a", "b"], ["baz", "c"], "-o", "z"]
        result = unbracket(to_test)
        self.assertEqual(result, expected)

    def test_mismatched_open_brackets(self):
        """Tests if there isn't a corresponding ']' it raises an error"""
        msg = "Mismatched bracket at position"
        with self.assertRaisesRegex(ValueError, msg):
            unbracket(["foo", "[", "bar"])

    def test_mismatched_close_brackets(self):
        """Tests if there isn't a corresponding '[' it raises an error"""
        msg = "Mismatched bracket at position"
        with self.assertRaisesRegex(ValueError, msg):
            unbracket(["foo", "]", "bar"])


def test_import_cli():
    """Test if `import improver.cli` pulls in heavy stuff like numpy.

    Large imports cause commands like help or exit due to incorrect arguments
    to be unnecessarily slow, so it's best to test we don't have them.
    """
    import subprocess  # nosec
    import sys

    # Must run in a subprocess to ensure "fresh" Python interpreter without
    # modules pulled by other tests
    script = (
        "import improver.cli, sys; "
        'assert "numpy" not in sys.modules, '
        '"rogue numpy import via improver.cli"'
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # nosec


def test_help_no_stderr():
    """Test if help writes to sys.stderr."""
    import contextlib
    import io

    stderr = io.StringIO()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            run_main(["improver", "help"])
        except SystemExit:
            pass
    result = stderr.getvalue()
    assert not result, "unexpected output on STDERR:\n" + result


if __name__ == "__main__":
    unittest.main()

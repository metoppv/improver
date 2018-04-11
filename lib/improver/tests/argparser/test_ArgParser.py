# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for argparser.ArgParser."""


import os
import unittest
from mock import patch

import argparse
from improver.argparser import ArgParser


# We might one day want to move this up to a more central place.
class QuietTestCase(unittest.TestCase):

    """A subclass of unittest.TestCase which prevents writing to stderr."""

    @classmethod
    def setUpClass(cls):
        """Patch the class by redirecting stderr to /dev/null."""
        cls.patch = patch('sys.stderr', open(os.devnull, 'w'))
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        """Stop the patch which redirects stderr to /dev/null."""
        cls.patch.stop()


class Test_init(QuietTestCase):

    """Test the __init__ method."""

    def test_create_argparser_with_no_arguments(self):
        """Test that creating an ArgParser with no arguments has no
        arguments."""

        compulsory_arguments = {}

        # it doesn't matter what the centralized arguments are, because we
        # select None of them
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None, specific_arguments=None)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertEqual(len(args), 0)

    def test_create_argparser_only_compulsory_arguments(self):
        """Test that creating an ArgParser with only compulsory arguments
        adds only the compulsory arguments."""

        compulsory_arguments = {'foo': (['--foo'], {})}

        # it doesn't matter what the centralized arguments are, because we
        # select None of them
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None, specific_arguments=None)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertItemsEqual(args, ['foo'])

    def test_create_argparser_only_centralized_arguments(self):
        """Test that creating an ArgParser with only centralized arguments
        adds only the selected centralized arguments."""

        compulsory_arguments = {}
        centralized_arguments = {'foo': (['--foo'], {})}

        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                       centralized_arguments):
                parser = ArgParser(central_arguments=['foo'],
                                   specific_arguments=None)
                args = parser.parse_args()
                args = vars(args).keys()
                self.assertItemsEqual(args, ['foo'])

    def test_create_argparser_only_specific_arguments(self):
        """Test that creating an ArgParser with only specific arguments
        adds only the specific arguments."""

        compulsory_arguments = {}
        specific_arguments = [(['--foo'], {})]

        # it doesn't matter what the centralized arguments are, because we
        # select None of them
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None,
                               specific_arguments=specific_arguments)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertItemsEqual(args, ['foo'])

    def test_create_argparser_compulsory_and_centralized_arguments(self):
        """Test that creating an ArgParser with compulsory and centralized
        arguments adds both of these and no others."""

        compulsory_arguments = {'foo': (['--foo'], {})}
        centralized_arguments = {'bar': (['--bar'], {})}

        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                       centralized_arguments):
                parser = ArgParser(central_arguments=['bar'],
                                   specific_arguments=None)
                args = parser.parse_args()
                args = vars(args).keys()
                self.assertItemsEqual(args, ['foo', 'bar'])

    def test_create_argparser_compulsory_and_specfic_arguments(self):
        """Test that creating an ArgParser with compulsory and specific
        arguments adds both of these and no others."""

        compulsory_arguments = {'foo': (['--foo'], {})}
        specific_arguments = [(['--bar'], {})]

        # it doesn't matter what the centralized arguments are, because we
        # select None of them
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None,
                               specific_arguments=specific_arguments)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertItemsEqual(args, ['foo', 'bar'])

    def test_create_argparser_all_arguments(self):
        """Test that creating an ArgParser with compulsory, centralized and
        specific arguments adds the arguments from all 3 collections."""

        compulsory_arguments = {'foo': (['--foo'], {})}
        centralized_arguments = {'bar': (['--bar'], {})}
        specific_arguments = [(['--baz'], {})]

        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                       centralized_arguments):
                parser = ArgParser(central_arguments=['bar'],
                                   specific_arguments=specific_arguments)
                args = parser.parse_args()
                args = vars(args).keys()
                self.assertItemsEqual(args, ['foo', 'bar', 'baz'])


class Test_add_arguments(QuietTestCase):

    """Test the add_arguments method."""

    def test_adding_multiple_arguments(self):
        """Test that we can successfully add multiple arguments to the
        ArgParser."""

        # we will not actually pass anything in, so the Namespace will receive
        # the defaults (if any) - only check the keys of the Namespace derived
        # dictionary
        args_to_add = [
                        (['--foo'], {}),
                        (['--bar', '--b'], {})
                      ]
        expected_namespace_keys = ['foo', 'bar']  # + compulsory...

        # explicitly pass nothing in - will only have compulsory arguments
        # and the ones we added...
        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        parser.add_arguments(args_to_add)
        result_args = parser.parse_args()
        result_args = vars(result_args).keys()
        # we could also add compulsory arguments to expected_namespace_keys
        # and then assertItemsEqual - (order unimportant), but this
        # is unnecessary - just use loop:
        # (or we could patch compulsory arguments to be an empty dictionary)
        for expected_arg in expected_namespace_keys:
            self.assertIn(expected_arg, result_args)

    def test_adding_argument_with_defined_kwargs_dict(self):
        """Test that we can successfully add an argument to the ArgParser,
        when the argspec contained kwargs."""

        # length of argspec is 2...
        args_to_add = [
                        (['--foo'], {'action': 'store_true'}),
                      ]
        expected_arg = 'foo'

        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        parser.add_arguments(args_to_add)
        result_args = parser.parse_args()
        result_args = vars(result_args).keys()
        self.assertIn(expected_arg, result_args)

    def test_adding_argument_with_missing_kwargs_dict(self):
        """Test that we can successfully add an argument to the ArgParser,
        when the argspec did not contain kwargs."""

        # length of argspec is 1...
        args_to_add = [
                        (['--foo'],),
                      ]
        expected_arg = 'foo'

        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        parser.add_arguments(args_to_add)
        result_args = parser.parse_args()
        result_args = vars(result_args).keys()
        self.assertIn(expected_arg, result_args)

    def test_adding_single_argument_with_unexpected_length_argspec(self):
        """Test that attempting to add an argument to the ArgParser when
        the wrong format argspec raises an exception."""

        # length of argspec is 3 - this is unexpected
        args_to_add = [
                        (['--foo'], 'bar', {}),
                      ]

        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        with self.assertRaises(AttributeError):
            parser.add_arguments(args_to_add)

    def test_adding_empty_argument_list_does_nothing(self):
        """Test that attempting to add an empty list of argspecs to the
        ArgParser does not add any new arguments."""

        args_to_add = []

        # add a specific (optional) argument - ensures that even if there are
        # no compulsory arguments, we have something...
        # adding arguments after calling parse_args/args will do nothing, so
        # instead create 2 instances:
        parser1 = ArgParser(central_arguments=None,
                            specific_arguments=[[['--optional'], {}]])

        parser2 = ArgParser(central_arguments=None,
                            specific_arguments=[[['--optional'], {}]])

        parser2.add_arguments(args_to_add)
        self.assertEqual(parser1.parse_args(), parser2.parse_args())


class Test__args_from_selected_central_args(QuietTestCase):

    """Test the _args_from_selected_args method."""

    def setUp(self):
        """Patch the ArgParser.CENTRALIZED_ARGUMENTS dictionary so test cases
        know exactly what to expect."""
        DUMMY_CENTRALIZED_ARGUMENTS = {'foo': (['central_foo'], {}),
                                       'bar': (['central_bar'], {})}
        self.patch = patch(
                        'improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                        DUMMY_CENTRALIZED_ARGUMENTS)
        self.patch.start()

    def tearDown(self):
        """Stop the patch of ArgParser.CENTRALIZED_ARGUMENTS."""
        self.patch.stop()

    def test_retrieving_known_central_argument(self):
        """Test that we can successfully retrieve centralized arguments from
        the centralized argument dictionary."""

        central_args_to_fetch = ('foo',)
        expected_result = [(['central_foo'], {})]
        self.assertEqual(ArgParser._args_from_selected_central_args(
            central_args_to_fetch), expected_result)

    def test_retrieving_unknown_central_argument(self):
        """Test that we raise an exception when attempting to retrieve
        centralized arguments which are not centralized argument dictionary."""

        central_args_to_fetch = ('missing_central_arg',)
        with self.assertRaises(KeyError):
            ArgParser._args_from_selected_central_args(central_args_to_fetch)


class Test_args(QuietTestCase):

    """Test the args method."""

    def test_result_same_as_parse_args(self):
        """Test that the args method returns the same result as parse_args."""

        # we don't need any arguments, but if we choose to include some
        # (more useful than checking for an empty list), then they must be
        # optional as they will not have been passed in.
        optional_argument_spec = [[['--optional'], {}]]
        parser = ArgParser(central_arguments=None,
                           specific_arguments=optional_argument_spec)
        self.assertEqual(parser.parse_args(), parser.args())


class Test_wrong_args_error(QuietTestCase):

    """Test the wrong_args_error method."""

    def test_error_raised(self, args='foo', method='bar'):
        """Test that an exception is raised containing the args and method."""

        msg = ("Method: {} does not accept arguments: {}".format(
                   method, args))
        with self.assertRaises(SystemExit, msg=msg):
            ArgParser().wrong_args_error(args, method)


if __name__ == '__main__':
    unittest.main()

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
from unittest.mock import patch

from improver.argparser import ArgParser


# We might one day want to move this up to a more central place.
class QuietTestCase(unittest.TestCase):

    """A subclass of unittest.TestCase which prevents writing to stderr and
    calling of sys.exit."""

    @classmethod
    def setUpClass(cls):
        """Patch the class by redirecting stderr to /dev/null, and disabling
        calls to sys.exit. Currently used to prevent
        ArgumentParser.parse_args() from writing its output to the screen and
        exiting early when using unittest discover."""
        cls.file_handle = open(os.devnull, 'w')
        cls.stderr_patch = patch('sys.stderr', cls.file_handle)
        cls.exit_patch = patch('sys.exit')
        cls.stderr_patch.start()
        cls.exit_patch.start()

    @classmethod
    def tearDownClass(cls):
        """Stop the patches which redirect stderr to /dev/null and prevents
        sys.exit from being called."""
        cls.file_handle.close()
        cls.stderr_patch.stop()
        cls.exit_patch.stop()


class Test_init(QuietTestCase):

    """Test the __init__ method."""

    def test_create_argparser_with_no_arguments(self):
        """Test that creating an ArgParser with no arguments has no
        arguments."""

        compulsory_arguments = {}

        # it doesn't matter what the centralized arguments are, because we
        # select None of them - we only need to patch the COMPULSORY_ARGUMENTS
        # to ensure there are none of them
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
        # select None of them - only patch COMPULSORY_ARGUMENTS so we know
        # what to expect
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None, specific_arguments=None)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertCountEqual(args, ['foo'])

    def test_create_argparser_fails_with_unknown_centralized_argument(self):
        """Test that we raise an exception when attempting to retrieve
        centralized arguments which are not centralized argument dictionary."""

        centralized_arguments = {'foo': (['--foo'], {})}

        central_args_to_fetch = ('missing_central_arg',)
        # patch the CENTRALIZED_ARGUMENTS so we know that `missing_central_arg`
        # is not there, and we can raise an exception
        with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                   centralized_arguments):
            with self.assertRaises(KeyError):
                ArgParser(central_arguments=central_args_to_fetch,
                          specific_arguments=None)

    def test_create_argparser_only_centralized_arguments(self):
        """Test that creating an ArgParser with only centralized arguments
        adds only the selected centralized arguments."""

        compulsory_arguments = {}
        centralized_arguments = {'foo': (['--foo'], {})}

        # patch the COMPULSORY_ARGUMENTS to an empty dict (so there are none)
        # and patch CENTRALIZED_ARGUMENTS so we know that `foo` can be selected
        # from it
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                       centralized_arguments):
                parser = ArgParser(central_arguments=['foo'],
                                   specific_arguments=None)
                args = parser.parse_args()
                args = vars(args).keys()
                self.assertCountEqual(args, ['foo'])

    def test_create_argparser_only_specific_arguments(self):
        """Test that creating an ArgParser with only specific arguments
        adds only the specific arguments."""

        compulsory_arguments = {}
        specific_arguments = [(['--foo'], {})]

        # it doesn't matter what the centralized arguments are, because we
        # select None of them - patch the COMPULSORY_ARGUMENTS to be an empty
        # dict so that we don't add any of them
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None,
                               specific_arguments=specific_arguments)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertCountEqual(args, ['foo'])

    def test_create_argparser_compulsory_and_centralized_arguments(self):
        """Test that creating an ArgParser with compulsory and centralized
        arguments adds both of these and no others."""

        compulsory_arguments = {'foo': (['--foo'], {})}
        centralized_arguments = {'bar': (['--bar'], {})}

        # patch the COMPULSORY_ARGUMENTS so we know that `foo` exists
        # and the CENTRALIZED_ARGUMENTS so we know that `bar` exists.
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                       centralized_arguments):
                parser = ArgParser(central_arguments=['bar'],
                                   specific_arguments=None)
                args = parser.parse_args()
                args = vars(args).keys()
                self.assertCountEqual(args, ['foo', 'bar'])

    def test_create_argparser_compulsory_and_specfic_arguments(self):
        """Test that creating an ArgParser with compulsory and specific
        arguments adds both of these and no others."""

        compulsory_arguments = {'foo': (['--foo'], {})}
        specific_arguments = [(['--bar'], {})]

        # it doesn't matter what the centralized arguments are, because we
        # select None of them - patch only the COMPULSORY_ARGUMENTS so we know
        # that `foo` is added from here
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            parser = ArgParser(central_arguments=None,
                               specific_arguments=specific_arguments)
            args = parser.parse_args()
            args = vars(args).keys()
            self.assertCountEqual(args, ['foo', 'bar'])

    def test_create_argparser_all_arguments(self):
        """Test that creating an ArgParser with compulsory, centralized and
        specific arguments adds the arguments from all 3 collections."""

        compulsory_arguments = {'foo': (['--foo'], {})}
        centralized_arguments = {'bar': (['--bar'], {})}
        specific_arguments = [(['--baz'], {})]

        # patch both the COMPULSORY_ARGUMENTS and CENTRALIZED_ARGUMENTS, so
        # that `foo` and `bar` are added from these (respectively)
        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.ArgParser.CENTRALIZED_ARGUMENTS',
                       centralized_arguments):
                parser = ArgParser(central_arguments=['bar'],
                                   specific_arguments=specific_arguments)
                args = parser.parse_args()
                args = vars(args).keys()
                self.assertCountEqual(args, ['foo', 'bar', 'baz'])

    def test_argparser_compulsory_args_has_profile(self):
        """Test that creating an ArgParser instance with the compulsory
        arguments adds the profiling options."""

        expected_profile_options = ['profile', 'profile_file']
        parser = ArgParser(central_arguments=None, specific_arguments=None)
        args = parser.parse_args()
        args = vars(args).keys()
        self.assertCountEqual(args, expected_profile_options)


class Test_add_arguments(QuietTestCase):

    """Test the add_arguments method."""

    def test_adding_multiple_arguments(self):
        """Test that we can successfully add multiple arguments to the
        ArgParser."""

        # we will not actually pass anything in, so the Namespace will receive
        # the defaults (if any) - only check the keys of the Namespace derived
        # dictionary
        args_to_add = [(['--foo'], {}),
                       (['--bar', '--b'], {})]

        expected_namespace_keys = ['foo', 'bar']  # + compulsory...

        # explicitly pass nothing in - will only have compulsory arguments
        # and the ones we added...
        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        parser.add_arguments(args_to_add)
        result_args = parser.parse_args()
        result_args = vars(result_args).keys()
        # we could also add compulsory arguments to expected_namespace_keys
        # and then assertCountEqual - (order unimportant), but this
        # is unnecessary - just use loop:
        # (or we could patch compulsory arguments to be an empty dictionary)
        for expected_arg in expected_namespace_keys:
            self.assertIn(expected_arg, result_args)

    def test_adding_argument_with_defined_kwargs_dict(self):
        """Test that we can successfully add an argument to the ArgParser,
        when the argspec contained kwargs."""

        # length of argspec is 2...
        args_to_add = [(['--foo'], {'default': 1})]
        expected_arg = 'foo'

        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        parser.add_arguments(args_to_add)
        result_args = parser.parse_args()
        result_args = vars(result_args).keys()
        self.assertIn(expected_arg, result_args)

    def test_adding_argument_with_defined_kwargs_dict_has_defualt(self):
        """Test that we can successfully add an argument to the ArgParser,
        when the argspec contained kwargs, and that the default value is
        captured."""

        args_to_add = [(['--one'], {'default': 1})]

        parser = ArgParser(central_arguments=None,
                           specific_arguments=None)

        parser.add_arguments(args_to_add)
        result_args = parser.parse_args()
        # `--one` was not passed in, so we pick up the default - let's check
        # they agree...
        self.assertEqual(1, result_args.one)

    def test_adding_single_argument_with_unexpected_length_argspec(self):
        """Test that attempting to add an argument to the ArgParser when
        the wrong format argspec raises an exception."""

        # length of argspec is 3 - this is unexpected
        args_to_add = [(['--foo'], 'bar', {})]

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


class Test_parse_args(QuietTestCase):

    """Test the parse_args method."""

    def test_profile_is_called_when_enabled(self):
        """Test that calling parse_args enables profiling when the --profile
        option is added."""

        # temporarily patch compulsory args so that profiling is enabled by
        # default
        compulsory_arguments = {'profile': (
                                    ['--profile'],
                                    {'default': True}),
                                'profile_file': (
                                    ['--profile-file'],
                                    {'default': None})}

        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.profile_hook_enable') as \
                    mock_profile:
                parser = ArgParser(central_arguments=None,
                                   specific_arguments=None)
                args = parser.parse_args()
                self.assertEqual(mock_profile.call_count, 1)

    def test_profile_is_not_called_when_disbaled(self):
        """Test that calling parse_args does not enable profiling when the
        --profile option is not added."""

        # temporarily patch compulsory args so that profiling is disabled by
        # default
        compulsory_arguments = {'profile': (
                                    ['--profile'],
                                    {'default': False}),
                                'profile_file': (
                                    ['--profile-file'],
                                    {'default': None})}

        with patch('improver.argparser.ArgParser.COMPULSORY_ARGUMENTS',
                   compulsory_arguments):
            with patch('improver.argparser.profile_hook_enable') as \
                    mock_profile:
                parser = ArgParser(central_arguments=None,
                                   specific_arguments=None)
                parser.parse_args()
                self.assertEqual(mock_profile.call_count, 0)


# inherit from only TestCase - we want to explicitly catch the SystemExit
class Test_wrong_args_error(unittest.TestCase):

    """Test the wrong_args_error method."""

    def test_error_raised(self, args='foo', method='bar'):
        """Test that an exception is raised containing the args and method."""

        msg = ("Method: {} does not accept arguments: {}".format(
               method, args))

        # argparser will write to stderr independently of SystemExit
        with open(os.devnull, 'w') as file_handle:
            with patch('sys.stderr', file_handle):
                with self.assertRaises(SystemExit, msg=msg):
                    ArgParser().wrong_args_error(args, method)


if __name__ == '__main__':
    unittest.main()

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
"""Unit tests for the warnings handler."""

import unittest
import warnings

from iris.tests import IrisTest

from improver.utilities.warnings_handler import ManageWarnings


def dummy_func(arg1, keyval1=None):
    """ Dummy function to test """
    if arg1 == 'User':
        msg = 'Raising User error'
        warnings.warn(msg)
    return arg1, keyval1


class Test__init__(IrisTest):

    """Test __init__ of ManageWarnings."""

    def test_basic(self):
        """Test the default values are set correctly."""
        plugin = ManageWarnings()
        self.assertIsNone(plugin.messages)
        self.assertIsNone(plugin.warning_types)
        self.assertFalse(plugin.record)

    def test_ignored_messages_not_list(self):
        """Test Raises type error if ignored_messages is not a list."""
        msg = "Expecting list of strings for ignored_messages"
        with self.assertRaisesRegex(TypeError, msg):
            ManageWarnings(ignored_messages="Wrong")

    def test_ignored_messages(self):
        """Test defaults set correctly when ignored_messages set."""
        messages = ["Testing", "Testing2"]
        plugin = ManageWarnings(ignored_messages=messages)
        warning_types = plugin.warning_types
        self.assertEqual(plugin.messages, messages)
        self.assertTrue(any(item == UserWarning
                            for item in warning_types))

    def test_ignored_messages_and_warning_types(self):
        """Test OK when ignored_messages and warning_types set."""
        messages = ["Testing", "Testing2"]
        warning_types = [UserWarning, PendingDeprecationWarning]
        plugin = ManageWarnings(ignored_messages=messages,
                                warning_types=warning_types)
        self.assertEqual(plugin.messages, messages)
        self.assertTrue(any(item == UserWarning
                            for item in warning_types))
        self.assertTrue(any(item == PendingDeprecationWarning
                            for item in warning_types))

    def test_mismatch_warning_types(self):
        """Test Raises error when warning_types does not match."""
        messages = ["Testing", "Testing2"]
        warning_types = [UserWarning]
        msg = "Length of warning_types"
        with self.assertRaisesRegex(ValueError, msg):
            ManageWarnings(ignored_messages=messages,
                           warning_types=warning_types)

    def test_record_is_true(self):
        """Test record is set to True."""
        plugin = ManageWarnings(record=True)
        self.assertIsNone(plugin.messages)
        self.assertIsNone(plugin.warning_types)
        self.assertTrue(plugin.record)


class Test__call__(IrisTest):

    """Test __call__ of ManageWarnings."""

    @ManageWarnings()
    def test_basic(self):
        """Test the Function still works with wrapper"""
        argval, keyval = dummy_func('Test')
        self.assertEqual(argval, 'Test')
        self.assertIsNone(keyval)

    @ManageWarnings(record=True)
    def test_warning_list(self, warning_list=None):
        """Test when record is True"""
        argval, keyval = dummy_func('Test2', keyval1='Test3')
        self.assertEqual(argval, 'Test2')
        self.assertEqual(keyval, 'Test3')
        self.assertEqual(warning_list, [])

    @ManageWarnings(record=True)
    def test_tests_warnings(self, warning_list=None):
        """Test picks up user error correctly"""
        dummy_func('User')
        user_warning_msg = 'Raising User error'
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(user_warning_msg in str(item)
                            for item in warning_list))

    @ManageWarnings(ignored_messages=["Raising User error"],
                    record=True)
    def test_traps_warnings(self, warning_list=None):
        """Test ignores user error correctly"""
        dummy_func('User')
        self.assertEqual(warning_list, [])


if __name__ == '__main__':
    unittest.main()

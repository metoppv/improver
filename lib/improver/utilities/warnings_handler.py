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
"""Classes and functions for managing warnings in the IMPROVER code."""

import sys
import warnings


class ManageWarnings:
    """
    A decorator used to manage the warnings that are raised by a function.
    Ignore a selection of warnings, and either raise any remaining warnings
    to standard error or record them in a list of warning objects.
    """
    def __init__(self, ignored_messages=None, warning_types=None,
                 record=False):
        """
        Set up a decorator with the warnings we want to ignore and what
        we want to do with any remaining warnings.

        Args:
            ignored_messages (list of str):
                A list of messages, one for each warning message we
                want to ignore.
            warning_types (list):
                A list containing the Warning category for each of the
                messages. If not provided then the Warning Category is
                assumed to be UserWarning for each of the messages.
            record (bool):
                A flag for whether to store any warnings that are not
                ignored. Default is False which means warnings go to
                standard error. When set to True the warnings are
                recorded in a warning list which is passed to the function
                being decorated.

        Raises:
            TypeError: ignored_messages not list.
            ValueError: Raise error if both ignored messages and warning_types
                        are given and they are not the same length.
        """
        if ignored_messages is not None:
            if not isinstance(ignored_messages, list):
                msg = 'Expecting list of strings for ignored_messages'
                raise TypeError(msg)
        self.messages = ignored_messages
        if warning_types is None and self.messages is not None:
            self.warning_types = [UserWarning for _ in self.messages]
        else:
            self.warning_types = warning_types
        self.record = record
        if self.messages and (len(self.warning_types) != len(self.messages)):
            message = ("Length of warning_types ({}) does no equal length"
                       "of warning messages({})")
            message = message.format(len(self.warning_types),
                                     len(self.messages))
            raise ValueError(message)

    @staticmethod
    def reset_warning_registry():
        """
        Clears the hidden __warningregistry__ attribute from
        all imported modules.
        """
        for mod in list(sys.modules.values()):
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()

    def __call__(self, func):
        """
        Call the decorator on a function.
        Wrap the function in the warnings.catch_warnings context manager
        and set up the filters to ignore the specified warnings.
        If we are recording the warnings, the list of warnings is passed
        to the input function as a keyword argument for checking if
        required.

        Args:
            func (function):
                A function that we want to wrap with this decorator.
        Returns:
            function:
                The wrapped function with the warnings context manager and
                necessary filters turned on.
        """
        def warnings_wrapper(*args, **kwargs):
            """
            Wrapper function to set up the warnings.catch_warnings context
            manager and the filters before calling the input function, with
            the warning_list if needed.
            """
            with warnings.catch_warnings(record=self.record) as warning_list:
                warnings.filterwarnings("always")
                self.reset_warning_registry()
                if self.messages is not None:
                    for message, warning_type in zip(self.messages,
                                                     self.warning_types):
                        warnings.filterwarnings("ignore", message,
                                                warning_type)
                if self.record:
                    result = func(*args, warning_list=warning_list, **kwargs)
                else:
                    result = func(*args, **kwargs)
                self.reset_warning_registry()
                return result
        return warnings_wrapper

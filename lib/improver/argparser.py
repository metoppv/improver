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
"""Common option utilities for improver CLIs."""

from argparse import ArgumentParser


class ArgParser(ArgumentParser):
    """Argument parser for improver CLIs.
    (Subclass of argparse.ArgumentParser)"""

    # CENTRALIZED_ARGUMENTS, is a dictionary where, for each element:
    #   the key is a string representing the argument name,
    #   the value is a list containing 2 elements:
    #       1. a list of strings containing the different flags
    #       2. a dictionary containing all of the kwargs which are passed
    #          to the ArgumentParser.add_argument() method

    # All CLIs should select something from this dictionary:
    # NB: --help included for free with ArgumentParser
    CENTRALIZED_ARGUMENTS = {
        'input_file': (
            ['input_filepath'],
            {}),
        'output_file': (
            ['output_filepath'],
            {}),
    }

    # *All* CLIs will use the options here (no option to disable them):
    COMPULSORY_ARGUMENTS = {
        # TODO: Implement (some of?) these:
        #        'debug': (
        #            ['--debug'],
        #            {'action': 'store_true'}),
        #        'dry_run': (
        #            ['--dry-run'],
        #            {'action': 'store_true'}),
        #        'profile' : (
        #            ['--profile'],
        #            {'action': 'store_true'}),
        #        'verbose': (
        #            ['--verbose', '-v'],
        #            {'action': 'count'}),
        #        'version': (
        #            ['--version'],
        #            {'action': 'version',
        #             'version': improver.version_message() }),
    }

    # We can override including these, but options common to everything should
    # be in a list here:
    # DEFAULT_CENTRALIZED_ARGUMENTS = ('input_file', 'output_file')
    DEFAULT_CENTRALIZED_ARGUMENTS = []

    def __init__(self, central_arguments=DEFAULT_CENTRALIZED_ARGUMENTS,
                 specific_arguments=None, **kwargs):
        """Create an ArgParse instance, which is a subclass of
        argparse.ArgumentParser and automatically add all of the arguments.
        (Note: The ArgParse.COMPULSORY_ARGUMENTS are always added.)

        Args:
            central_arguments (list):
              A list containing the centralized arguments we require.
              (Keys of the centralized argument dictionary). By default this is
              set as ArgParse.DEFAULT_CENTRALIZED_ARGUMENTS.
            specfic_arguments (list):
              A list of argument specifications required to add arguments
              which are not contained within the centralized argument
              dictionary. The format of these argument specifications should
              be the same as the values in the centralized argument dictionary.
              (For more details, see the add_arguments method).
              Default is None, which does not add additional arguments.
            kwargs (dictionary):
              Additional keyword arguments which are passed to the superclass
              constructor (argparse.ArgumentParser).
        """

        self._args = None

        # Allow either central_arguments or specific_arguments to be None
        # (or empty lists)
        if central_arguments is None:
            central_arguments = []
        if specific_arguments is None:
            specific_arguments = []

        # argspecs of the compulsory arguments (no switch here)
        compulsory_arguments = ArgParser.COMPULSORY_ARGUMENTS.values()

        # get argspecs of the central arguments
        central_arguments = self._args_from_selected_central_args(
                             central_arguments)

        # specific arguments must be passed in with the correct format
        # (argspecs)- we don't need to do anything special here...

        # create instance of ArgumentParser (pass along kwargs)
        super(self.__class__, self).__init__(**kwargs)

        cli_arguments = (compulsory_arguments + central_arguments +
                         specific_arguments)

        # automatically add all of the arguments
        self.add_arguments(cli_arguments)
        # Done. Now we can get the arguments with self.args()

    @staticmethod
    def _args_from_selected_central_args(arg_name_list):
        """Return a list of argument specifications from the
        ArgParser.CENTRALIZED_ARGUMENTS, given a list of keys (argument names).

        Args:
            arg_name_list (list):
              A list containing the keys required from the centralized argument
              dictionary.

        Returns:
            list:
              The argument specifications which may later be used to add
              arguments to the parser.
        """
        return [ArgParser.CENTRALIZED_ARGUMENTS[arg_name] for arg_name
                in arg_name_list]

    def add_arguments(self, argspec_list):
        """Adds a list of arguments to the ArgumentParser.

        The input argspec_list is a list of argument specifications, where each
        element (argument specification) is a tuple/list of length 1 or 2.
        The first element of an argument specification is a list of strings
        which the name/flags used to add the argument.
        (Optionally) the second element of the argument spec shall be a
        dictionary containing the keyword arguments which are passed into the
        add_arguments() method.

        Args:
            argspec_list (list):
              A list containing the specifications required to add the
              arguments (see above)

        Raises:
            AttributeError:
                Notifies the user if any of the argument specifications has
                the wrong length (not 1 or 2).
        """
        for argspec in argspec_list:
            # each should be a list/tuple of length 1 or 2 (no more):
            if len(argspec) == 1:
                argflags, argkwargs = (argspec[0], {})
            elif len(argspec) == 2:
                argflags, argkwargs = argspec
            else:
                # AttributeError most appropriate?
                # can't assign to argflags/argkwargs
                raise AttributeError("The argument specification has an "
                                     "unexpected length.")
            self.add_argument(*argflags, **argkwargs)

    def args(self):
        """Returns the arguments passed in through the command line.

        Automatically calls the parse_args method to return the arguments.
        This means we do not necessarily need to call this manually in order
        access the arguments.

        Returns:
            self._args (argparse.Namespace):
                A namespace containing the arguments that were passed in
                though the command line.
        """
        if self._args is None:
            # we only need to call parse_args() once
            self._args = self.parse_args()
        return self._args

    def wrong_args_error(self, args, method):
        """Raise a parser error.

        Some CLI scripts have multiple methods of carrying out an action, with
        each method having different arguments. This method provides a
        standard error to be used when incompatible method-argument
        combinations are passed in.

        Args:
            args (string):
              The incompatible arguments
            method (string):
              The method with which the arguments are incompatible

        Raises:
            parser.error:
                To notify user of incompatible method-argument
                combinations.
        """
        msg = 'Method: {} does not accept arguments: {}'
        self.error(msg.format(method, args))

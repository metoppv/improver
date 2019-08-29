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
"""Common option utilities for improver CLIs."""

import pathlib
import sys
from argparse import ArgumentParser

from improver.profile import profile_hook_enable


class ArgParser(ArgumentParser):
    """Argument parser for improver CLIs.

    The main purpose of this class is to make it easier to create CLIs which
    have arguments which are selected from centralized collections.

    To fulfil these requirements, we define 2 class level dictionaries,
    ArgParser.CENTRALIZED_ARGUMENTS, and ArgParser.COMPULSORY_ARGUMENTS.

    For these dictionaries, each element has:
        - a key, which is a string representing the argument name - used
          internally to refer to a particular argument (which, in the case of
          the CENTRALIZED_ARGUMENTS may be selected from when creating an
          instance of the ArgParser)
        - a value, which is a list containing 2 elements:
            1. a list of strings containing the different flags which are
               associated with the argument (ie.: the first argument to the
               add_arguments() method, e.g: ['--profile', '-p'])
            2. a dictionary containing all of the kwargs which are passed
               to the add_argument() method (e.g:
               {'action': 'store_true', 'default': False, 'help': ... })

    The CENTRALIZED_ARGUMENTS will be selected from, as necessary, for each
    of the CLIs that we create, and the COMPULSORY_ARGUMENTS will be
    automatically added to all CLIs (with no option to exclude them).

    ArgParser.DEFAULT_CENTRALIZED_ARG_NAMES defines the centralized arguments
    which are to be included by default when creating instances of this
    class (i.e: when nothing is explicitly passed
    into the constructor). This is a tuple containing keys associated with the
    ArgParser.CENTRALIZED_ARGUMENTS dictionary.
    """

    # Ideally, all CLIs should select something from this dictionary:
    # NB: --help included by default with ArgumentParser
    CENTRALIZED_ARGUMENTS = {
        'input_file': (
            ['input_filepath'],
            {'metavar': 'INPUT_FILE',
             'help': 'A path to an input NetCDF file to be processed'}),
        'output_file': (
            ['output_filepath'],
            {'metavar': 'OUTPUT_FILE',
             'help': 'The output path for the processed NetCDF'}),
    }

    # *All* CLIs will use the options here (no option to disable them):
    COMPULSORY_ARGUMENTS = {
        'profile': (
            ['--profile'],
            {'action': 'store_true',
             'help': 'Switch on profiling information.'}),
        'profile_file': (
            ['--profile_file'],
            {'metavar': 'PROFILE_FILE',
             'help': 'Dump profiling info to a file. Implies --profile.'})
    }

    # We can override including these, but options common to everything should
    # be in a list here:
    # DEFAULT_CENTRALIZED_ARG_NAMES = ('input_file', 'output_file')
    DEFAULT_CENTRALIZED_ARG_NAMES = ()

    def __init__(self, central_arguments=DEFAULT_CENTRALIZED_ARG_NAMES,
                 specific_arguments=None, **kwargs):
        """Create an ArgParse instance, which is a subclass of
        argparse.ArgumentParser and automatically add all of the arguments.
        (Note: The ArgParse.COMPULSORY_ARGUMENTS are always added.)

        Args:
            central_arguments (list):
                A list containing the centralized arguments we require.
                (Keys of the centralized argument dictionary). By default this
                is set as ArgParse.DEFAULT_CENTRALIZED_ARG_NAMES.
            specific_arguments (list):
                A list of argument specifications required to add arguments
                which are not contained within the centralized argument
                dictionary. The format of these argument specifications should
                be the same as the values in the
                ArgParser.CENTRALIZED_ARGUMENTS dictionary.
                (For more details, see the add_arguments method).
                Default is None, which does not add additional arguments.
            **kwargs:
                Additional keyword arguments which are passed to the superclass
                constructor (argparse.ArgumentParser), e.g: the `description`
                of the ArgumentParser.
        """

        # Allow either central_arguments or specific_arguments to be None
        # (or empty lists)
        if central_arguments is None:
            central_arguments = []
        if specific_arguments is None:
            specific_arguments = []

        # argspecs of the compulsory arguments (no switch here)
        compulsory_arguments = list(ArgParser.COMPULSORY_ARGUMENTS.values())

        # get argspecs of the central arguments from the list of keys passed in
        central_arguments = [ArgParser.CENTRALIZED_ARGUMENTS[arg_name] for
                             arg_name in central_arguments]

        if 'prog' not in kwargs:
            improver_oper = pathlib.PurePosixPath(
                sys.argv[0]).stem.replace("_", "-")
            kwargs['prog'] = ("improver " + improver_oper)

        # create instance of ArgumentParser (pass along kwargs)
        super(ArgParser, self).__init__(**kwargs)

        # all arguments
        cli_arguments = (compulsory_arguments + central_arguments +
                         specific_arguments)

        # automatically add all of the arguments
        self.add_arguments(cli_arguments)
        # Done. Now we can get the arguments with self.parse_args()

    def add_arguments(self, argspec_list):
        """Adds a list of arguments to the ArgumentParser.

        The input argspec_list is a list of argument specifications, where each
        element (argument specification) is a tuple/list of length 2.
        The first element of an argument specification is a list of strings
        which the name/flags used to add the argument.
        The second element of the argument spec shall be a dictionary
        containing the keyword arguments which are passed into the
        add_argument() method.

        Args:
            argspec_list (list):
                A list containing the specifications required to add the
                arguments (see above)

        Raises:
            AttributeError:
                Notifies the user if any of the argument specifications has
                the wrong length (not 2).
        """
        for argspec in argspec_list:
            if len(argspec) != 2:
                raise AttributeError(
                    "The argument specification has an unexpected length. "
                    "Each argument specification should be a 2-tuple, of a "
                    "list (of strings) and a dictionary.")
            argflags, argkwargs = argspec
            self.add_argument(*argflags, **argkwargs)

    def parse_args(self, args=None, namespace=None):
        """Wrap in order to implement some compulsory behaviour."""
        args = super(ArgParser, self).parse_args(args=args,
                                                 namespace=namespace)
        if hasattr(args, 'profile') and (args.profile or args.profile_file):
            profile_hook_enable(dump_filename=args.profile_file)
        return args

    def wrong_args_error(self, args, method):
        """Raise a parser error.

        Some CLI scripts have multiple methods of carrying out an action, with
        each method having different arguments. This method provides a
        standard error to be used when incompatible method-argument
        combinations are passed in - ie: when there are mutually exclusive
        groups of arguments.

        Args:
            args (str):
                The incompatible arguments
            method (str):
                The method with which the arguments are incompatible

        Raises:
            parser.error:
                To notify user of incompatible method-argument
                combinations.
        """
        msg = 'Method: {} does not accept arguments: {}'
        self.error(msg.format(method, args))


def safe_eval(command, module, allowed):
    """
    A wrapper for the python eval() function that enforces the use of a list of
    allowable commands and excludes python builtin functions. This enables the
    use of an eval statement to convert user string input into a function or
    method without it being readily possible to trigger malicious code.

    Args:
        command (str):
            A string identifying the function/method/object that is to be
            returned from the provided module.
        module (module):
            The python module from within which the function/method/object is
            to be found.
        allowed (list):
            A list of the functions/methods/objects that the user is allowed to
            request.
    Returns:
        function/method/object:
            The desired function, method, or object.
    Raises:
        TypeError if the requested module component is not allowed or does not
        exist.
    """
    no_builtins = {"__builtins__": None}
    safe_dict = {k: module.__dict__.get(k, None) for k in allowed}

    try:
        result = eval('{}'.format(command), no_builtins, safe_dict)
    except TypeError:
        raise TypeError(
            'Function/method/object "{}" not available in module {}.'.format(
                command, module.__name__))
    return result

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
"""init for cli and clize"""

import pathlib
import shlex
from collections import OrderedDict
from functools import partial

import clize
from clize import parameters
from clize.help import ClizeHelp, HelpForAutodetectedDocstring
from clize.parser import value_converter
from clize.runner import Clize
from sigtools.wrappers import decorator

# Imports are done in their functions to make calls to -h quicker.
# selected clize imports/constants

IGNORE = clize.Parameter.IGNORE
LAST_OPTION = clize.Parameter.LAST_OPTION
REQUIRED = clize.Parameter.REQUIRED
UNDOCUMENTED = clize.Parameter.UNDOCUMENTED


# help helpers


def docutilize(obj):
    """Convert Numpy or Google style docstring into reStructuredText format.

    Args:
        obj (str or object):
            Takes an object and changes it's docstrings to a reStructuredText
            format.
    Returns:
        str or object:
            A converted string or an object with replaced docstring depending
            on the type of the input.
    """
    from inspect import cleandoc, getdoc
    from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring
    if isinstance(obj, str):
        doc = cleandoc(obj)
    else:
        doc = getdoc(obj)
    doc = str(NumpyDocstring(doc))
    doc = str(GoogleDocstring(doc))
    doc = doc.replace(':exc:', '')
    doc = doc.replace(':data:', '')
    doc = doc.replace(':keyword', ':param')
    doc = doc.replace(':kwtype', ':type')

    if isinstance(obj, str):
        return doc
    obj.__doc__ = doc
    return obj


class HelpForNapoleonDocstring(HelpForAutodetectedDocstring):
    """Subclass to add support for google style docstrings"""
    def add_docstring(self, docstring, *args, **kwargs):
        """Adds the updated docstring."""
        docstring = docutilize(docstring)
        super().add_docstring(docstring, *args, **kwargs)


class DocutilizeClizeHelp(ClizeHelp):
    """Subclass to build Napoleon docstring from subject."""
    def __init__(self, subject, owner,
                 builder=HelpForNapoleonDocstring.from_subject):
        super().__init__(subject, owner, builder)


# input handling


class ObjectAsStr(str):
    """Hide object under a string to pass it through Clize parser."""
    __slots__ = ('original_object',)

    def __new__(cls, obj, name=None):
        if isinstance(obj, cls):  # pass object through if already wrapped
            return obj
        if name is None:
            name = cls.obj_to_name(obj)
        self = str.__new__(cls, name)
        self.original_object = obj
        return self

    @staticmethod
    def obj_to_name(obj, cls=None):
        """Helper function to create the string."""
        if cls is None:
            cls = type(obj)
        try:
            obj_id = hash(obj)
        except TypeError:
            obj_id = id(obj)
        return '<%s.%s@%i>' % (cls.__module__, cls.__name__, obj_id)


def maybe_coerce_with(converter, obj, **kwargs):
    """Apply converter if str, pass through otherwise."""
    obj = getattr(obj, 'original_object', obj)
    return converter(obj, **kwargs) if isinstance(obj, str) else obj


@value_converter
def inputcube(to_convert):
    """Loads cube from file or returns passed object.

    Args:
        to_convert (string or iris.cube.Cube):
            File name or Cube object.

    Returns:
        Loaded cube or passed object.

    """
    from improver.utilities.load import load_cube
    return maybe_coerce_with(load_cube, to_convert)


@value_converter
def inputjson(to_convert):
    """Loads json from file or returns passed object.

    Args:
        to_convert (string or dict):
            File name or json dictionary.

    Returns:
        Loaded json dictionary or passed object.

    """
    from improver.utilities.cli_utilities import load_json_or_none
    return maybe_coerce_with(load_json_or_none, to_convert)


@value_converter
def comma_separated_list(to_convert):
    """Converts comma separated string to list or returns passed object.

    Args:
        to_convert (string or list)
            comma separated string or list

    Returns:
       list
    """
    return maybe_coerce_with(lambda s: s.split(','), to_convert)


@value_converter
def inputpath(to_convert):
    """Converts string paths to pathlib Path objects

    Args:
        to_convert (string or pathlib.Path):
            path represented as string

    Returns:
        (pathlib.Path): Path object

    """
    return maybe_coerce_with(pathlib.Path, to_convert)


def create_constrained_inputcubelist_converter(*constraints):
    """Makes function that the input constraints are used in a loop.

    The function is value_converter, this means it is used by clize to convert
    strings into objects.
    This is a way of not using the IMPROVER load_cube which will try to merge
    cubes. Iris load on the other hand won't deal with meta data properly.
    So an example if you wanted to load an X cube and a Y cube from a cubelist
    of 2. You call this function with a list of constraints.
    These cubes get loaded and returned as a CubeList.

    Args:
        *constraints (tuple of str or list):
            Constraints to be used in the loading of cubes into a CubeList.
            If the tuple contains a string or multiple strings, then each
            string is expected to return exactly one match. If the tuple
            contains a list or multiple lists, then each list is treated as a
            group that is expected to return a single match. The tuple
            can contain a mixture of strings and lists as required.

    Returns:
        function:
            A function with the constraints used for a list comprehension.

    """
    @value_converter
    def constrained_inputcubelist_converter(to_convert):
        """Passes the cube and constraints onto maybe_coerce_with.

        Args:
            to_convert (string):
                The filename to be loaded.

        Returns:
            iris.cube.CubeList:
                The loaded cubelist of constrained cubes.

        Raises:
            ValueError:
                Each constraint (either a string or a list) is expected to
                return a single match. An error is raised if no match or more
                than one match is found.
        """
        from improver.utilities.load import load_cube
        from iris.cube import CubeList

        cubelist = CubeList()
        for constr in constraints:
            constr_list = [constr] if isinstance(constr, str) else constr
            found_cubes = []
            for constr_item in constr_list:
                try:
                    found_cubes.append(maybe_coerce_with(
                        load_cube, to_convert, constraints=constr_item))
                except ValueError:
                    pass
            if len(found_cubes) != 1:
                msg = (f"Incorrect number of valid inputs available for the "
                       "{constr} constraint. "
                       f"Number of valid inputs: {len(found_cubes)} "
                       f"The valid inputs found are: {found_cubes}")
                raise ValueError(msg)
            cubelist.extend(found_cubes)

        return cubelist

    return constrained_inputcubelist_converter

# output handling


@decorator
def with_output(wrapped, *args, output=None, **kwargs):
    """Add `output` keyword only argument.

    This is used to add an extra `output` CLI option. If provided, it saves
    the result of calling `wrapped` to file and returns None, otherwise it
    returns the result.

    Args:
        wrapped (obj):
            The function to be wrapped.
        output (str, optional):
            Output file name. If not supplied, the output object will be
            printed instead.

    Returns:
        Result of calling `wrapped` or None if `output` is given.
    """
    from improver.utilities.save import save_netcdf
    result = wrapped(*args, **kwargs)
    if output:
        save_netcdf(result, output)
        return
    return result


@decorator
def with_intermediate_output(wrapped, *args, intermediate_output=None,
                             **kwargs):
    """Add `intermediate_output` keyword only argument.

    Args:
        wrapped (obj):
            The function to be wrapped.
        intermediate_output (str, optional):
            Output file name for intermediate result.
    """

    from improver.utilities.save import save_netcdf
    result, intermediate_result = wrapped(*args, **kwargs)
    if intermediate_output:
        save_netcdf(intermediate_result, intermediate_output)
    return result


# cli object creation


def clizefy(obj=None, helper_class=DocutilizeClizeHelp, **kwargs):
    """Decorator for creating CLI objects.
    """
    if obj is None:
        return partial(clizefy, helper_class=helper_class, **kwargs)
    if hasattr(obj, 'cli'):
        return obj
    if not callable(obj):
        return Clize.get_cli(obj, **kwargs)
    return Clize.keep(obj, helper_class=helper_class, **kwargs)


# help command


@clizefy(help_names=())
def improver_help(prog_name: parameters.pass_name,
                  command=None, *, usage=False):
    """Show command help."""
    prog_name = prog_name.split()[0]
    args = filter(None, [command, '--help', usage and '--usage'])
    result = execute_command(SUBCOMMANDS_DISPATCHER, prog_name, *args)
    if not command and usage:
        result = '\n'.join(line for line in result.splitlines()
                           if not line.endswith('--help [--usage]'))
    return result


def _cli_items():
    """Dynamically discover CLIs."""
    import importlib
    import pkgutil
    from improver.cli import __path__ as improver_cli_pkg_path
    yield ('help', improver_help)
    for minfo in pkgutil.iter_modules(improver_cli_pkg_path):
        mod_name = minfo.name
        if mod_name != '__main__':
            mcli = importlib.import_module('improver.cli.' + mod_name)
            yield (mod_name, clizefy(mcli.process))


SUBCOMMANDS_TABLE = OrderedDict(sorted(_cli_items()))


# main CLI object with subcommands


SUBCOMMANDS_DISPATCHER = clizefy(
    SUBCOMMANDS_TABLE,
    description="""IMPROVER NWP post-processing toolbox""",
    footnotes="""See also improver --help for more information.""")


# IMPROVER top level main


def unbracket(args):
    """Convert input list with bracketed items into nested lists.

    >>> unbracket('foo [ bar a b ] [ baz c ] -o z'.split())
    ['foo', ['bar', 'a', 'b'], ['baz', 'c'], '-o', 'z']

    """
    outargs = []
    stack = []
    mismatch_msg = 'Mismatched bracket at position %i.'
    for i, arg in enumerate(args):
        if arg == '[':
            stack.append(outargs)
            outargs = []
        elif arg == ']':
            if not stack:
                raise ValueError(mismatch_msg % i)
            stack[-1].append(outargs)
            outargs = stack.pop()
        else:
            outargs.append(arg)
    if stack:
        raise ValueError(mismatch_msg % len(args))
    return outargs


def execute_command(dispatcher, prog_name, *args,
                    verbose=False, dry_run=False):
    """Common entry point for command execution."""
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (list, tuple)):
            # process nested commands recursively
            arg = execute_command(dispatcher, prog_name, *arg,
                                  verbose=verbose, dry_run=dry_run)
        if isinstance(arg, pathlib.PurePath):
            arg = str(arg)
        elif not isinstance(arg, str):
            arg = ObjectAsStr(arg)
        args[i] = arg

    if verbose or dry_run:
        print(" ".join([shlex.quote(x) for x in (prog_name, *args)]))
    if dry_run:
        return args

    result = dispatcher(prog_name, *args)

    if verbose and result is not None:
        print(ObjectAsStr.obj_to_name(result))
    return result


@clizefy()
def main(prog_name: parameters.pass_name,
         command: LAST_OPTION,
         *args,
         profile: value_converter(lambda _: _, name='FILENAME') = None,
         memprofile: value_converter(lambda _: _, name='FILENAME') = None,
         verbose=False,
         dry_run=False):
    """IMPROVER NWP post-processing toolbox

    Results from commands can be passed into file-like arguments
    of other commands by surrounding them by square brackets::

        improver command [ command ... ] ...

    Spaces around brackets are mandatory.

    Args:
        prog_name:
            The program name from argv[0].
        command (str):
            Command to execute
        args (tuple):
            Command arguments
        profile (str):
            If given, will write profiling to the file given.
            To write to stdout, use a hyphen (-)
        memprofile (str):
            Creates 2 files, a tracemalloc snapsot at the point of
            highest memory consumption of your program (*_SNAPSHOT)
            and a track of the maximum memory used by your program
            over time (*_MAX_TRACKER).
        verbose (bool):
            Print executed commands
        dry_run (bool):
            Print commands to be executed

    See improver help [--usage] [command] for more information
    on available command(s).
    """
    args = unbracket(args)
    exec_cmd = execute_command
    if profile is not None:
        from improver.profile import profile_hook_enable
        profile_hook_enable(dump_filename=None if profile == '-' else profile)
    if memprofile is not None:
        from improver.memprofile import memory_profile_decorator
        exec_cmd = memory_profile_decorator(exec_cmd, memprofile)
    result = exec_cmd(SUBCOMMANDS_DISPATCHER, prog_name, command,
                      *args, verbose=verbose, dry_run=dry_run)
    return result


def run_main(argv=None):
    """Overrides argv[0] to be 'improver' then runs main.

    Args:
        argv (list of str):
            Arguments that were from the command line.

    """
    from clize import run
    import sys
    # clize help shows module execution as `python -m improver.cli`
    # override argv[0] and pass it explicitly in order to avoid this
    # so that the help command reflects the way that we call improver.
    if argv is None:
        argv = sys.argv[:]
        argv[0] = 'improver'
    run(main, args=argv)  # pylint: disable=E1124

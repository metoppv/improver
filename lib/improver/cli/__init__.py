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
"init for cli and clize"

from clize import (
    Clize,
    Parameter,
    run,
)
from clize.help import (
    HelpForAutodetectedDocstring,
    ClizeHelp,
)
from clize.parameters import pass_name as pass_program_name
from clize.parser import value_converter
from sigtools.wrappers import decorator

_clizefy = Clize.keep
#get_cli = Clize.get_cli

def get_cli(obj, **kwargs):
    # TODO: set get_cli = Clize.get_cli after all CLIs are clizefied
    import os

    if callable(obj) and (not os.environ.get('IMPROVER_USE_CLIZE')
            or not hasattr(obj, 'cli')):
        import sys
        impr_main = sys.modules[obj.__module__].main
        description = obj.__doc__.split('\n')[0].strip()

        def clized_main(prog: pass_program_name, *args):
            sys.argv[0] = prog.split()[-1]
            impr_main(args)

        return Clize.as_is(clized_main, description=description, **kwargs)

    return Clize.get_cli(obj, **kwargs)

# def is_using_clize():
#     #return True
#     # TODO: remove this function after transition to Clize is completed
#     try:
#         import clize
#         import os
#         return os.environ.get('IMPROVER_USE_CLIZE', True)
#     except ImportError:
#         return False
#     return False
#
#
# def identity(func, *args, **kwawrgs):
#     return func
#
#
# if is_using_clize():
#
#     from sigtools.wrappers import decorator
#     from clize import (
#         Clize,
#         Parameter,
#         run,
#     )
#     from clize.help import (
#         HelpForAutodetectedDocstring,
#         ClizeHelp,
#     )
#     from clize.parameters import pass_name as pass_program_name
#     from clize.parser import value_converter
#     _clizefy = Clize.keep
#     get_cli = Clize.get_cli
#
# else:
#
#     def decorator(outer):
#         from functools import wraps
#
#         @wraps(outer)
#         def outer_wrapper(inner):
#
#             @wraps(inner)
#             def inner_wrapper(*args, **kwargs):
#                 return outer(inner, *args, **kwargs)
#
#             return inner_wrapper
#
#         return outer_wrapper
#
#     def run(cli_func):
#         import sys
#         return cli_func(*sys.argv)
#
#     _clizefy = value_converter = identity
#     HelpForAutodetectedDocstring = ClizeHelp = object


# help helpers


def docutilize(obj):
    """

    Args:
        obj (str or obj):
            Takes an object and changes it's docstrings to a suitable format
            for clize.
    Returns:
        (str or obj):
            A string with replaced docstrings. or an altered string depending
            on the format of the input.
    """
    from inspect import cleandoc
    from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring
    if isinstance(obj, str):
        doc = obj
    else:
        doc = obj.__doc__
    doc = cleandoc(doc)
    doc = str(NumpyDocstring(doc))
    doc = str(GoogleDocstring(doc))
    # exception and keyword markup seems to trip up the docutils parser
    doc = doc.replace(':exc:', '')
    doc = doc.replace(':keyword', ':param')
    doc = doc.replace(':kwtype', ':type')
    if isinstance(obj, str):
        return doc
    obj.__doc__ = doc
    return obj


class HelpForNapoleonDocstring(HelpForAutodetectedDocstring):
    """Help for Napolean Docstrings."""
    def add_docstring(self, docstring, *args, **kwargs):
        """Adds the updated docstring."""
        docstring = docutilize(docstring)
        super().add_docstring(docstring, *args, **kwargs)


class DocutilizeClizeHelp(ClizeHelp):
    """Subclass to build Napoleon docstring from subject."""
    def __init__(self, subject, owner,
                 builder=HelpForNapoleonDocstring.from_subject):
        super().__init__(subject, owner, builder)


class ObjectAsStr(str):
    """Hide object under a string to pass it through Clize parser."""
    __slots__ = ('original_object',)

    def __new__(cls, obj, name=None):
        if isinstance(obj, cls):  # pass object through if already wrapped
            return obj
        if name is None:
            name = cls.object2name(obj)
        self = str.__new__(cls, name)
        self.original_object = obj
        return self

    @staticmethod
    def object2name(obj, cls=None):
        if cls is None:
            cls = type(obj)
        try:
            obj_id = hash(obj)
        except TypeError:
            obj_id = id(obj)
        return '<%s.%s@%i>' % (cls.__module__, cls.__name__, obj_id)


def maybe_coerce_with(convert, obj, **kwargs):
    """Apply converter if str, pass through otherwise."""
    obj = getattr(obj, 'original_object', obj)
    return convert(obj, **kwargs) if isinstance(obj, str) else obj


@value_converter
def inputcube(to_convert):
    """

    Args:
        to_convert (string or obj):
            calls maybe_coerce_with function with the input and load_cube.

    Returns:
        (obj):
            The result of maybe_coerce_with.

    """
    from improver.utilities.load import load_cube
    return maybe_coerce_with(load_cube, to_convert)


@value_converter
def inputjson(to_convert):
    """

    Args:
        to_convert (string or obj):
            calls maybe_coerce_with function with the input and
            load_json_or_none.

    Returns:
        (obj):
            The result of maybe_coerce_with.

    """
    from improver.utilities.cli_utilities import load_json_or_none
    return maybe_coerce_with(load_json_or_none, to_convert)


# output handling


# def save_output(saver, result, outfile, *, index=None):
#     """Helper function to pop and save one value out of multiple returned."""
#     if outfile is None:
#         return result
#     if index is None:
#         return saver(result, outfile)
#     saver(result[index], outfile)
#     # remove saved value(s) from returned result
#     idx_slice = (index if isinstance(index, slice) else
#                  slice(index, index + 1 or None))
#     idx_range = range(*idx_slice.indices(len(result)))
#     rtype = type(result)
#     result = rtype(v for i, v in enumerate(result) if i not in idx_range)
#     return result[0] if len(result) == 1 else result or None


@decorator
def with_output(wrapped, *args, output=None, **kwargs):
    """
    :param output: Output file name
    """
    from improver.utilities.save import save_netcdf
    result = wrapped(*args, **kwargs)
    if output:
        return save_netcdf(result, output)
    return result


@decorator
def with_intermediate_output(wrapped, *args, output=None,
                             intermediate_output=None, **kwargs):
    """
    :param intermediate_output: Output file name for intermediate result
    :param output: Output file name
    """
    from improver.utilities.save import save_netcdf
    result, intermediate = wrapped(*args, **kwargs)
    returns = ()
    if output:
        save_netcdf(result, output)
    else:
        returns += (result,)
    if intermediate_output:
        save_netcdf(intermediate, intermediate_output)
    else:
        returns += (intermediate,)
    returns = tuple(filter(None, returns))
    returns = returns[0] if len(returns) == 1 else returns or None
    return returns


# cli object creation and handling


def clizefy(func=None, with_output=with_output,
            helper_class=DocutilizeClizeHelp, **kwargs):
    """Decorator for creating CLI objects."""
    from functools import partial
    if func is None:
        return partial(clizefy, with_output=with_output,
                       helper_class=helper_class, **kwargs)
    if with_output:
        func = with_output(func)
    func = _clizefy(func, helper_class=helper_class, **kwargs)
    return func


def unbracket(args):
    """Convert input list with bracketed items into nested lists.

    >>> unbracket('foo [ bar a b ] [ baz c ] -o z'.split())
    ['foo', ['bar', 'a', 'b'], ['baz', 'c'], '-o', 'z']
    """
    outargs = []
    stack = []
    mismatch_msg = 'Mismatched bracket at position %i.'
    for i in range(0, len(args)):
        if args[i] == '[':
            stack.append(outargs)
            outargs = []
        elif args[i] == ']':
            if not stack:
                raise ValueError(mismatch_msg % i)
            stack[-1].append(outargs)
            outargs = stack.pop()
        else:
            outargs.append(args[i])
    if stack:
        raise ValueError(mismatch_msg % len(args))
    return outargs


# process nested commands recursively
def execute_command(dispatcher, progname, *args, verbose=False, dry_run=False):
    """Common entry point for command execution."""
    args = unbracket(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (list, tuple)):
            arg = execute_command(dispatcher, progname, *arg,
                                  verbose=verbose, dry_run=dry_run)
        if not isinstance(arg, str):
            arg = ObjectAsStr(arg)
        args[i] = arg
    if dry_run:
        result = args  # poor man's dry run!
    else:
        result = dispatcher(progname, *args)
    if verbose:
        print(progname, *args, ' -> ', ObjectAsStr.object2name(result))
    return result

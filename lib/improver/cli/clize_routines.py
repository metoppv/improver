#!/usr/bin/env python
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
"""Routines for clize to run."""
import clize
import clize.help
import sigtools.wrappers

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


class HelpForNapoleonDocstring(clize.help.HelpForAutodetectedDocstring):
    """Help for Napolean Docstrings."""
    def add_docstring(self, docstring, *args, **kwargs):
        """Adds the updated docstring."""
        docstring = docutilize(docstring)
        super().add_docstring(docstring, *args, **kwargs)


class DocutilizeClizeHelp(clize.help.ClizeHelp):
    """Subclass to build Napoleon docstring from subject."""
    def __init__(self, subject, owner,
                 builder=HelpForNapoleonDocstring.from_subject):
        super().__init__(subject, owner, builder)

# converters


def maybe_coerce_with(convert, obj, **kwargs):
    """Apply converter if str, pass through otherwise."""
    return convert(obj, **kwargs) if isinstance(obj, str) else obj


@clize.parser.value_converter
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


@clize.parser.value_converter
def optionalcube(to_convert):
    """

    Args:
        to_convert (string or obj):
            calls maybe_coerce_with function with the input and load_cube.

    Returns:
        (obj):
            The result of maybe_coerce_with.

    """
    from improver.utilities.load import load_cube
    return maybe_coerce_with(load_cube, to_convert, **{'allow_none': True})


@clize.parser.value_converter
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


def outputcube(cube, output):
    """Save Cube. Used as annotation of return values."""
    from improver.utilities.save import save_netcdf
    if output:
        save_netcdf(cube, output)
    return cube


def save_at_index(index, outfile, func, *args, **kwargs):
    """Helper function to save one value out of multiple returned."""
    result = result_selection = func(*args, **kwargs)
    saver = func.__annotations__.get('return')
    if not outfile or not saver:
        return result
    if isinstance(saver, tuple):
        saver = saver[index]
        result_selection = result[index]
    elif index:
        raise IndexError('Non-zero index is valid only for multiple '
                         'return values.')
    if not isinstance(index, slice):
        result_selection, outfile = (result_selection,), (outfile,)
    if len(result_selection) != len(outfile):
        raise ValueError('Number of selected results does not match '
                         'return annotation.')
    for res, out in zip(result_selection, outfile):
        saver(res, out)
    return result


@sigtools.wrappers.decorator
def with_output(wrapped, *args, output=None, **kwargs):
    """
    :param output: Output file name
    """
    return save_at_index(0, output, wrapped, *args, **kwargs)


@sigtools.wrappers.decorator
def with_intermediate_output(wrapped, *args, intermediate_output=None,
                             **kwargs):
    """
    :param intermediate_output: Output file name for intermediate result
    """
    return save_at_index(1, intermediate_output, wrapped, *args, **kwargs)

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
    func = clize.Clize.keep(func, helper_class=helper_class, **kwargs)
    return func

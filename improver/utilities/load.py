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
"""Module for loading cubes."""

import contextlib
import glob

import iris

from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering, merge_cubes)


@contextlib.contextmanager
def iris_nimrod_patcher():
    """Temporarily monkey patches iris.fileformats.nimrod* modules."""
    # FIXME: monkey patched nimrod loading in iris, so it works for radar files
    if hasattr(iris.fileformats.nimrod_load_rules, 'DEFAULT_UNITS'):
        raise RuntimeError('FIXME: nimrod monkey patch is no longer needed')
    try:
        from iris_nimrod_patch import nimrod, nimrod_load_rules
    except ImportError:
        yield
        return
    else:
        header_attrs = ['general_header_int16s', 'general_header_float32s',
                        'data_header_int16s', 'data_header_float32s']
        attribs = [{attr: getattr(m, attr) for attr in header_attrs}
                   for m in [nimrod, iris.fileformats.nimrod]]
        modules = [nimrod_load_rules, iris.fileformats.nimrod_load_rules]
        for m, d in zip(modules, attribs):
            vars(iris.fileformats.nimrod).update(d)
            iris.fileformats.nimrod_load_rules = m
            if m is nimrod_load_rules:
                yield


def load_cube(filepath, constraints=None, no_lazy_load=False,
              allow_none=False):
    """Load the filepath provided using Iris into a cube.

    Args:
        filepath (str or list):
            Filepath that will be loaded or list of filepaths that can be
            merged into a single cube upon loading.
        constraints (iris.Constraint, str or None):
            Constraint to be applied when loading from the input filepath.
            This can be in the form of an iris.Constraint or could be a string
            that is intended to match the name of the cube.
            The default is None.
        no_lazy_load (bool):
            If True, bypass cube deferred (lazy) loading and load the whole
            cube into memory. This can increase performance at the cost of
            memory. If False (default) then lazy load.
        allow_none (bool):
            If True, when the filepath is None, returns None.
            If False, normal error handling applies.
            Default is False.

    Returns:
        iris.cube.Cube:
            Cube that has been loaded from the input filepath given the
            constraints provided.
    """
    if filepath is None and allow_none:
        return None
    # Remove metadata prefix cube if present
    constraints = iris.Constraint(
        cube_func=lambda cube: cube.long_name != 'prefixes') & constraints

    # Load each file individually to avoid partial merging (not used
    # iris.load_raw() due to issues with time representation)
    with iris_nimrod_patcher():
        if isinstance(filepath, str):
            cubes = iris.load(filepath, constraints=constraints)
        else:
            cubes = iris.cube.CubeList([])
            for item in filepath:
                cubes.extend(iris.load(item, constraints=constraints))

    # Merge loaded cubes
    if not cubes:
        message = "No cubes found using constraints {}".format(constraints)
        raise ValueError(message)

    if len(cubes) == 1:
        cube = cubes[0]
    else:
        cube = merge_cubes(cubes)

    # Remove metadata prefix cube attributes
    if 'bald__isPrefixedBy' in cube.attributes.keys():
        cube.attributes.pop('bald__isPrefixedBy')

    # Ensure the probabilistic coordinates are the first coordinates within a
    # cube and are in the specified order.
    enforce_coordinate_ordering(
        cube, ["realization", "percentile", "threshold"])
    # Ensure the y and x dimensions are the last dimensions within the cube.
    y_name = cube.coord(axis="y").name()
    x_name = cube.coord(axis="x").name()
    enforce_coordinate_ordering(
        cube, [y_name, x_name], anchor_start=False)
    if no_lazy_load:
        # Force the cube's data into memory by touching the .data attribute.
        # pylint: disable=pointless-statement
        cube.data
    return cube


def load_cubelist(filepath, constraints=None, no_lazy_load=False):
    """Load one cube from each of the filepath(s) provided using Iris into
    a cubelist.

    Args:
        filepath (str or list):
            Filepath(s) that will be loaded, each containing a single cube.  If
            wildcarded, this will be expanded into a list of file names so that
            only one cube is loaded per file.
        constraints (iris.Constraint, str or None):
            Constraint to be applied when loading from the input filepath.
            This can be in the form of an iris.Constraint or could be a string
            that is intended to match the name of the cube.
            The default is None.
        no_lazy_load (bool)
            If True, bypass cube deferred (lazy) loading and load the whole
            cube into memory. This can increase performance at the cost of
            memory. If False (default) then lazy load.

    Returns:
        iris.cube.CubeList:
            CubeList that has been created from the input filepath given the
            constraints provided.
    """
    if isinstance(filepath, list) and len(filepath) == 1:
        filepath = filepath[0]

    # If the filepath is a string, then use glob, in case the str contains
    # wildcards.
    if isinstance(filepath, str):
        filepaths = glob.glob(filepath)
    else:
        filepaths = filepath

    # Construct a cubelist using the load_cube function.
    cubelist = iris.cube.CubeList([])
    for filepath in filepaths:
        try:
            cube = load_cube(filepath, constraints=constraints)
        except ValueError:
            continue
        if no_lazy_load:
            # Force the cube's data into memory by touching the .data.
            # pylint: disable=pointless-statement
            cube.data
        cubelist.append(cube)
    return cubelist

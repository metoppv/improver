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
"""Utilities for interrogating IMPROVER probabilistic metadata"""

import re

import iris
from iris.exceptions import CoordinateNotFoundError


def probability_cube_name_regex(cube_name):
    """
    Regular expression matching IMPROVER probability cube name.  Returns
    None if the cube_name does not match the regular expression (ie does
    not start with 'probability_of').

    Args:
        cube_name (str):
            Probability cube name
    """
    regex = re.compile(
        '(probability_of_)'  # always starts this way
        '(?P<diag>.*?)'      # named group for the diagnostic name
        '(_in_vicinity|)'    # optional group, may be empty
        '(?P<thresh>_above_threshold|_below_threshold|_between_thresholds|$)')
    return regex.match(cube_name)


def in_vicinity_name_format(cube_name):
    """Generate the correct name format for an 'in_vicinity' probability
    cube, taking into account the 'above/below_threshold' or
    'between_thresholds' suffix required by convention.

    Args:
        cube_name (str):
            The non-vicinity probability cube name to be formatted.

    Returns:
        str:
            Correctly formatted name following the accepted convention e.g.
            'probability_of_X_in_vicinity_above_threshold'.
    """
    regex = probability_cube_name_regex(cube_name)
    new_cube_name = 'probability_of_{diag}_in_vicinity{thresh}'.format(
        **regex.groupdict())
    return new_cube_name


def extract_diagnostic_name(cube_name):
    """
    Extract the standard or long name X of the diagnostic from a probability
    cube name of the form 'probability_of_X_above/below_threshold',
    'probability_of_X_between_thresholds', or
    'probability_of_X_in_vicinity_above/below_threshold'.

    Args:
        cube_name (str):
            The probability cube name

    Returns:
        str:
            The name of the diagnostic underlying this probability

    Raises:
        ValueError: If the input name does not match the expected regular
            expression (ie if cube_name_regex(cube_name) returns None).
    """
    try:
        diagnostic_name = probability_cube_name_regex(cube_name).group('diag')
    except AttributeError:
        raise ValueError(
            'Input {} is not a valid probability cube name'.format(cube_name))
    return diagnostic_name


def find_threshold_coordinate(cube):
    """Find threshold coordinate in cube.

    Compatible with both the old (cube.coord("threshold")) and new
    (cube.coord.var_name == "threshold") IMPROVER metadata standards.

    Args:
        cube (iris.cube.Cube):
            Cube containing thresholded probability data

    Returns:
        iris.coords.Coord:
            Threshold coordinate

    Raises:
        TypeError: If cube is not of type iris.cube.Cube.
        CoordinateNotFoundError: If no threshold coordinate is found.
    """
    if not isinstance(cube, iris.cube.Cube):
        msg = ('Expecting data to be an instance of '
               'iris.cube.Cube but is {0}.'.format(type(cube)))
        raise TypeError(msg)

    threshold_coord = None
    try:
        threshold_coord = cube.coord("threshold")
    except CoordinateNotFoundError:
        for coord in cube.coords():
            if coord.var_name == "threshold":
                threshold_coord = coord
                break

    if threshold_coord is None:
        msg = ('No threshold coord found on {0:s} data'.format(
               cube.name()))
        raise CoordinateNotFoundError(msg)

    return threshold_coord


def find_percentile_coordinate(cube):
    """Find percentile coord in cube.

    Args:
        cube (iris.cube.Cube):
            Cube contain one or more percentiles.
    Returns:
        iris.coords.Coord:
            Percentile coordinate.
    Raises:
        TypeError: If cube is not of type iris.cube.Cube.
        CoordinateNotFoundError: If no percentile coordinate is found in cube.
        ValueError: If there is more than one percentile coords in the cube.
    """
    if not isinstance(cube, iris.cube.Cube):
        msg = ('Expecting data to be an instance of '
               'iris.cube.Cube but is {0}.'.format(type(cube)))
        raise TypeError(msg)
    standard_name = cube.name()
    perc_coord = None
    perc_found = 0
    for coord in cube.coords():
        if coord.name().find('percentile') >= 0:
            perc_found += 1
            perc_coord = coord

    if perc_found == 0:
        msg = ('No percentile coord found on {0:s} data'.format(
               standard_name))
        raise CoordinateNotFoundError(msg)

    if perc_found > 1:
        msg = ('Too many percentile coords found on {0:s} data'.format(
               standard_name))
        raise ValueError(msg)

    return perc_coord

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
from typing import Match, Optional

import iris
from iris.coords import Coord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError


def probability_cube_name_regex(cube_name: str) -> Optional[Match]:
    """
    Regular expression matching IMPROVER probability cube name.  Returns
    None if the cube_name does not match the regular expression (ie does
    not start with 'probability_of').

    Args:
        cube_name:
            Probability cube name

    Returns:
        The regex match
    """
    regex = re.compile(
        "(probability_of_)"  # always starts this way
        "(?P<diag>.*?)"  # named group for the diagnostic name
        "(?P<vicinity>_in_vicinity|_in_variable_vicinity)?"  # optional group, may be empty
        "(?P<thresh>_above_threshold|_below_threshold|_between_thresholds|$)"
    )
    return regex.match(cube_name)


def in_vicinity_name_format(cube_name: str) -> str:
    """Generate the correct name format for an 'in_vicinity' probability
    cube, taking into account the 'above/below_threshold' or
    'between_thresholds' suffix required by convention.

    Args:
        cube_name:
            The non-vicinity probability cube name to be formatted.

    Returns:
        Correctly formatted name following the accepted convention e.g.
        'probability_of_X_in_vicinity_above_threshold'.
    """
    regex = probability_cube_name_regex(cube_name)
    new_cube_name = "probability_of_{diag}_in_vicinity{thresh}".format(
        **regex.groupdict()
    )
    return new_cube_name


def get_threshold_coord_name_from_probability_name(cube_name: str) -> str:
    """Get the name of the threshold coordinate from the name of the probability
    cube.  This can be used to set or modify a threshold coordinate name after
    renaming or conversion from probabilities to percentiles / realizations."""
    return _extract_diagnostic_name(cube_name)


def get_diagnostic_cube_name_from_probability_name(cube_name: str) -> str:
    """Get the name of the original diagnostic cube, including vicinity, from
    the name of the probability cube."""
    return _extract_diagnostic_name(cube_name, check_vicinity=True)


def _extract_diagnostic_name(cube_name: str, check_vicinity: bool = False) -> str:
    """
    Extract the standard or long name X of the diagnostic from a probability
    cube name of the form 'probability_of_X_above/below_threshold',
    'probability_of_X_between_thresholds', or
    'probability_of_X_in_vicinity_above/below_threshold'.

    Args:
        cube_name:
            The probability cube name
        check_vicinity:
            If False the function will return X as described above, which matches
            the name of the threshold-type coordinate on the cube.  If True, the
            cube name is checked to see whether it is a vicinity diagnostic, and
            if so the function returns "X_in.*_vicinity".  This is the name of the
            equivalent diagnostic in percentile or realization space.

    Returns:
        The name of the diagnostic underlying this probability

    Raises:
        ValueError: If the input name does not match the expected regular
            expression (ie if cube_name_regex(cube_name) returns None).
    """
    cube_name_parts = probability_cube_name_regex(cube_name)
    try:
        diagnostic_name = cube_name_parts.group("diag")
    except AttributeError:
        raise ValueError(
            "Input {} is not a valid probability cube name".format(cube_name)
        )

    vicinity_match = cube_name_parts.group("vicinity")
    if check_vicinity and vicinity_match:
        diagnostic_name += vicinity_match

    return diagnostic_name


def is_probability(cube: Cube) -> bool:
    """Determines whether a cube contains probability data at a range of
    thresholds.

    Args:
        cube:
            Cube to check for probability threshold data.

    Returns:
        True if in threshold representation.
    """
    try:
        find_threshold_coordinate(cube)
    except CoordinateNotFoundError:
        return False
    return True


def find_threshold_coordinate(cube: Cube) -> Coord:
    """Find threshold coordinate in cube.

    Compatible with both the old (cube.coord("threshold")) and new
    (cube.coord.var_name == "threshold") IMPROVER metadata standards.

    Args:
        cube:
            Cube containing thresholded probability data

    Returns:
        Threshold coordinate

    Raises:
        TypeError: If cube is not of type iris.cube.Cube.
        CoordinateNotFoundError: If no threshold coordinate is found.
    """
    if not isinstance(cube, iris.cube.Cube):
        msg = (
            "Expecting data to be an instance of "
            "iris.cube.Cube but is {0}.".format(type(cube))
        )
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
        msg = "No threshold coord found on {0:s} data".format(cube.name())
        raise CoordinateNotFoundError(msg)

    return threshold_coord


def probability_is_above_or_below(cube: Cube) -> Optional[str]:
    """Checks the spp__relative_to_threshold attribute and outputs
    whether it is above or below the threshold given. If there isn't
    a spp__relative_to_threshold attribute it returns None.

    Args:
        cube:
            Cube containing thresholded probability data

    Returns:
        Which indicates whether the cube has data that is
        above or below the threshold
    """

    threshold_attribute = None
    thresh_coord = find_threshold_coordinate(cube)
    thresh = thresh_coord.attributes.get("spp__relative_to_threshold", None)
    if thresh in ("above", "greater_than", "greater_than_or_equal_to"):
        threshold_attribute = "above"
    elif thresh in ("below", "less_than", "less_than_or_equal_to"):
        threshold_attribute = "below"

    return threshold_attribute


def is_percentile(cube: Cube) -> bool:
    """Determines whether a cube contains probability data at a range of
    percentiles.

    Args:
        cube:
            Cube to check for percentile data.

    Returns:
        True if in percentile representation.
    """
    try:
        find_percentile_coordinate(cube)
    except (CoordinateNotFoundError, ValueError):
        return False
    return True


def find_percentile_coordinate(cube: Cube) -> Coord:
    """Find percentile coord in cube.

    Args:
        cube:
            Cube contain one or more percentiles.

    Returns:
        Percentile coordinate.

    Raises:
        TypeError: If cube is not of type iris.cube.Cube.
        CoordinateNotFoundError: If no percentile coordinate is found in cube.
        ValueError: If there is more than one percentile coords in the cube.
    """
    if not isinstance(cube, iris.cube.Cube):
        msg = (
            "Expecting data to be an instance of "
            "iris.cube.Cube but is {0}.".format(type(cube))
        )
        raise TypeError(msg)
    standard_name = cube.name()
    perc_coord = None
    perc_found = 0
    for coord in cube.coords():
        if coord.name().find("percentile") >= 0:
            perc_found += 1
            perc_coord = coord

    if perc_found == 0:
        msg = "No percentile coord found on {0:s} data".format(standard_name)
        raise CoordinateNotFoundError(msg)

    if perc_found > 1:
        msg = "Too many percentile coords found on {0:s} data".format(standard_name)
        raise ValueError(msg)

    return perc_coord


def format_cell_methods_for_probability(cube: Cube, threshold_name: str) -> None:
    """Update cell methods on a diagnostic cube to reflect the fact that the
    data to which they now refer is on a coordinate.  Modifies cube in place.

    Args:
        cube:
            Cube to update
        threshold_name:
            Name of the threshold-type coordinate to which the cell
            method now refers
    """
    cell_methods = []
    for cell_method in cube.cell_methods:
        new_cell_method = iris.coords.CellMethod(
            cell_method.method,
            coords=cell_method.coord_names,
            intervals=cell_method.intervals,
            comments=f"of {threshold_name}",
        )
        cell_methods.append(new_cell_method)
    cube.cell_methods = cell_methods


def format_cell_methods_for_diagnostic(cube: Cube) -> None:
    """Remove reference to threshold-type coordinate from cell method comments that
    were previously on a probability cube.  Modifies cube in place.

    Args:
        cube:
            Cube to update
    """
    cell_methods = []
    for cell_method in cube.cell_methods:
        new_cell_method = iris.coords.CellMethod(
            cell_method.method,
            coords=cell_method.coord_names,
            intervals=cell_method.intervals,
        )
        cell_methods.append(new_cell_method)
    cube.cell_methods = cell_methods

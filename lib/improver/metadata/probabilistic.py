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
        new_cube_name (str):
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
        diagnostic_name (str):
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

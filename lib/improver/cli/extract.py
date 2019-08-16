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
"""Script to extract a subset of input file data, given constraints."""

from improver.argparser import ArgParser
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Invoke data extraction."""

    parser = ArgParser(description='Extracts subset of data from a single '
                       'input file, subject to equality-based constraints.')
    parser.add_argument('input_file', metavar='INPUT_FILE',
                        help="File containing a dataset to extract from.")
    parser.add_argument('output_file', metavar='OUTPUT_FILE',
                        help="File to write the extracted dataset to.")
    parser.add_argument('constraints', metavar='CONSTRAINTS', nargs='+',
                        help='The constraint(s) to be applied.  These must be'
                        ' of the form "key=value", eg "threshold=1".  Scalars'
                        ', boolean and string values are supported.  Comma-'
                        'separated lists (eg "key=[value1,value2]") are '
                        'supported. These comma-separated lists can either '
                        'extract all values specified in the list or '
                        'all values specified within a range e.g. '
                        'key=[value1:value2]. When a range is specified, '
                        'this is inclusive of the endpoints of the range.')
    parser.add_argument('--units', metavar='UNITS', nargs='+', default=None,
                        help='Optional: units of coordinate constraint(s) to '
                        'be applied, for use when the input coordinate '
                        'units are not ideal (eg for float equality). If '
                        'used, this list must match the CONSTRAINTS list in '
                        'order and length (with null values set to None).')
    parser.add_argument('--ignore-failure', action='store_true', default=False,
                        help='Option to ignore constraint match failure and '
                        'return the input cube.')
    args = parser.parse_args(args=argv)

    # Load Cube
    cube = load_cube(args.input_file)

    # Process Cube
    output_cube = process(cube, args.constraints, args.units)

    # Save Cube
    if output_cube is None and args.ignore_failure:
        save_netcdf(cube, args.output_file)
    elif output_cube is None:
        msg = "Constraint(s) could not be matched in input cube"
        raise ValueError(msg)
    else:
        save_netcdf(output_cube, args.output_file)


def process(cube, constraints, units=None):
    """ Extract a subset of a single cube.

    Extracts subset of data from a single cube, subject to equality-based
    constraints.
    Using a set of constraints, extract a sub-cube from the provided cube if it
    is available.

    Args:
        cube (iris.cube.Cube):
            The Cube from which a sub-cube is extracted
        constraints (list):
            The constraint(s) to be applied.  These must be of the form
            "key=value", eg "threshold=1".  Scalars, boolean and string
            values are supported.  Comma-separated lists
            e.g. key=[value1,value2,value3]
            are supported.
            These comma-separated lists can either extract all values
            specified in the list or all values specified within a range
            e.g. key=[value1:value3].
            When a range is specified, this is inclusive of the endpoints of
            the range.
        units (list):
            List of units as strings corresponding to each coordinate in the
            list of constraints. One or more "units" may be None and units may
            only be associated with coordinate constraints.

    Returns:
        (iris.cube.Cube):
            A single cube matching the input constraints or None. If no
            sub-cube is found within the cube that matches the constraints.
    """
    return extract_subcube(cube, constraints, units)


if __name__ == '__main__':
    main()

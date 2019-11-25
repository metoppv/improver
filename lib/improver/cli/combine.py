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
"""Script to combine netcdf data."""

import warnings

from improver import cli


def main(argv=None):
    """Load in arguments for the cube combiner plugin.
    """
    from improver.argparser import ArgParser

    parser = ArgParser(
        description="Combine the input files into a single file using "
                    "the requested operation e.g. + - min max etc.")
    parser.add_argument("input_filenames", metavar="INPUT_FILENAMES",
                        nargs="+", type=str,
                        help="Paths to the input NetCDF files. Each input"
                        " file should be able to be loaded as a single "
                        " iris.cube.Cube instance. The resulting file"
                        " metadata will be based on the first file but"
                        " its metadata can be overwritten via"
                        " the metadata_jsonfile option.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF.")
    parser.add_argument("--operation", metavar="OPERATION",
                        default="+",
                        choices=["+", "-", "*",
                                 "add", "subtract", "multiply",
                                 "min", "max", "mean"],
                        help="Operation to use in combining NetCDF datasets"
                        " Default=+ i.e. add ", type=str)
    parser.add_argument("--new-name", metavar="NEW_NAME",
                        default=None,
                        help="New name for the resulting dataset. Will"
                        " default to the name of the first dataset if "
                        "not set.", type=str)
    parser.add_argument("--metadata_jsonfile", metavar="METADATA_JSONFILE",
                        default=None,
                        help="Filename for the json file containing "
                        "information for bounds expansion.", type=str)
    parser.add_argument('--warnings_on', action='store_true',
                        help="If warnings_on is set (i.e. True), "
                        "Warning messages where metadata do not match "
                        "will be given. Default=False", default=False)

    args = parser.parse_args(args=argv)

    new_metadata = cli.inputjson(args.metadata_jsonfile)
    # Load cubes
    cubelist = []
    new_cube_name = args.new_name
    for filename in args.input_filenames:
        new_cube = cli.inputcube(filename)
        cubelist.append(new_cube)
        if new_cube_name is None:
            new_cube_name = new_cube.name()
        if args.warnings_on:
            if (args.new_name is None and
                    new_cube_name != new_cube.name()):
                msg = ("Defaulting to first cube name, {} but combining with"
                       "a cube with name, {}.".format(
                            new_cube_name, new_cube.name()))
                warnings.warn(msg)

    # Process Cube
    # pylint: disable=E1123
    process(*cubelist, operation=args.operation, new_name=new_cube_name,
            new_metadata=new_metadata, warnings_on=args.warnings_on,
            output=args.output_filepath)


@cli.clizefy
@cli.with_output
def process(*cubelist: cli.inputcube,
            operation='+',
            new_name=None,
            new_metadata: cli.inputjson = None,
            warnings_on=False):
    r"""Combine input cubes.

    Combine the input cubes into a single cube using the requested operation.

    Args:
        cubelist (iris.cube.CubeList or list of iris.cube.Cube):
            An iris CubeList to be combined.
        operation (str):
            An operation to use in combining input cubes. One of:
            +, -, \*, add, subtract, multiply, min, max, mean
        new_name (str):
            New name for the resulting dataset.
        new_metadata (dict):
            Dictionary containing information on coordinates to expand.
        warnings_on (bool):
            If True, warning messages where metadata do not match will be
            given.

    Returns:
        result (iris.cube.Cube):
            Returns a cube with the combined data.
    """
    from improver.cube_combiner import CubeCombiner
    from iris.cube import CubeList
    coords_to_expand = None
    if new_metadata and 'expanded_coord' in new_metadata:
        coords_to_expand = new_metadata['expanded_coord']
    if not cubelist:
        raise TypeError("A cube is needed to be combined.")
    if new_name is None:
        new_name = cubelist[0].name()
    result = CubeCombiner(operation, warnings_on=warnings_on).process(
        CubeList(cubelist), new_name, coords_to_expand=coords_to_expand)

    return result


if __name__ == "__main__":
    main()

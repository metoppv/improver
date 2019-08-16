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

import iris

from improver.argparser import ArgParser
from improver.cube_combiner import CubeCombiner
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments for the cube combiner plugin.
    """
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
                        "required changes to the metadata. "
                        " default=None", type=str)
    parser.add_argument('--warnings_on', action='store_true',
                        help="If warnings_on is set (i.e. True), "
                        "Warning messages where metadata do not match "
                        "will be given. Default=False", default=False)

    args = parser.parse_args(args=argv)

    new_metadata = load_json_or_none(args.metadata_jsonfile)
    # Load cubes
    cubelist = iris.cube.CubeList([])
    new_cube_name = args.new_name
    for filename in args.input_filenames:
        new_cube = load_cube(filename)
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
    result = process(cubelist, args.operation, new_cube_name,
                     new_metadata, args.warnings_on)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cubelist, operation, new_cube_name,
            new_metadata=None, warnings_on=False):
    """Module for combining Cubes.

    Combine the input cubes into a single cube using the requested operation.
    e.g. '+', '-', '*', 'add', 'subtract', 'multiply', 'min', 'max', 'mean'

    Args:
        cubelist (iris.cube.CubeList):
            An iris CubeList to be combined.
        operation (str):
            "+", "-", "*", "add", "subtract", "multiply", "min", "max", "mean"
            An operation to use in combining Cubes.
        new_cube_name (str):
            New name for the resulting dataset.
        new_metadata (dict):
            Dictionary of required changes to the metadata.
            Default is None.
        warnings_on (bool):
            If True, warning messages where metadata do not match will be
            given.
            Default is False.

    Returns
        result (iris.cube.Cube):
            Returns a cube with the combined data.
    """
    # Load the metadata changes if required
    new_coords = None
    new_attr = None
    expanded_coord = None
    if new_metadata:
        if 'coordinates' in new_metadata:
            new_coords = new_metadata['coordinates']
        if 'attributes' in new_metadata:
            new_attr = new_metadata['attributes']
        if 'expanded_coord' in new_metadata:
            expanded_coord = new_metadata['expanded_coord']

    result = (
        CubeCombiner(operation, warnings_on=warnings_on).process(
            cubelist,
            new_cube_name,
            revised_coords=new_coords,
            revised_attributes=new_attr,
            expanded_coord=expanded_coord))
    return result


if __name__ == "__main__":
    main()

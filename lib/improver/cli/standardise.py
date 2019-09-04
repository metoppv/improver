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
"""Script to standardise a NetCDF file by one or more of regridding, updating
meta-data and demoting float64 data to float32"""

import warnings

import iris

from improver.argparser import ArgParser
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.cube_checker import check_cube_not_float64
from improver.utilities.cube_metadata import amend_metadata
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.spatial import RegridLandSea


def main(argv=None):
    """
    Standardise a source cube. Available options are regridding (bilinear or
    nearest-neighbour, optionally with land-mask awareness), updating meta-data
    and converting float64 data to float32. A check for float64 data compliance
    can be made by only specify a source NetCDF file with no other arguments.
    """
    parser = ArgParser(
        description='Standardise a source data cube. Three main options are '
                    'available; fixing float64 data, regridding and updating '
                    'metadata. If regridding then additional options are '
                    'available to use bilinear or nearest-neighbour '
                    '(optionally with land-mask awareness) modes. If only a '
                    'source file is specified with no other arguments, then '
                    'an exception will be raised if float64 data are found on '
                    'the source.')

    parser.add_argument('source_data_filepath', metavar='SOURCE_DATA',
                        help='A cube of data that is to be standardised and '
                             'optionally fixed for float64 data, regridded '
                             'and meta data changed')

    parser.add_argument("--output_filepath", metavar="OUTPUT_FILE",
                        default=None,
                        help="The output path for the processed NetCDF. "
                             "If only a source file is specified and no "
                             "output file, then the source will be checked"
                             "for float64 data.")

    regrid_group = parser.add_argument_group("Regridding options")
    regrid_group.add_argument(
        "--target_grid_filepath", metavar="TARGET_GRID",
        help=('If specified then regridding of the source '
              'against the target grid is enabled. If also using '
              'landmask-aware regridding, then this must be land_binary_mask '
              'data.'))

    regrid_group.add_argument(
        "--regrid_mode", default='bilinear',
        choices=['bilinear', 'nearest', 'nearest-with-mask'],
        help=('Selects which regridding technique to use. Default uses '
              'iris.analysis.Linear(); "nearest" uses Nearest() (Use for less '
              'continuous fields, e.g. precipitation.); "nearest-with-mask" '
              'ensures that target data are sourced from points with the same '
              'mask value (Use for coast-line-dependent variables like '
              'temperature).'))

    regrid_group.add_argument(
        "--extrapolation_mode", default='nanmask',
        help='Mode to use for extrapolating data into regions '
             'beyond the limits of the source_data domain. '
             'Refer to online documentation for iris.analysis. '
             'Modes are: '
             'extrapolate - The extrapolation points will '
             'take their value from the nearest source point. '
             'nan - The extrapolation points will be be '
             'set to NaN. '
             'error - A ValueError exception will be raised, '
             'notifying an attempt to extrapolate. '
             'mask  - The extrapolation points will always be '
             'masked, even if the source data is not a '
             'MaskedArray. '
             'nanmask - If the source data is a MaskedArray '
             'the extrapolation points will be masked. '
             'Otherwise they will be set to NaN. '
             'Defaults to nanmask.')

    regrid_group.add_argument(
        "--input_landmask_filepath", metavar="INPUT_LANDMASK_FILE",
        help=("A path to a NetCDF file describing the land_binary_mask on "
              "the source-grid if coastline-aware regridding is required."))

    regrid_group.add_argument(
        "--landmask_vicinity", metavar="LANDMASK_VICINITY",
        default=25000., type=float,
        help=("Radius of vicinity to search for a coastline, in metres. "
              "Default value; 25000 m"))

    parser.add_argument("--fix_float64", action='store_true', default=False,
                        help="Check and fix cube for float64 data. Without "
                             "this option an exception will be raised if "
                             "float64 data is found but no fix applied.")

    parser.add_argument("--json_file", metavar="JSON_FILE", default=None,
                        help='Filename for the json file containing required '
                             'changes that will be applied '
                             'to the metadata. Defaults to None.')

    args = parser.parse_args(args=argv)

    if args.target_grid_filepath or args.json_file or args.fix_float64:
        if not args.output_filepath:
            msg = ("An argument has been specified that requires an output "
                   "filepath but none has been provided")
            raise ValueError(msg)

    if (args.input_landmask_filepath and
            "nearest-with-mask" not in args.regrid_mode):
        msg = ("Land-mask file supplied without appropriate regrid_mode. "
               "Use --regrid_mode=nearest-with-mask.")
        raise ValueError(msg)

    if args.input_landmask_filepath and not args.target_grid_filepath:
        msg = ("Cannot specify input_landmask_filepath without "
               "target_grid_filepath")
        raise ValueError(msg)

    # Load Cube and json
    metadata_dict = load_json_or_none(args.json_file)
    # source file data path is a mandatory argument
    output_data = load_cube(args.source_data_filepath)
    target_grid = None
    source_landsea = None
    if args.target_grid_filepath:
        target_grid = load_cube(args.target_grid_filepath)
        if args.regrid_mode in ["nearest-with-mask"]:
            if not args.input_landmask_filepath:
                msg = ("An argument has been specified that requires an input "
                       "landmask filepath but none has been provided")
                raise ValueError(msg)
            source_landsea = load_cube(args.input_landmask_filepath)

    # Process Cube
    output_data = process(output_data, target_grid, source_landsea,
                          metadata_dict, args.regrid_mode,
                          args.extrapolation_mode, args.landmask_vicinity,
                          args.fix_float64)

    # Save Cube
    if args.output_filepath:
        save_netcdf(output_data, args.output_filepath)


def process(output_data, target_grid=None, source_landsea=None,
            metadata_dict=None, regrid_mode='bilinear',
            extrapolation_mode='nanmask', landmask_vicinity=25000,
            fix_float64=False):
    """Standardises a cube by one or more of regridding, updating meta-data etc

    Standardise a source cube. Available options are regridding
    (bi-linear or nearest-neighbour, optionally with land-mask
    awareness), updating meta-data and converting float64 data to
    float32. A check for float64 data compliance can be made by only
    specifying a source cube with no other arguments.

    Args:
        output_data (iris.cube.Cube):
            Output cube. If the only argument, then it is checked bor float64
            data.
        target_grid (iris.cube.Cube):
            If specified, then regridding of the source against the target
            grid is enabled. If also using landmask-aware regridding then this
            must be land_binary_mask data.
            Default is None.
        source_landsea (iris.cube.Cube):
            A cube describing the land_binary_mask on the source-grid if
            coastline-aware regridding is required.
            Default is None.
        metadata_dict (dict):
            Dictionary containing required changes that will be applied to
            the metadata.
            Default is None.
        regrid_mode (str):
            Selects which regridding techniques to use. Default uses
            iris.analysis.Linear(); "nearest" uses Nearest() (Use for less
            continuous fields, e.g precipitation.); "nearest-with-mask"
            ensures that target data are sources from points with the same
            mask value (Use for coast-line-dependant variables
            like temperature).
        extrapolation_mode (str):
            Mode to use for extrapolating data into regions beyond the limits
            of the source_data domain. Refer to online documentation for
            iris.analysis.
            Modes are -
            extrapolate -The extrapolation points will take their values
            from the nearest source point.
            nan - The extrapolation points will be set to NaN.
            error - A ValueError exception will be raised notifying an attempt
            to extrapolate.
            mask - The extrapolation points will always be masked, even if
            the source data is not a MaskedArray.
            nanmask - If the source data is a MaskedArray the extrapolation
            points will be masked. Otherwise they will be set to NaN.
            Defaults is 'nanmask'.
        landmask_vicinity (float):
            Radius of vicinity to search for a coastline, in metres.
            Defaults is 25000 m
        fix_float64 (bool):
            If True, checks and fixes cube for float64 data. Without this
            option an exception will be raised if float64 data is found but no
            fix applied.
            Default is False.

    Returns:
        output_data (iris.cube.Cube):
            Processed cube.

    Raises:
        ValueError:
            If source landsea is supplied but regrid mode not
            nearest-with-mask.
        ValueError:
            If source landsea is supplied but not target grid.
        ValueError:
            If regrid_mode is "nearest-with-mask" but no landmask cube has
            been provided.

    Warns:
        warning:
            If the 'source_landsea' did not have a cube named land_binary_mask.
        warning:
            If the 'target_grid' did not have a cube named land_binary_mask.

    """
    if (source_landsea and
            "nearest-with-mask" not in regrid_mode):
        msg = ("Land-mask file supplied without appropriate regrid_mode. "
               "Use --regrid_mode=nearest-with-mask.")
        raise ValueError(msg)

    if source_landsea and not target_grid:
        msg = ("Cannot specify input_landmask_filepath without "
               "target_grid_filepath")
        raise ValueError(msg)
    # Process
    # Re-grid with options:
    check_cube_not_float64(output_data, fix=fix_float64)
    # if a target grid file has been specified, then regrid optionally
    # applying float64 data check, metadata change, Iris nearest and
    # extrapolation mode as required.
    if target_grid:
        regridder = iris.analysis.Linear(
            extrapolation_mode=extrapolation_mode)

        if regrid_mode in ["nearest", "nearest-with-mask"]:
            regridder = iris.analysis.Nearest(
                extrapolation_mode=extrapolation_mode)

        output_data = output_data.regrid(target_grid, regridder)

        if regrid_mode in ["nearest-with-mask"]:
            if not source_landsea:
                msg = ("An argument has been specified that requires an input "
                       "landmask cube but none has been provided")
                raise ValueError(msg)

            if "land_binary_mask" not in source_landsea.name():
                msg = ("Expected land_binary_mask in input_landmask cube "
                       "but found {}".format(repr(source_landsea)))
                warnings.warn(msg)

            if "land_binary_mask" not in target_grid.name():
                msg = ("Expected land_binary_mask in target_grid cube "
                       "but found {}".format(repr(target_grid)))
                warnings.warn(msg)

            output_data = RegridLandSea(
                vicinity_radius=landmask_vicinity).process(
                output_data, source_landsea, target_grid)

        target_grid_attributes = (
            {k: v for (k, v) in target_grid.attributes.items()
             if 'mosg__' in k or 'institution' in k})
        amend_metadata(output_data, attributes=target_grid_attributes)
    # Change metadata only option:
    # if output file path and json metadata file specified,
    # change the metadata
    if metadata_dict:
        output_data = amend_metadata(output_data, **metadata_dict)

    check_cube_not_float64(output_data, fix=fix_float64)

    return output_data


if __name__ == "__main__":
    main()

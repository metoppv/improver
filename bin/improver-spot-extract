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
"""Script to run spotdata extraction."""

import json
import warnings
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError

from improver.argparser import ArgParser
from improver.ensemble_copula_coupling.ensemble_copula_coupling import \
    GeneratePercentilesFromProbabilities
from improver.percentile import PercentileConverter
from improver.spotdata.apply_lapse_rate import SpotLapseRateAdjust
from improver.spotdata.spot_extraction import SpotExtraction
from improver.spotdata.neighbour_finding import NeighbourSelection
from improver.utilities.cube_metadata import amend_metadata
from improver.utilities.cube_checker import find_percentile_coordinate
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main():
    """Load in arguments and start spotdata extraction process."""
    parser = ArgParser(
        description="Extract diagnostic data from gridded fields for spot data"
        " sites. It is possible to apply a temperature lapse rate adjustment"
        " to temperature data that helps to account for differences between"
        " the spot sites real altitude and that of the grid point from which"
        " the temperature data is extracted.")

    # Input and output files required.
    parser.add_argument("neighbour_filepath", metavar="NEIGHBOUR_FILEPATH",
                        help="Path to a NetCDF file of spot-data neighbours. "
                        "This file also contains the spot site information.")
    parser.add_argument("diagnostic_filepath", metavar="DIAGNOSTIC_FILEPATH",
                        help="Path to a NetCDF file containing the diagnostic "
                             "data to be extracted.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        help="The output path for the resulting NetCDF")

    method_group = parser.add_argument_group(
        title="Neighbour finding method",
        description="If none of these options are set, the nearest grid point "
        "to a spot site will be used without any other constraints.")
    method_group.add_argument(
        "--land_constraint", default=False, action='store_true',
        help="If set the neighbour cube will be interrogated for grid point"
        " neighbours that were identified using a land constraint. This means"
        " that the grid points should be land points except for sites where"
        " none were found within the search radius when the neighbour cube was"
        " created. May be used with minimum_dz.")
    method_group.add_argument(
        "--minimum_dz", default=False, action='store_true',
        help="If set the neighbour cube will be interrogated for grid point"
        " neighbours that were identified using a minimum height difference"
        " constraint. These are grid points that were found to be the closest"
        " in altitude to the spot site within the search radius defined when"
        " the neighbour cube was created. May be used with land_constraint.")

    percentile_group = parser.add_argument_group(
        title="Extract percentiles",
        description="Extract particular percentiles from probabilistic, "
        "percentile, or realization inputs. If deterministic input is "
        "provided a warning is raised and all leading dimensions are included "
        "in the returned spot-data cube.")
    percentile_group.add_argument(
        "--extract_percentiles", default=None, nargs='+', type=int,
        help="If set to a percentile value or a list of percentile values, "
        "data corresponding to those percentiles will be returned. For "
        "example setting '--extract_percentiles 25 50 75' will result in the "
        "25th, 50th, and 75th percentiles being returned from a cube of "
        "probabilities, percentiles, or realizations. Note that for "
        "percentile inputs, the desired percentile(s) must exist in the input "
        "cube.")
    parser.add_argument(
        "--ecc_bounds_warning", default=False, action="store_true",
        help="If True, where calculated percentiles are outside the ECC "
        "bounds range, raise a warning rather than an exception.")

    lapse_group = parser.add_argument_group(
        "Temperature lapse rate adjustment")
    lapse_group.add_argument(
        "--temperature_lapse_rate_filepath",
        help="Filepath to a NetCDF file containing temperature lapse rates. "
        "If this cube is provided, and a screen temperature cube is being "
        "processed, the lapse rates will be used to adjust the temperatures "
        "to better represent each spot's site-altitude.")

    meta_group = parser.add_argument_group("Metadata")
    meta_group.add_argument(
        "--grid_metadata_identifier", default="mosg__grid",
        help="A string (or None) to identify attributes from the input netCDF"
        " files that should be compared to ensure that the data is compatible."
        " Spot data works using grid indices, so it is important that the"
        " grids are matching or the data extracted may not match the location"
        " of the spot data sites. The default is 'mosg__grid'. If set to None"
        " no check is made; this can be used if the cubes are known to be"
        " appropriate but lack relevant metadata.")

    meta_group.add_argument(
        "--json_file", metavar="JSON_FILE", default=None,
        help="If provided, this JSON file can be used to modify the metadata "
        "of the returned netCDF file. Defaults to None.")

    output_group = parser.add_argument_group("Suppress Verbose output")
    # This CLI may be used to prepare data for verification without knowing the
    # form of the input, be it deterministic, realizations or probabilistic.
    # A warning is normally raised when attempting to extract a percentile from
    # deterministic data as this is not possible; the spot-extraction of the
    # entire cube is returned. When preparing data for verification we know
    # that we will produce a large number of these warnings when passing in
    # deterministic data. This option to suppress warnings is provided to
    # reduce the amount of unneeded logging information that is written out.

    output_group.add_argument(
        "--suppress_warnings", default=False, action="store_true",
        help="Suppress warning output. This option should only be used if "
        "it is known that warnings will be generated but they are not "
        "required.")

    args = parser.parse_args()
    neighbour_cube = load_cube(args.neighbour_filepath)
    diagnostic_cube = load_cube(args.diagnostic_filepath)

    neighbour_selection_method = NeighbourSelection(
        land_constraint=args.land_constraint,
        minimum_dz=args.minimum_dz).neighbour_finding_method_name()

    plugin = SpotExtraction(
        neighbour_selection_method=neighbour_selection_method,
        grid_metadata_identifier=args.grid_metadata_identifier)
    result = plugin.process(neighbour_cube, diagnostic_cube)

    # If a probability or percentile diagnostic cube is provided, extract
    # the given percentile if available. This is done after the spot-extraction
    # to minimise processing time; usually there are far fewer spot sites than
    # grid points.
    if args.extract_percentiles:
        try:
            perc_coordinate = find_percentile_coordinate(result)
        except CoordinateNotFoundError:
            if 'probability_of_' in result.name():
                result = GeneratePercentilesFromProbabilities(
                    ecc_bounds_warning=args.ecc_bounds_warning).process(
                        result, percentiles=args.extract_percentiles)
                result = iris.util.squeeze(result)
            elif result.coords('realization', dim_coords=True):
                fast_percentile_method = (
                    False if np.ma.isMaskedArray(result.data) else True)
                result = PercentileConverter(
                    'realization', percentiles=args.extract_percentiles,
                    fast_percentile_method=fast_percentile_method).process(
                        result)
                # This ensures the output for percentiles derived from
                # realization input looks like that derived from other inputs.
                result.coord('percentile_over_realization').rename(
                    'percentile')
                result.coord('percentile').units = '%'
            else:
                msg = ('Diagnostic cube is not a known probabilistic type. '
                       'The {} percentile could not be extracted. Extracting '
                       'data from the cube including any leading '
                       'dimensions.'.format(
                           args.extract_percentiles))
                if not args.suppress_warnings:
                    warnings.warn(msg)
        else:
            constraint = ['{}={}'.format(perc_coordinate.name(),
                                         args.extract_percentiles)]
            perc_result = extract_subcube(result, constraint)
            if perc_result is not None:
                result = perc_result
            else:
                msg = ('The percentile diagnostic cube does not contain the '
                       'requested percentile value. Requested {}, available '
                       '{}'.format(args.extract_percentiles,
                                   perc_coordinate.points))
                raise ValueError(msg)

    # Check whether a lapse rate cube has been provided and we are dealing with
    # temperature data.
    if (args.temperature_lapse_rate_filepath and
            diagnostic_cube.name() == "air_temperature"):

        lapse_rate_cube = load_cube(args.temperature_lapse_rate_filepath)
        try:
            lapse_rate_height_coord = lapse_rate_cube.coord("height")
        except (ValueError, CoordinateNotFoundError):
            msg = ("Lapse rate cube does not contain a single valued height "
                   "coordinate. This is required to ensure it is applied to "
                   "equivalent temperature data.")
            raise ValueError(msg)

        # Check the height of the temperature data matches that used to
        # calculate the lapse rates. If so, adjust temperatures using the lapse
        # rate values.
        if diagnostic_cube.coord("height") == lapse_rate_height_coord:
            plugin = SpotLapseRateAdjust(
                args.grid_metadata_identifier,
                neighbour_selection_method=neighbour_selection_method)
            result = plugin.process(result, neighbour_cube, lapse_rate_cube)
        else:
            msg = ("A lapse rate cube was provided, but the height of "
                   "the temperature data does not match that of the data used "
                   "to calculate the lapse rates. As such the temperatures "
                   "were not adjusted with the lapse rates.")
            if not args.suppress_warnings:
                warnings.warn(msg)
    elif args.temperature_lapse_rate_filepath:
        msg = ("A lapse rate cube was provided, but the diagnostic being "
               "processed is not air temperature. The lapse rate cube was "
               "not used.")
        if not args.suppress_warnings:
            warnings.warn(msg)

    # Modify final metadata as described by provided JSON file.
    if args.json_file:
        with open(args.json_file, 'r') as input_file:
            metadata_dict = json.load(input_file)
        result = amend_metadata(result, **metadata_dict)

    # Save the spot data cube
    save_netcdf(result, args.output_filepath)


if __name__ == "__main__":
    main()

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

import warnings

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError

from improver.argparser import ArgParser
from improver.ensemble_copula_coupling.ensemble_copula_coupling import \
    GeneratePercentilesFromProbabilities
from improver.percentile import PercentileConverter
from improver.spotdata.apply_lapse_rate import SpotLapseRateAdjust
from improver.spotdata.neighbour_finding import NeighbourSelection
from improver.spotdata.spot_extraction import SpotExtraction
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.cube_checker import find_percentile_coordinate
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.cube_metadata import amend_metadata
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
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
    parser.add_argument("temperature_lapse_rate_filepath",
                        metavar="LAPSE_RATE_FILEPATH", nargs='?',
                        help="(Optional) Filepath to a NetCDF file containing"
                        " temperature lapse rates. If this cube is provided,"
                        " and a screen temperature cube is being processed,"
                        " the lapse rates will be used to adjust the"
                        " temperatures to better represent each spot's"
                        " site-altitude.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        help="The output path for the resulting NetCDF")

    parser.add_argument(
        "--apply_lapse_rate_correction",
        default=False, action="store_true",
        help="If the option is set and a lapse rate cube has been "
        "provided, extracted screen temperatures will be adjusted to "
        "better match the altitude of the spot site for which they have "
        "been extracted.")

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
        "--extract_percentiles", default=None, nargs='+', type=float,
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

    meta_group = parser.add_argument_group("Metadata")
    meta_group.add_argument(
        "--metadata_json", metavar="METADATA_JSON", default=None,
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

    args = parser.parse_args(args=argv)

    # Load Cube and JSON.
    neighbour_cube = load_cube(args.neighbour_filepath)
    diagnostic_cube = load_cube(args.diagnostic_filepath)
    lapse_rate_cube = load_cube(args.temperature_lapse_rate_filepath,
                                allow_none=True)
    metadata_dict = load_json_or_none(args.metadata_json)

    # Process Cube
    result = process(neighbour_cube, diagnostic_cube, lapse_rate_cube,
                     args.apply_lapse_rate_correction, args.land_constraint,
                     args.minimum_dz, args.extract_percentiles,
                     args.ecc_bounds_warning, metadata_dict,
                     args.suppress_warnings)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(neighbour_cube, diagnostic_cube, lapse_rate_cube=None,
            apply_lapse_rate_correction=False, land_constraint=False,
            minimum_dz=False, extract_percentiles=None,
            ecc_bounds_warning=False, metadata_dict=None,
            suppress_warnings=False):
    """Module to run spot data extraction.

    Extract diagnostic data from gridded fields for spot data sites. It is
    possible to apply a temperature lapse rate adjustment to temperature data
    that helps to account for differences between the spot site's real altitude
    and that of the grid point from which the temperature data is extracted.

    Args:
        neighbour_cube (iris.cube.Cube):
            Cube of spot-data neighbours and the spot site information.
        diagnostic_cube (iris.cube.Cube):
            Cube containing the diagnostic data to be extracted.
        lapse_rate_cube (iris.cube.Cube):
            Cube containing temperature lapse rates. If this cube is provided
            and a screen temperature cube is being processed, the lapse rates
            will be used to adjust the temperature to better represent each
            spot's site-altitude.
        apply_lapse_rate_correction (bool):
            If True, and a lapse rate cube has been provided, extracted
            screen temperature will be adjusted to better match the altitude
            of the spot site for which they have been extracted.
            Default is False.
        land_constraint (bool):
            If True, the neighbour cube will be interrogated for grid point
            neighbours that were identified using a land constraint. This means
            that the grid points should be land points except for sites where
            none were found within the search radius when the neighbour cube
            was created. May be used with minimum_dz.
            Default is False.
        minimum_dz (bool):
            If True, the neighbour cube will be interrogated for grid point
            neighbours that were identified using the minimum height
            difference constraint. These are grid points that were found to be
            the closest in altitude to the spot site within the search radius
            defined when the neighbour cube was created. May be used with
            land_constraint.
            Default is False.
        extract_percentiles (list or int):
            If set to a percentile value or a list of percentile values,
            data corresponding to those percentiles will be returned. For
            example [25, 50, 75] will result in the 25th, 50th and 75th
            percentiles being returned from a cube of probabilities,
            percentiles or realizations.
            Note that for percentiles inputs, the desired percentile(s) must
            exist in the input cube.
            Default is None.
        ecc_bounds_warning (bool):
            If True, where calculated percentiles are outside the ECC bounds
            range, raises a warning rather than an exception.
            Default is False.
        metadata_dict (dict):
            If provided, this dictionary can be used to modify the metadata
            of the returned cube.
            Default is None.
        suppress_warnings (bool):
            Suppress warning output. This option should only be used if it
            is known that warnings will be generated but they are not required.
            Default is None.

    Returns:
        result (iris.cube.Cube):
           The processed cube.

    Raises:
        ValueError:
            If the percentile diagnostic cube does not contain the requested
            percentile value.
        ValueError:
            If the lapse rate cube was provided but the diagnostic being
            processed is not air temperature.
        ValueError:
            If the lapse rate cube provided does not have the name
            "air_temperature_lapse_rate"
        ValueError:
            If the lapse rate cube does not contain a single valued height
            coordinate.

    Warns:
        warning:
           If diagnostic cube is not a known probabilistic type.
        warning:
            If a lapse rate cube was provided, but the height of the
            temperature does not match that of the data used.
        warning:
            If a lapse rate cube was not provided, but the option to apply
            the lapse rate correction was enabled.

    """
    neighbour_selection_method = NeighbourSelection(
        land_constraint=land_constraint,
        minimum_dz=minimum_dz).neighbour_finding_method_name()
    plugin = SpotExtraction(
        neighbour_selection_method=neighbour_selection_method)
    result = plugin.process(neighbour_cube, diagnostic_cube)

    # If a probability or percentile diagnostic cube is provided, extract
    # the given percentile if available. This is done after the spot-extraction
    # to minimise processing time; usually there are far fewer spot sites than
    # grid points.
    if extract_percentiles is not None:
        try:
            perc_coordinate = find_percentile_coordinate(result)
        except CoordinateNotFoundError:
            if 'probability_of_' in result.name():
                result = GeneratePercentilesFromProbabilities(
                    ecc_bounds_warning=ecc_bounds_warning).process(
                    result, percentiles=extract_percentiles)
                result = iris.util.squeeze(result)
            elif result.coords('realization', dim_coords=True):
                fast_percentile_method = (
                    False if np.ma.isMaskedArray(result.data) else True)
                result = PercentileConverter(
                    'realization', percentiles=extract_percentiles,
                    fast_percentile_method=fast_percentile_method).process(
                    result)
            else:
                msg = ('Diagnostic cube is not a known probabilistic type. '
                       'The {} percentile could not be extracted. Extracting '
                       'data from the cube including any leading '
                       'dimensions.'.format(extract_percentiles))
                if not suppress_warnings:
                    warnings.warn(msg)
        else:
            constraint = ['{}={}'.format(perc_coordinate.name(),
                                         extract_percentiles)]
            perc_result = extract_subcube(result, constraint)
            if perc_result is not None:
                result = perc_result
            else:
                msg = ('The percentile diagnostic cube does not contain the '
                       'requested percentile value. Requested {}, available '
                       '{}'.format(extract_percentiles,
                                   perc_coordinate.points))
                raise ValueError(msg)
    # Check whether a lapse rate cube has been provided and we are dealing with
    # temperature data and the lapse-rate option is enabled.
    if apply_lapse_rate_correction and lapse_rate_cube:
        if not result.name() == "air_temperature":
            msg = ("A lapse rate cube was provided, but the diagnostic being "
                   "processed is not air temperature and cannot be adjusted.")
            raise ValueError(msg)

        if not lapse_rate_cube.name() == 'air_temperature_lapse_rate':
            msg = ("A cube has been provided as a lapse rate cube but does "
                   "not have the expected name air_temperature_lapse_rate: "
                   "{}".format(lapse_rate_cube.name()))
            raise ValueError(msg)

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
                neighbour_selection_method=neighbour_selection_method)
            result = plugin.process(result, neighbour_cube, lapse_rate_cube)
        elif not suppress_warnings:
            warnings.warn(
                "A lapse rate cube was provided, but the height of the "
                "temperature data does not match that of the data used "
                "to calculate the lapse rates. As such the temperatures "
                "were not adjusted with the lapse rates.")

    elif apply_lapse_rate_correction and not lapse_rate_cube:
        if not suppress_warnings:
            warnings.warn(
                "A lapse rate cube was not provided, but the option to "
                "apply the lapse rate correction was enabled. No lapse rate "
                "correction could be applied.")

    # Modify final metadata as described by provided JSON file.
    if metadata_dict:
        result = amend_metadata(result, **metadata_dict)
    # Remove the internal model_grid_hash attribute if present.
    result.attributes.pop('model_grid_hash', None)
    return result


if __name__ == "__main__":
    main()

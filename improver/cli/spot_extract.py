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

from improver import cli


@cli.clizefy
@cli.with_output
def process(neighbour_cube: cli.inputcube,
            cube: cli.inputcube,
            lapse_rate: cli.inputcube = None,
            *,
            apply_lapse_rate_correction=False,
            land_constraint=False,
            similar_altitude=False,
            extract_percentiles: cli.comma_separated_list = None,
            ignore_ecc_bounds=False,
            new_title: str = None,
            suppress_warnings=False):
    """Module to run spot data extraction.

    Extract diagnostic data from gridded fields for spot data sites. It is
    possible to apply a temperature lapse rate adjustment to temperature data
    that helps to account for differences between the spot site's real altitude
    and that of the grid point from which the temperature data is extracted.

    Args:
        neighbour_cube (iris.cube.Cube):
            Cube of spot-data neighbours and the spot site information.
        cube (iris.cube.Cube):
            Cube containing the diagnostic data to be extracted.
        lapse_rate (iris.cube.Cube):
            Optional cube containing temperature lapse rates. If this cube is
            provided and a screen temperature cube is being processed, the
            lapse rates will be used to adjust the temperature to better
            represent each spot's site-altitude.
        apply_lapse_rate_correction (bool):
            Use to apply a lapse-rate correction to screen temperature data so
            that the data are a better match the altitude of the spot site for
            which they have been extracted.
        land_constraint (bool):
            Use to select the nearest-with-land-constraint neighbour-selection
            method from the neighbour_cube. This means that the grid points
            should be land points except for sites where none were found within
            the search radius when the neighbour cube was created. May be used
            with similar_altitude.
        similar_altitude (bool):
            Use to select the nearest-with-height-constraint
            neighbour-selection method from the neighbour_cube. These are grid
            points that were found to be the closest in altitude to the spot
            site within the search radius defined when the neighbour cube was
            created. May be used with land_constraint.
        extract_percentiles (list or int):
            If set to a percentile value or a list of percentile values,
            data corresponding to those percentiles will be returned. For
            example "25, 50, 75" will result in the 25th, 50th and 75th
            percentiles being returned from a cube of probabilities,
            percentiles or realizations. Deterministic input data will raise
            a warning message.
            Note that for percentiles inputs, the desired percentile(s) must
            exist in the input cube.
        ignore_ecc_bounds (bool):
            Demotes exceptions where calculated percentiles are outside the ECC
            bounds range to warnings.
        new_title (str):
            New title for the spot-extracted data.  If None, this attribute is
            removed from the output cube since it has no prescribed standard
            and may therefore contain grid information that is no longer
            correct after spot-extraction.
        suppress_warnings (bool):
            Suppress warning output. This option should only be used if it
            is known that warnings will be generated but they are not required.

    Returns:
        iris.cube.Cube:
           Cube of spot data.

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

    import warnings

    import iris
    import numpy as np
    from iris.exceptions import CoordinateNotFoundError

    from improver.ensemble_copula_coupling.ensemble_copula_coupling import \
        ConvertProbabilitiesToPercentiles
    from improver.metadata.probabilistic import find_percentile_coordinate
    from improver.percentile import PercentileConverter
    from improver.spotdata.apply_lapse_rate import SpotLapseRateAdjust
    from improver.spotdata.neighbour_finding import NeighbourSelection
    from improver.spotdata.spot_extraction import SpotExtraction
    from improver.utilities.cube_extraction import extract_subcube

    neighbour_selection_method = NeighbourSelection(
        land_constraint=land_constraint,
        minimum_dz=similar_altitude).neighbour_finding_method_name()
    plugin = SpotExtraction(
        neighbour_selection_method=neighbour_selection_method)
    result = plugin.process(neighbour_cube, cube,
                            new_title=new_title)

    # If a probability or percentile diagnostic cube is provided, extract
    # the given percentile if available. This is done after the spot-extraction
    # to minimise processing time; usually there are far fewer spot sites than
    # grid points.
    if extract_percentiles:
        extract_percentiles = [np.float32(x) for x in extract_percentiles]
        try:
            perc_coordinate = find_percentile_coordinate(result)
        except CoordinateNotFoundError:
            if 'probability_of_' in result.name():
                result = ConvertProbabilitiesToPercentiles(
                    ecc_bounds_warning=ignore_ecc_bounds).process(
                    result, percentiles=extract_percentiles)
                result = iris.util.squeeze(result)
            elif result.coords('realization', dim_coords=True):
                fast_percentile_method = not np.ma.isMaskedArray(result.data)
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
    if apply_lapse_rate_correction and lapse_rate:
        if not result.name() == "air_temperature":
            msg = ("A lapse rate cube was provided, but the diagnostic being "
                   "processed is not air temperature and cannot be adjusted.")
            raise ValueError(msg)

        if not lapse_rate.name() == 'air_temperature_lapse_rate':
            msg = ("A cube has been provided as a lapse rate cube but does "
                   "not have the expected name air_temperature_lapse_rate: "
                   "{}".format(lapse_rate.name()))
            raise ValueError(msg)

        try:
            lapse_rate_height_coord = lapse_rate.coord("height")
        except (ValueError, CoordinateNotFoundError):
            msg = ("Lapse rate cube does not contain a single valued height "
                   "coordinate. This is required to ensure it is applied to "
                   "equivalent temperature data.")
            raise ValueError(msg)

        # Check the height of the temperature data matches that used to
        # calculate the lapse rates. If so, adjust temperatures using the lapse
        # rate values.
        if cube.coord("height") == lapse_rate_height_coord:
            plugin = SpotLapseRateAdjust(
                neighbour_selection_method=neighbour_selection_method)
            result = plugin(result, neighbour_cube, lapse_rate)
        elif not suppress_warnings:
            warnings.warn(
                "A lapse rate cube was provided, but the height of the "
                "temperature data does not match that of the data used "
                "to calculate the lapse rates. As such the temperatures "
                "were not adjusted with the lapse rates.")

    elif apply_lapse_rate_correction and not lapse_rate:
        if not suppress_warnings:
            warnings.warn(
                "A lapse rate cube was not provided, but the option to "
                "apply the lapse rate correction was enabled. No lapse rate "
                "correction could be applied.")

    # Remove the internal model_grid_hash attribute if present.
    result.attributes.pop('model_grid_hash', None)
    return result

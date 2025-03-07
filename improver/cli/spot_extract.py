#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run spotdata extraction."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    apply_lapse_rate_correction: bool = False,
    fixed_lapse_rate: float = None,
    land_constraint: bool = False,
    similar_altitude: bool = False,
    extract_percentiles: cli.comma_separated_list = None,
    ignore_ecc_bounds_exceedance: bool = False,
    skip_ecc_bounds: bool = False,
    new_title: str = None,
    suppress_warnings: bool = False,
    realization_collapse: bool = False,
    subset_coord: str = None,
):
    """Module to run spot data extraction.

    Extract diagnostic data from gridded fields for spot data sites. It is
    possible to apply a temperature lapse rate adjustment to temperature data
    that helps to account for differences between the spot site's real altitude
    and that of the grid point from which the temperature data is extracted.

    Args:
        cubes (iris.cube.Cube):
            A list of cubes containing the diagnostic data to be extracted,
            the lapse rate (optional) and the neighbour cube. Where the lapse
            rate cube contains temperature lapse rates. If this cube is
            provided and a screen temperature cube is being processed, the
            lapse rates will be used to adjust the temperature to better
            represent each spot's site-altitude.
            And the neighbour cube is a cube of spot-data neighbours and
            the spot site information.
        apply_lapse_rate_correction (bool):
            Use to apply a lapse-rate correction to screen temperature
            forecasts so that they better represent the altitude of the
            spot site for which they have been extracted. This lapse rate
            will be applied for a fixed orographic difference between the
            site and grid point altitude. Differences in orography in
            excess of this fixed limit will use the Environmental Lapse
            Rate (also known as the Standard Atmosphere Lapse Rate).
            Lapse rate adjustment cannot be applied to existing spot
            forecasts that are passed in for subsetting.
        fixed_lapse_rate (float):
            If provided, use this fixed value as a lapse-rate for adjusting
            the forecast values if apply_lapse_rate_correction is True. This
            can be used instead of providing a lapse rate cube. Value is
            given in Kelvin / metre of temperature change with ascent. For
            example a dry adiabatic lapse rate would be given as -0.0098.
            This lapse rate will be applied for a fixed orographic difference
            between the site and gridpoint altitude. Differences in orography
            in excess of this fixed limit will use the Environmental Lapse
            Rate (also known as the Standard Atmosphere Lapse Rate).
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
            Note that for percentile inputs, if the desired percentile(s) do
            not exist in the input cube the available percentiles will be
            resampled to produce those requested.
        ignore_ecc_bounds_exceedance (bool):
            Demotes exceptions where calculated percentiles are outside the ECC
            bounds range to warnings.
        skip_ecc_bounds (bool):
            If True, ECC bounds are not included when converting probabilities to
            percentiles or from one set of percentiles to another. This has the
            effect that percentiles outside of the range given by the input
            percentiles will be computed by nearest neighbour interpolation from
            the nearest available percentile, rather than using linear
            interpolation between the nearest available percentile and the ECC
            bound.
        new_title (str):
            New title for the spot-extracted data.  If None, this attribute is
            removed from the output cube since it has no prescribed standard
            and may therefore contain grid information that is no longer
            correct after spot-extraction.
        suppress_warnings (bool):
            Suppress warning output. This option should only be used if it
            is known that warnings will be generated but they are not required.
        realization_collapse (bool):
            Triggers equal-weighting blending of the realization coord if required.
            Use this if a threshold coord is also present on the input cube.
        subset_coord (str):
            If a spot cube is provided as input this plugin can return a subset of
            the sites based on the sites specified in the neighbour cube. To
            achieve this the plugin needs the name of the site ID coordinate to be
            used for matching, e.g. wmo_id. If subset_coord is not provided, and a
            spot forecast is passed in, the entire spot cube will be processed and
            returned. The neighbour selection method options have no impact if a
            spot cube is passed in.

    Returns:
        iris.cube.Cube:
           Cube of spot data.
    """
    from improver.spotdata.spot_manipulation import SpotManipulation

    return SpotManipulation(
        apply_lapse_rate_correction=apply_lapse_rate_correction,
        fixed_lapse_rate=fixed_lapse_rate,
        land_constraint=land_constraint,
        similar_altitude=similar_altitude,
        extract_percentiles=extract_percentiles,
        ignore_ecc_bounds_exceedance=ignore_ecc_bounds_exceedance,
        skip_ecc_bounds=skip_ecc_bounds,
        new_title=new_title,
        suppress_warnings=suppress_warnings,
        realization_collapse=realization_collapse,
        subset_coord=subset_coord,
    )(cubes)

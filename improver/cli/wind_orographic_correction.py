#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply unresolved-orography wind corrections."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    wind_profile_cube: cli.inputcube,
    high_res_orog_cube: cli.inputcube,
    model_orog_cube: cli.inputcube,
    model_orog_stddev_cube: cli.inputcube,
    model_silhouette_roughness_cube: cli.inputcube,
    *,
    target_height_levels: cli.comma_separated_list = None,
):
    """Apply an unresolved-orography wind-speed correction to a profile cube.

    This command fits a vertical wind-speed profile from ``wind_profile_cube``
    and applies the resulting unresolved-orography correction to the same
    wind field. The input cube is therefore used both for the profile fit and
    the corrected output.

    Args:
        wind_profile_cube:
            Wind-speed cube containing the profile to fit and the field to
            correct. It should include wind speeds from 0 to (at least) 300 m
            above ground level, as these heights are used for the logarithmic
            profile used to estimate the effective surface roughness.

        high_res_orog_cube:
            High-resolution orography cube.

        model_orog_cube:
            Model-resolution orography cube.

        model_orog_stddev_cube:
            Sub-grid orography standard deviation cube.

        model_silhouette_roughness_cube:
            Sub-grid silhouette roughness cube.

        target_height_levels:
            Comma-separated list of target heights above ground level, in
            metres. If omitted, the correction is applied at the heights
            already present on ``wind_profile_cube``. Values may be supplied as
            a comma-separated string, a Python list, or the literal ``None``;
            these are converted to a list of floating-point heights before the
            plugin is called.

    Returns:
        iris.cube.Cube:
            Corrected wind-speed cube on the selected target heights.
    """
    from improver.wind_calculations.wind_orographic_correction import (
        OrographicWindCorrection,
    )

    if target_height_levels is None:
        parsed_target_height_levels = None
    elif isinstance(target_height_levels, list):
        if (
            len(target_height_levels) == 1
            and str(target_height_levels[0]).lower() == "none"
        ):
            parsed_target_height_levels = None
        else:
            parsed_target_height_levels = [
                float(value) for value in target_height_levels
            ]
    elif isinstance(target_height_levels, str):
        if target_height_levels.lower() == "none":
            parsed_target_height_levels = None
        else:
            parsed_target_height_levels = [
                float(value) for value in target_height_levels.split(",")
            ]
    else:
        parsed_target_height_levels = target_height_levels

    plugin = OrographicWindCorrection(
        high_res_orog_cube,
        model_orog_cube,
        model_orog_stddev_cube,
        model_silhouette_roughness_cube,
    )

    return plugin(
        wind_profile_cube,
        target_height_levels=parsed_target_height_levels,
    )

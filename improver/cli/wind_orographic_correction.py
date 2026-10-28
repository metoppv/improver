#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to interpolate wind speeds and apply unresolved-orography correction."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    uncorrected_wind_cube: cli.inputcube,
    unsmoothed_orog_cube: cli.inputcube,
    model_orog_cube: cli.inputcube,
    model_orog_stddev_cube: cli.inputcube,
    model_silhouette_roughness_cube: cli.inputcube,
    *,
    target_height_levels: cli.comma_separated_list = None,
):
    """Interpolate wind speeds to target height levels (if provided) and adjust
    using a terrain-induced speed-up factor derived from unresolved-orography
    ancillaries.

    The input ``uncorrected_wind_cube`` must include height levels from the
    surface up to at least 300 m above ground level, as this lower profile is
    used to estimate the surface roughness.

    For ensemble input, each realization is corrected independently.

    Args:
        uncorrected_wind_cube:
            Cube containing the uncorrected wind-speed forecast on height
            levels. The levels must span 0 m to at least 300 m above ground
            level, since the lower profile is used to estimate effective
            surface roughness. Its height coordinate must be in metres.

        unsmoothed_orog_cube:
            Cube containing the unsmoothed (i.e. true) orography, in metres.
            This cube should be on the same grid as ``model_orog_cube``.

        model_orog_cube:
            Cube containing the smoothed (i.e. NWP model representation of)
            orography, in metres.

        model_orog_stddev_cube:
            Cube containing the sub-grid orography standard deviation, in
            metres. This cube should be on the same grid as ``model_orog_cube``.

        model_silhouette_roughness_cube:
            Cube containing the dimensionless sub-grid silhouette roughness.
            This cube should be on the same grid as ``model_orog_cube``.

        target_height_levels:
            Optional target heights in metres. If omitted, the heights
            already present on ``uncorrected_wind_cube`` are used.

    Note:
        All orography inputs must be provided on grids consistent with the
        forecast data. If the forecast and model orography data have been
        re-gridded from their native NWP model grid (e.g. using StaGE), the
        ancillary inputs should also be re-gridded to the same grid. The
        orography and standard-deviation cubes must use metres; silhouette
        roughness is dimensionless.

    Returns:
        iris.cube.Cube:
            Wind-speed cube after unresolved-orography correction on the
            selected target heights.
    """
    from improver.wind_calculations.wind_orographic_correction import (
        OrographicWindCorrection,
    )

    plugin = OrographicWindCorrection(
        unsmoothed_orog_cube,
        model_orog_cube,
        model_orog_stddev_cube,
        model_silhouette_roughness_cube,
    )

    return plugin(
        uncorrected_wind_cube,
        target_height_levels=_parse_target_height_levels(target_height_levels),
    )


def _parse_target_height_levels(
    target_height_levels: list[float | str] | str | None,
) -> list[float] | None:
    """Convert CLI target height levels to a list of floats.

    Args:
        target_height_levels:
            Target heights above ground level, in metres, specified as a list
            of numbers or numeric strings, or a comma-separated string.
            ``None``, the case-insensitive string ``"none"``, or a list
            containing only that string selects the input cube heights.

    Returns:
        list[float] | None:
            Target heights as floats, or ``None`` to use the input cube heights.
    """
    if target_height_levels is None:
        return None

    if isinstance(target_height_levels, str):
        if target_height_levels.lower() == "none":
            return None
        values = target_height_levels.split(",")
    else:
        if (
            len(target_height_levels) == 1
            and str(target_height_levels[0]).lower() == "none"
        ):
            return None
        values = target_height_levels

    return [float(value) for value in values]

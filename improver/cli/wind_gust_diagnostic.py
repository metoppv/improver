#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to create wind-gust data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    wind_gust: cli.inputcube,
    wind_speed: cli.inputcube,
    *,
    wind_gust_percentile: float = 50.0,
    wind_speed_percentile: float = 95.0,
):
    """Create a cube containing the wind_gust diagnostic.

    Calculate revised wind-gust data using a specified percentile of
    wind-gust data and a specified percentile of wind-speed data through the
    WindGustDiagnostic plugin. The wind-gust diagnostic will be the max of the
    specified percentile data.

    Args:
        wind_gust (iris.cube.Cube):
            Cube containing one or more percentiles of wind_gust data.
        wind_speed (iris.cube.Cube):
            Cube containing one or more percentiles of wind_speed data.
        wind_gust_percentile (float):
            Percentile value required from wind-gust cube.
        wind_speed_percentile (float):
            Percentile value required from wind-speed cube.

    Returns:
        iris.cube.Cube:
            Cube containing the wind-gust diagnostic data.
    """
    from improver.wind_calculations.wind_gust_diagnostic import WindGustDiagnostic

    result = WindGustDiagnostic(wind_gust_percentile, wind_speed_percentile)(
        wind_gust, wind_speed
    )
    return result

#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to create wind-gust data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    wind_speed: cli.inputcube,
    *,
    period: float = 24.0,
    coords: cli.comma_separated_list = ['realization', 'time'],
    start_hour: float = 0.0,
):
    """Create a cube containing the "typical" wind speed over 24 hours
    using the fast percentile method for collapsing the time coordinate
    and to create the percentiles. It alters the metadata to be
    consistent e.g. change to "percentile" from "percentile_over_time"
    or "percentile_over_time_realization. The cube is then renamed 
    typical wind speed.

    Args:
        wind_speed (iris.cube.Cube):
            A wind speed cube containing a time coordinate.
        period (float):
            The period that we want to calculate the "typical" wind speed
            over.
        coords (list of str):
            The coordinates that we would like to collapse to obtain the
            typical wind speed over 24 hours.
        start_hour (float):
            The hour that the 24 hour periods should start from.
        
    Returns:
        iris.cube.Cube:
            A cube containing the "typical" wind data over 24 hours that
        has been created. This includes overlapping periods.
    """
    from improver.utilities.cube_manipulation import create_period_cubes
   
    result = create_period_cubes(wind_speed, period)
    result.rename("typical_wind_speed")
    return result

#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate mean wind direction from ensemble realizations."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wind_direction: cli.inputcube, *, backup_method="neighbourhood"):
    """Calculates mean wind direction from ensemble realization.

    Create a cube containing the wind direction averaged over the ensemble
    realizations.

    Args:
        wind_direction (iris.cube.Cube):
            Cube containing the wind direction from multiple ensemble
            realizations.
        backup_method (str):
            Backup method to use if the complex numbers approach has low
            confidence.
            "neighbourhood" (default) recalculates using the complex numbers
            approach with additional realization extracted from neighbouring
            grid points from all available realizations.
            "first_realization" uses the value of realization zero, and should
            only be used with global lat-lon data.

    Returns:
        iris.cube.Cube:
            Cube containing the wind direction averaged from the ensemble
            realizations.
    """
    from improver.wind_calculations.wind_direction import WindDirection

    result = WindDirection(backup_method=backup_method)(wind_direction)
    return result

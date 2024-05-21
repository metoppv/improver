#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply a scaling factor to account for a correction linked to the
difference in altitude between the grid point and the site location."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    scaling_factor: cli.inputcube,
    *,
    site_id_coord: str = "wmo_id",
):
    """Apply a scaling factor to account for a correction linked to the difference
    in altitude between the grid point and the site location.

    Args:
        forecast (iris.cube.Cube):
            Percentile forecasts.
        rescaling_cube (iris.cube.Cube):
            Multiplicative scaling factor to adjust the percentile forecasts.
            This cube is expected to contain multiple values for the forecast_period
            and forecast_reference_time_hour coordinates. The most appropriate
            forecast period and forecast reference_time_hour pair within the
            rescaling cube are chosen using the forecast reference time hour from
            the forecast and the nearest forecast period that is greater than or
            equal to the forecast period of the forecast. However, if the forecast
            period of the forecast exceeds all forecast periods within the rescaling
            cube, the scaling factor from the maximum forecast period is used.
            This cube is generated using the estimate_dz_rescaling CLI.
        site_id_coord (str):
            The name of the site ID coordinate. This defaults to 'wmo_id'.

    Returns:
        iris.cube.Cube:
            Percentile forecasts that have been rescaled to account for a difference
            in altitude between the grid point and the site location.
    """

    from improver.calibration.dz_rescaling import ApplyDzRescaling

    return ApplyDzRescaling(site_id_coord=site_id_coord)(forecast, scaling_factor)

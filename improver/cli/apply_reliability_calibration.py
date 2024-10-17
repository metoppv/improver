#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply reliability calibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    reliability_table: cli.inputcubelist = None,
    point_by_point: bool = False,
):
    """
    Calibrate a probability forecast using the provided reliability calibration
    table. This calibration is designed to improve the reliability of
    probability forecasts without significantly degrading their resolution. If
    a reliability table is not provided, the input forecast is returned
    unchanged.

    The method implemented here is described in Flowerdew J. 2014. Calibrating
    ensemble reliability whilst preserving spatial structure. Tellus, Ser. A
    Dyn. Meteorol. Oceanogr. 66.

    Args:
        forecast (iris.cube.Cube):
            The forecast to be calibrated.
        reliability_table (iris.cube.Cube or iris.cube.CubeList):
            The reliability calibration table to use in calibrating the
            forecast. If input is a CubeList the CubeList should contain
            separate cubes for each threshold in the forecast cube.
        point_by_point:
            Whether to process each point in the input cube independently.
            Please note this option is memory intensive and is unsuitable
            for gridded input

    Returns:
        iris.cube.Cube:
            Calibrated forecast.
    """
    from improver.calibration.reliability_calibration import ApplyReliabilityCalibration

    if reliability_table is None:
        return forecast
    plugin = ApplyReliabilityCalibration(point_by_point=point_by_point)
    return plugin(forecast, reliability_table)

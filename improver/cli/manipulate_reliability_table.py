#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to manipulate a reliability table cube."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    reliability_table: cli.inputcube,
    *,
    minimum_forecast_count: int = 200,
    point_by_point: bool = False,
):
    """
    Manipulate a reliability table to ensure sufficient sample counts in
    as many bins as possible by combining bins with low sample counts.
    Also enforces a monotonic observation frequency.

    Args:
        reliability_table (iris.cube.Cube):
            The reliability table that needs to be manipulated after the
            spatial dimensions have been aggregated.
        minimum_forecast_count (int):
            The minimum number of forecast counts in a forecast probability
            bin for it to be used in calibration.
            The default value of 200 is that used in Flowerdew 2014.
        point_by_point:
            Whether to process each point in the input cube independently.
            Please note this option is memory intensive and is unsuitable
            for gridded input

    Returns:
        iris.cube.CubeList:
            The reliability table that has been manipulated to ensure
            sufficient sample counts in each probability bin and a monotonic
            observation frequency.
            The cubelist contains a separate cube for each threshold in
            the original reliability table.
    """
    from improver.calibration.reliability_calibration import ManipulateReliabilityTable

    plugin = ManipulateReliabilityTable(
        minimum_forecast_count=minimum_forecast_count, point_by_point=point_by_point,
    )
    return plugin(reliability_table)

#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to aggregate reliability tables."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, coordinates: cli.comma_separated_list = None):
    """Aggregate reliability tables.

    Aggregate multiple reliability calibration tables and/or aggregate over
    coordinates within the table(s) to produce a new reliability calibration
    table.

    Args:
        cubes (list of iris.cube.Cube):
            The cube or cubes containing the reliability calibration tables
            to aggregate.
        coordinates (list):
            A list of coordinates over which to aggregate the reliability
            calibration table using summation. If the list is empty
            and a single cube is provided, this cube will be returned
            unchanged.
    Returns:
        iris.cube.Cube:
            Aggregated reliability table.
    """
    from improver.calibration.reliability_calibration import (
        AggregateReliabilityCalibrationTables,
    )

    return AggregateReliabilityCalibrationTables()(cubes, coordinates=coordinates)

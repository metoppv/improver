#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to relabel a diagnostic as a period diagnostic."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, *, period: int = None):
    """Relabel a diagnostic as a period diagnostic.

    Modify an existing diagnostic to represent a period. This will either
    relabel an instantaneous diagnostic to be a period diagnostic, or
    modify a period diagnostic to have a different period. This may be
    useful when trying to combine instantaneous and period diagnostics.

    Args:
        cube (iris.cube.Cube):
            The cube for a diagnostic that will be modified to represent the
            required period.
        period (int):
            The period in hours.

    Returns:
        iris.cube.Cube:
            Cube with metadata updated to represent the required period.

    """
    from improver.utilities.temporal import relabel_to_period

    return relabel_to_period(cube, period)

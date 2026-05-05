#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to select which subperiods contain the phenomenon identified over the main period."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    main_period_cube: cli.inputcube,
    subperiod_cube: cli.inputcube,
    percentile: float = 50.0,
    **threshold_kwargs: cli.inputjson,
):
    """
    Select which subperiods contain the phenomenon identified over the main period.

    For example, if the 50th percentile of hours of light rain over a 24 hour period is 0.25 (6 hours),
    then this plugin can be used to identify which 6 hours of the 24 hour period are most likely
    to contain light rain. The result can be used in the weather symbol decision tree to force
    the selection of a wet symbol.

    Args:
        main_period_cube (iris.cube.Cube):
            Cube containing the main period diagnostic, with a percentile coordinate and one or more threshold coordinates.
        subperiod_cube (iris.cube.Cube):
            Cube containing the subperiod diagnostic, with a time coordinate and one or more threshold coordinates that match those on the main period cube.
        percentile (float):
            The percentile of the main period diagnostic to select.
        **threshold_kwargs:
            Keyword arguments specifying the names and values of threshold coords associated with the main period diagnostic to select. One of these will also match the threshold coord on the subperiod diagnostic, which will be used to identify which subperiods to select.

    Returns:
        iris.cube.Cube
    """
    from improver.categorical.subperiod_selector import SubperiodSelector

    return SubperiodSelector(percentile=percentile, **threshold_kwargs)(
        main_period_cube, subperiod_cube
    )

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
    *subperiod_cubes: cli.inputcube,
    percentile: float = 50.0,
    new_name: str = "selected_subperiods",
    model_id_attr: str = None,
    record_run_attr: str = None,
    threshold_kwargs: cli.inputjson = None,
):
    """
    Deaggregate a fraction-of-period-that-is-xxx diagnostic into this-subperiod-is-xxx.

    For example, if light rain is expected for 6 hours within a 24 hour period (e.g. the 50th
    percentile of light rain over a 24 hour period is 0.25), this plugin selects the 6 hours
    most likely to contain that light rain. The result can be used in the weather symbol
    decision tree to force the selection of a wet symbol.

    Args:
        main_period_cube (iris.cube.Cube):
            Cube containing the main period diagnostic, with a percentile coordinate and one or more threshold coordinates.
        subperiod_cubes (iris.cube.CubeList):
            Cubes containing the subperiod diagnostic, with a time coordinate and one or more threshold coordinates that match those on the main period cube.
        percentile (float):
            The percentile of the main period diagnostic to select.
        new_name (str):
            Name of output cube.
        model_id_attr (str):
            Name of attribute recording source models to be copied to the
            output cube.
        record_run_attr:
            Name of attribute used to record models and cycles to be copied to
            the output cube.
        threshold_kwargs:
            Keyword arguments specifying the names and values of threshold coords associated with the main period
            diagnostic to select. One of these will also match the threshold coord on the subperiod diagnostic,
            which will be used to identify which subperiods to select.

    Returns:
        iris.cube.Cube:
            A cube of subperiods marked as 1 (is) or 0 (is not) representative of the phenomenon.
    """
    from improver.categorical.subperiod_selector import SubperiodSelector
    from improver.utilities.cube_manipulation import MergeCubes

    # Merge the subperiod cubes ready for use in the SubperiodSelector plugin.
    subperiod_cube = MergeCubes()(subperiod_cubes)

    return SubperiodSelector(
        percentile=percentile,
        new_name=new_name,
        model_id_attr=model_id_attr,
        record_run_attr=record_run_attr,
        **threshold_kwargs,
    )(main_period_cube, subperiod_cube)

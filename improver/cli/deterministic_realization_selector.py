#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to select the deterministic realization"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    target_realization_number: int = 0,
    attribute="primary_input_realizations_to_clusters",
):
    """Extract a specific realization from a forecast cube using a cluster cube's
    attribute: "primary_input_realizations_to_clusters".

    Args:
        cubes:
            A list of two cubes containing a forecast and a cluster cube.
            The cluster cube will contain the attribute:
            "primary_input_realizations_to_clusters".
            This will be used to split the forecasts and cluster cube and
            determine which realizations to extract from the forecast cube.
        target_realization_number:
            The number of the realization of interest. Default value = 0.
        attribute:
            The attribute of the cluster cube used to identify target realization,
            and it's associated cluster.
            Default value = "primary_input_realizations_to_clusters".

    Returns:
        output_cube:
            Forecast cube containing only the target realization.
    """
    from improver.utilities.deterministic_realization_selector import (
        DeterministicRealizationSelector,
    )

    output_cube = DeterministicRealizationSelector(
        target_realization_number=target_realization_number, attribute=attribute
    )(cubes)

    return output_cube

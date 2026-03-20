#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to select the Deterministic Realization"""

from improver import cli

@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, control_member: int = 0):
    """CLI Wrapper with uses the DeterministicRealizationSelection_Plugin.
    The plugin takes forecasts and a cluster cube with the attribute:
    primary_input_realizations_to_clusters.
    Then, extracts only the realization which contains the control member.
    Then returns the subsetted Iris Cube. Description of the CLI
    Args:
        cubes:
            A list of cubes containing forecasts and a cluster cube.
            The cluster cube will contain the attribute:
            "primary_input_realizations_to_clusters".
            This will be used to split the forecasts and cluster cube and
            determine which realizations to extract from the forecast cube.
        control_member:
            The number of the ensemble member acting as the control member.
            Default value = 0.

        Returns:
            output_cube:
                Forecast cube containing,
                 only the realization with the control member.
        """
    from improver.utilities.deterministic_realization_selector import (
        DeterministicRealizationSelector
    )

    output_cube = DeterministicRealizationSelector(
        control_member=control_member)(cubes)

    return output_cube
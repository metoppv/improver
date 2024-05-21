# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Interface to significant_phase_mask."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, phase: str, *, model_id_attr: str = None):
    """
    Make phase-mask cube for the specified phase.

    Args:
        cube (iris.cube.Cube):
            The input snow-fraction data to derive the phase mask from.
        phase (str):
            One of "rain", "sleet" or "snow". This is the phase mask that will be
            returned.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.
    """
    from improver.psychrometric_calculations.significant_phase_mask import (
        SignificantPhaseMask,
    )

    return SignificantPhaseMask(model_id_attr=model_id_attr)(cube, phase)

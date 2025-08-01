#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Script to run beta recalibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    recalibration_config: cli.inputjson,
):
    """Runs probability recalibration.

    Recalibrate probability forecast using CDF of beta distribution.

    Args:
        cube (iris.cube.Cube):
            Probability cube to be recalibrated.
        recalibration_config (dict):
            Dictionary from which to interpolate parameters of
            beta distribution. Dictionary format is as specified in
            improver.blending.recalibrate.Recalibrate

    Returns:
        iris.cube.Cube:
            Recalibrated cube
    """
    from improver.calibration.beta_recalibration import BetaRecalibrate

    return BetaRecalibrate(recalibration_config)(cube)

# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to generate freezing rain probabilities."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """
    Calculates a probability of freezing-rain near the ground using rain, sleet,
    and temperature probabilities.

    P(freezing_rain rate or accumulation > threshold) = (
        (P(rain rate or accumulation >  threshold) +
         P(sleet rate or accumulation >  threshold)) * P(temperature < 0C)

    If the input data is multi-realization, the realization coordinate is
    collapsed as part of this calculation to yield an ensemble average
    probability of freezing rain.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of rain, sleet, and temperature probabilities. The
            precipitation cubes may be either rates or accumulations. The
            temperature should be a surface or screen temperature. It may be a
            period minimum, with a period that matches the precipitation
            accumulation period, or an instantaneous temperature if using
            precipitation rates.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A cube of freezing rain rate or accumulation probabilities.

    """
    from iris.cube import CubeList

    from improver.precipitation_type.freezing_rain import FreezingRain

    return FreezingRain(model_id_attr=model_id_attr)(CubeList(cubes))

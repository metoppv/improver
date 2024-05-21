# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run GenerateClearSkySolarRadiation ancillary generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    target_grid: cli.inputcube,
    surface_altitude: cli.inputcube = None,
    linke_turbidity: cli.inputcube = None,
    *,
    time: cli.inputdatetime,
    accumulation_period: int,
    temporal_spacing: int = 30,
    new_title: str = None,
):
    """Generate a cube containing clearsky solar radiation data, evaluated on the target grid
    for the specified time and accumulation period. Accumulated clearsky solar radiation is used
    as an input to the RainForests calibration for rainfall.

    Args:
        target_grid (iris.cube.Cube):
            A cube containing the desired spatial grid.
        surface_altitude (iris.cube.Cube):
            Surface altitude data, specified in metres, used in the evaluation of the clearsky
            solar irradiance values. If not provided, a cube with constant value 0.0 m is used,
            created from target_grid.
        linke_turbidity (iris.cube.Cube):
            Linke turbidity data used in the evaluation of the clearsky solar irradiance
            values. Linke turbidity is a dimensionless quantity that accounts for the
            atmospheric scattering of radiation due to aerosols and water vapour, relative
            to a dry atmosphere. If not provided, a cube with constant value 3.0 is used,
            created from target_grid.
        time (str):
            A datetime specified in the format YYYYMMDDTHHMMZ at which to evaluate the
            accumulated clearsky solar radiation. This time is taken to be the end of
            the accumulation period.
        accumulation_period (int):
            The number of hours over which the solar radiation accumulation is defined.
        temporal_spacing (int):
            The time stepping, specified in mins, used in the integration of solar irradiance
            to produce the accumulated solar radiation.
        new_title:
            New title for the output cube attributes. If None, this attribute is left out
            since it has no prescribed standard.

    Returns:
        iris.cube.Cube:
            A cube containing accumulated clearsky solar radiation.
    """
    from improver.generate_ancillaries.generate_derived_solar_fields import (
        GenerateClearskySolarRadiation,
    )

    return GenerateClearskySolarRadiation()(
        target_grid,
        time,
        accumulation_period,
        surface_altitude=surface_altitude,
        linke_turbidity=linke_turbidity,
        temporal_spacing=temporal_spacing,
        new_title=new_title,
    )

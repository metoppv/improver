#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate orographic enhancement."""


from improver import cli


def extract_and_check(cube, height_value, units):
    """
    Function to attempt to extract a height level.
    If no matching level is available an error is raised.

    Args:
        cube (cube):
            Cube to be extracted from and checked it worked.
        height_value (float):
            The boundary height to be extracted with the input units.
        units (str):
            The units of the height level to be extracted.
    Returns:
        iris.cube.Cube:
            A cube containing the extracted height level.
    Raises:
        ValueError: If height level is not found in the input cube.
    """
    from improver.utilities.cube_extraction import extract_subcube

    # Write constraint in this format so a constraint is constructed that
    # is suitable for floating point comparison
    height_constraint = [
        "height=[{}:{}]".format(height_value - 0.1, height_value + 0.1)
    ]
    cube = extract_subcube(cube, height_constraint, units=[units])

    if cube is not None:
        return cube

    raise ValueError("No data available at height {}{}".format(height_value, units))


@cli.clizefy
@cli.with_output
def process(
    temperature: cli.inputcube,
    humidity: cli.inputcube,
    pressure: cli.inputcube,
    wind_speed: cli.inputcube,
    wind_direction: cli.inputcube,
    orography: cli.inputcube,
    *,
    boundary_height: float = 1000.0,
    boundary_height_units="m",
):
    """Calculate orographic enhancement

    Uses the ResolveWindComponents() and OrographicEnhancement() plugins.
    Outputs data on the high resolution orography grid.

    Args:
        temperature (iris.cube.Cube):
             Cube containing temperature at top of boundary layer.
        humidity (iris.cube.Cube):
            Cube containing relative humidity at top of boundary layer.
        pressure (iris.cube.Cube):
            Cube containing pressure at top of boundary layer.
        wind_speed (iris.cube.Cube):
            Cube containing wind speed values.
        wind_direction (iris.cube.Cube):
            Cube containing wind direction values relative to true north.
        orography (iris.cube.Cube):
            Cube containing height of orography above sea level on high
            resolution (1 km) UKPP domain grid.
        boundary_height (float):
            Model height level to extract variables for calculating orographic
            enhancement, as proxy for the boundary layer.
        boundary_height_units (str):
            Units of the boundary height specified for extracting model levels.

    Returns:
        iris.cube.Cube:
            Precipitation enhancement due to orography on the high resolution
            input orography grid.
    """
    from improver.orographic_enhancement import OrographicEnhancement
    from improver.wind_calculations.wind_components import ResolveWindComponents

    constraint_info = (boundary_height, boundary_height_units)

    temperature = extract_and_check(temperature, *constraint_info)
    humidity = extract_and_check(humidity, *constraint_info)
    pressure = extract_and_check(pressure, *constraint_info)
    wind_speed = extract_and_check(wind_speed, *constraint_info)
    wind_direction = extract_and_check(wind_direction, *constraint_info)

    # resolve u and v wind components
    u_wind, v_wind = ResolveWindComponents()(wind_speed, wind_direction)
    # calculate orographic enhancement
    return OrographicEnhancement()(
        temperature, humidity, pressure, u_wind, v_wind, orography
    )

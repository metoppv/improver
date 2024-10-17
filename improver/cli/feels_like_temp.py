#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run the feels like temperature plugin."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    temperature: cli.inputcube,
    wind_speed: cli.inputcube,
    relative_humidity: cli.inputcube,
    pressure: cli.inputcube,
    *,
    model_id_attr: str = None,
):
    """Calculates the feels like temperature using the data in the input cube.

    Calculate the feels like temperature using a combination of the wind chill
    index and Steadman's apparent temperature equation with the following
    method:

    If temperature < 10 degrees C: The feels like temperature is equal to the
    wind chill.

    If temperature > 20 degrees C: The feels like temperature is equal to the
    apparent temperature.

    If 10 <= temperature <= degrees C: A weighting (alpha) is calculated in
    order to blend between the wind chill and the apparent temperature.

    Args:
        temperature (iris.cube.Cube):
            Cube of air temperatures at screen level
        wind_speed (iris.cube.Cube):
            Cube of wind speed at 10m
        relative_humidity (iris.cube.Cube):
            Cube of relative humidity at screen level
        pressure (iris.cube.Cube):
            Cube of surface pressure
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of feels like temperature. The units of feels like temperature
            will be the same as the units of temperature cube when it is input
            into the function.
    """
    from improver.feels_like_temperature import calculate_feels_like_temperature

    return calculate_feels_like_temperature(
        temperature,
        wind_speed,
        relative_humidity,
        pressure,
        model_id_attr=model_id_attr,
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Script to run the feels like temperature plugin."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(temperature: cli.inputcube,
            wind_speed: cli.inputcube,
            relative_humidity: cli.inputcube,
            pressure: cli.inputcube,
            *,
            model_id_attr: str = None):
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
            Cube of mean sea level pressure
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of feels like temperature. The units of feels like temperature
            will be the same as the units of temperature cube when it is input
            into the function.
    """
    from improver.feels_like_temperature import (
        calculate_feels_like_temperature)
    return calculate_feels_like_temperature(
        temperature, wind_speed, relative_humidity, pressure,
        model_id_attr=model_id_attr)

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
"""Script to create wind-gust data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wind_gust: cli.inputcube,
            wind_speed: cli.inputcube,
            *,
            wind_gust_percentile: float = 50.0,
            wind_speed_percentile: float = 95.0):
    """Create a cube containing the wind_gust diagnostic.

    Calculate revised wind-gust data using a specified percentile of
    wind-gust data and a specified percentile of wind-speed data through the
    WindGustDiagnostic plugin. The wind-gust diagnostic will be the max of the
    specified percentile data.

    Args:
        wind_gust (iris.cube.Cube):
            Cube containing one or more percentiles of wind_gust data.
        wind_speed (iris.cube.Cube):
            Cube containing one or more percentiles of wind_speed data.
        wind_gust_percentile (float):
            Percentile value required from wind-gust cube.
        wind_speed_percentile (float):
            Percentile value required from wind-speed cube.

    Returns:
        iris.cube.Cube:
            Cube containing the wind-gust diagnostic data.
    """
    from improver.wind_calculations.wind_gust_diagnostic import (
        WindGustDiagnostic)

    result = WindGustDiagnostic(
        wind_gust_percentile, wind_speed_percentile)(wind_gust, wind_speed)
    return result

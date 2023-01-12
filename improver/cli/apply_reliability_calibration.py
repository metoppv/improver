#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""CLI to apply reliability calibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    reliability_table: cli.inputcubelist = None,
    point_by_point: bool = False,
):
    """
    Calibrate a probability forecast using the provided reliability calibration
    table. This calibration is designed to improve the reliability of
    probability forecasts without significantly degrading their resolution. If
    a reliability table is not provided, the input forecast is returned
    unchanged.

    The method implemented here is described in Flowerdew J. 2014. Calibrating
    ensemble reliability whilst preserving spatial structure. Tellus, Ser. A
    Dyn. Meteorol. Oceanogr. 66.

    Args:
        forecast (iris.cube.Cube):
            The forecast to be calibrated.
        reliability_table (iris.cube.Cube or iris.cube.CubeList):
            The reliability calibration table to use in calibrating the
            forecast. If input is a CubeList the CubeList should contain
            separate cubes for each threshold in the forecast cube.
        point_by_point:
            Whether to process each point in the input cube independently.
            Please note this option is memory intensive and is unsuitable
            for gridded input

    Returns:
        iris.cube.Cube:
            Calibrated forecast.
    """
    from improver.calibration.reliability_calibration import ApplyReliabilityCalibration

    if reliability_table is None:
        return forecast
    plugin = ApplyReliabilityCalibration(point_by_point=point_by_point)
    return plugin(forecast, reliability_table)

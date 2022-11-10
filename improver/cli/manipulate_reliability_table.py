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
"""CLI to manipulate a reliability table cube."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    reliability_table: cli.inputcube,
    *,
    minimum_forecast_count: int = 200,
    point_by_point: bool = False,
):
    """
    Manipulate a reliability table to ensure sufficient sample counts in
    as many bins as possible by combining bins with low sample counts.
    Also enforces a monotonic observation frequency.

    Args:
        reliability_table (iris.cube.Cube):
            The reliability table that needs to be manipulated after the
            spatial dimensions have been aggregated.
        minimum_forecast_count (int):
            The minimum number of forecast counts in a forecast probability
            bin for it to be used in calibration.
            The default value of 200 is that used in Flowerdew 2014.
        point_by_point:
            Whether to process each point in the input cube independently.
            Please note this option is memory intensive and is unsuitable
            for gridded input

    Returns:
        iris.cube.CubeList:
            The reliability table that has been manipulated to ensure
            sufficient sample counts in each probability bin and a monotonic
            observation frequency.
            The cubelist contains a separate cube for each threshold in
            the original reliability table.
    """
    from improver.calibration.reliability_calibration import ManipulateReliabilityTable

    plugin = ManipulateReliabilityTable(
        minimum_forecast_count=minimum_forecast_count, point_by_point=point_by_point,
    )
    return plugin(reliability_table)

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
"""CLI to manipulate reliability tables."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            method=None,
            coordinates: cli.comma_separated_list = None):
    """Manipulate reliability tables.

    Aggregate multiple reliability calibration tables and/or aggregate over
    coordinates within the table(s) to produce a new reliability calibration
    table.

    Args:
        cubes (list of iris.cube.Cube):
            The cube or cubes containing the reliability calibration tables
            to aggregate.
        method (iris.analysis.WeightedAggregator):
            The aggregator method to use in aggregating multiple
            reliability calibration tables or aggregating over coordinates.
            Summation is the default behaviour.
        coordinates (list):
            A list of coordinates over which to aggregate the reliability
            calibration table using the chosen aggregator. If the list is empty
            and a single cube is provided, this cube will be returned
            unchanged.
    Returns:
        iris.cube.Cube:
            Aggregated reliability table.
    """
    from improver.calibration.reliability_calibration import (
        ManipulateReliabilityCalibrationTables)

    return ManipulateReliabilityCalibrationTables(
        method=method).process(cubes, coordinates=coordinates)
